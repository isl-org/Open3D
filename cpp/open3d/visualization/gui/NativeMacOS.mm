// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Native.h"

#import <ApplicationServices/ApplicationServices.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA 1
#include <GLFW/glfw3native.h>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace visualization {
namespace gui {

void MacTransformIntoApp() {
    // This is Deep Magic. Some versions of Python (for instance, MacPorts) do
    // not properly activate the python process when a window is created,
    // despite being created as a Framework. This results in the window being
    // visible, but it cannot get a menubar and it never gets text focus.
    // Some more official attempts that do NOT work are:
    // * ensure NSApp is created (by use of [NSApplication sharedApplication])
    // * [NSApp finishLaunching]
    // * [NSApp activateIgnoringOtherApps:NO] (YES also doesn't work)
    // * [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular]
    //     (the activation policy already appears to be Regular)
    // * Calling -makeKeyAndOrderFront on the NSWindow (GLFW surely does this
    //     anyway)
    // So, we come to the Deep Magic. ApplicationServices contains some
    // functions that were part of Carbon, and expanded on them. Most have been
    // deprecated and moved into NSRunnableApplication, but TransformProcessType
    // does not appear to have been. Documention is limited to the Processes.h
    // header file and various digitally-dusty archival tomes. One particularly
    // salient archive is
    // http://svn.python.org/projects/external/tk-8.5.11.0/macosx/tkMacOSXInit.c
    // (which appears to have changed in upstream Tk) wherein a comment says
    // that TransformProcessType() needs to be called if we aren't in a bundle.
    // Absent any concrete documentation, we chant the incantation and hope that
    // any side-effects or age-related effects are minimal.
    ProcessSerialNumber psn = { 0, kCurrentProcess };
    TransformProcessType(&psn, kProcessTransformToForegroundApplication);
}

void* GetNativeDrawable(GLFWwindow* glfw_window) {
    NSWindow* win = glfwGetCocoaWindow(glfw_window);
    NSView* view = [win contentView];
    return view;
}

void PostNativeExposeEvent(GLFWwindow* glfw_window) {
    NSWindow* win = glfwGetCocoaWindow(glfw_window);
    // You'd think there would be no way that needsDisplay == YES, especially
    // when we are within a draw, but you'd be wrong. In particular, it seems
    // to happen when clicking an OK button in a dialog or the inc/dec buttons
    // in an integer NumericEdit while running the UI in Python.
    // If the view already needDisplay, then nothing will happen, so post it
    // to the queue, which gives this draw call time to finish and get whatever
    // is causing the problem cleared up. (It's unclear to me why needsDisplay
    // would be YES if we actually within a draw, but there we are...)
    // This situation does not seem to get triggered by the viewer app.
    if ([win contentView].needsDisplay == YES) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [win contentView].needsDisplay = YES;
        });
    } else {
        [win contentView].needsDisplay = YES;
    }
}

void ShowNativeAlert(const char *message) {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:[NSString stringWithUTF8String:message]];
    [alert runModal];
}

/*void* SetupMetalLayer(void* nativeView) {
    NSView* view = (NSView*) nativeView;
    [view setWantsLayer:YES];
    CAMetalLayer* metalLayer = [CAMetalLayer layer];
    metalLayer.bounds = view.bounds;

    // It's important to set the drawableSize to the actual backing pixels. When rendering
    // full-screen, we can skip the macOS compositor if the size matches the display size.
    metalLayer.drawableSize = [view convertSizeToBacking:view.bounds.size];

    // This is set to NO by default, but is also important to ensure we can bypass the compositor
    // in full-screen mode
    // See "Direct to Display" http://metalkit.org/2017/06/30/introducing-metal-2.html.
    metalLayer.opaque = YES;

    [view setLayer:metalLayer];

    return metalLayer;
}

void* ResizeMetalLayer(void* nativeView) {
    NSView* view = (NSView*) nativeView;
    CAMetalLayer* metalLayer = (CAMetalLayer*)view.layer;
    metalLayer.drawableSize = [view convertSizeToBacking:view.bounds.size];
    return metalLayer;
}
*/

void SetNativeMenubar(void* menubar) {
    NSMenu *menu = (NSMenu*)menubar;
    NSApplication.sharedApplication.mainMenu = menu;
}

void ShowNativeFileDialog(FileDialog::Mode type,
                          const std::string& path,
                          const std::vector<std::pair<std::string, std::string>>& filters,
                          std::function<void(const char*)> on_ok,
                          std::function<void()> on_cancel) {
    NSSavePanel *dlg; // NSOpenPanel inherits from NSSavePanel, oddly enough
    if (type == FileDialog::Mode::OPEN) {
        NSOpenPanel* open_dlg = [NSOpenPanel openPanel];
        open_dlg.allowsMultipleSelection = NO;
        open_dlg.canChooseDirectories = NO;
        open_dlg.canChooseFiles = YES;
        dlg = open_dlg;
    } else if (type == FileDialog::Mode::OPEN_DIR) {
        NSOpenPanel *open_dlg = [NSOpenPanel openPanel];
        open_dlg.allowsMultipleSelection = NO;
        open_dlg.canChooseDirectories = YES;
        open_dlg.canChooseFiles = NO;
        dlg = open_dlg;
    } else {
        dlg = [NSSavePanel savePanel];
    }
    dlg.directoryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];

    if (type != FileDialog::Mode::OPEN_DIR) {
        if (!filters.empty()) {  // [NSMutableArray arrayWidthCapacity:0] returns nil
            NSMutableArray *allowed = [NSMutableArray arrayWithCapacity:2 * filters.size()];
            for (auto &f : filters) {
                if (f.first.empty() || f.first == "*.*") {
                    continue;
                }
                std::vector<std::string> exts = utility::SplitString(f.first, ", ");
                for (std::string ext : exts) {  // ext is a copy; might modify it
                    if (ext[0] == '.') {  // macOS assumes the dot in the extension
                        ext = ext.substr(1);
                    }
                    [allowed addObject:[NSString stringWithUTF8String:ext.c_str()]];
                }
            }
            dlg.allowedFileTypes = allowed;
        }
        dlg.allowsOtherFileTypes = YES;
    } else {
        dlg.allowsOtherFileTypes = NO;
    }

    NSWindow *current = NSApp.mainWindow;
    [dlg beginWithCompletionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            on_ok(dlg.URL.path.UTF8String);
        } else {
            on_cancel();
        }
        [current makeKeyWindow];
    }];
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
