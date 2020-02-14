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

#include "Native.h"

#include "Open3D/Utility/Helper.h"

#include <SDL_syswm.h>

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

namespace open3d {
namespace gui {

void* GetNativeDrawable(SDL_Window* sdlWindow) {
    SDL_SysWMinfo wmi;
    SDL_VERSION(&wmi.version);
    SDL_GetWindowWMInfo(sdlWindow, &wmi);
    NSWindow* win = wmi.info.cocoa.window;
    NSView* view = [win contentView];
    return view;
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

void ShowNativeFileDialog(FileDialog::Type type,
                          const std::string& path,
                          const std::vector<std::pair<std::string, std::string>>& filters,
                          std::function<void(const char*)> onOk,
                          std::function<void()> onCancel) {
    NSSavePanel *dlg; // NSOpenPanel inherits from NSSavePanel, oddly enough
    if (type == FileDialog::Type::OPEN) {
        NSOpenPanel *openDlg = [NSOpenPanel openPanel];
        openDlg.allowsMultipleSelection = NO;
        openDlg.canChooseDirectories = NO;
        openDlg.canChooseFiles = YES;
        dlg = openDlg;
    } else {
        dlg = [NSSavePanel savePanel];
    }
    dlg.directoryURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];

    NSMutableArray *allowed = [NSMutableArray arrayWithCapacity:2 * filters.size()];
    for (auto &f : filters) {
        if (f.first.empty() || f.first == "*.*") {
            continue;
        }
        std::vector<std::string> exts;
        utility::SplitString(exts, f.first, ", ");
        for (auto &ext : exts) {
            if (ext[0] == '.') {  // macOS assumes the dot in the extension
                ext = ext.substr(1);
            }
            [allowed addObject:[NSString stringWithUTF8String:ext.c_str()]];
        }
    }
    dlg.allowedFileTypes = allowed;
    dlg.allowsOtherFileTypes = YES;

    [dlg beginWithCompletionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            onOk(dlg.URL.path.UTF8String);
        } else {
            onCancel();
        }
    }];
}

} // gui
} // open3d
