// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#ifdef __APPLE__

#import <Cocoa/Cocoa.h>

#include "Open3DViewer.h"

#include "Open3D/GUI/Application.h"
#include "Open3D/GUI/Native.h"
#include "Open3D/Utility/FileSystem.h"

// ----------------------------------------------------------------------------
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@interface AppDelegate ()
{
    bool open_empty_window_;
}
@property (retain) NSTimer *timer;
@end

@implementation AppDelegate
- (id)init {
    if ([super init]) {
        open_empty_window_ = true;
    }
    return self;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // -application:openFile: runs befure applicationDidFinishLaunching: so we
    // need to check if we loaded a file or we need to display an empty window.
    if (open_empty_window_) {
        LoadAndCreateWindow("");
    }
}

// Called by [NSApp run] if the user passes command line arguments (which may
// be multiple times if multiple files are given), or the app is launched by
// double-clicking a file in the Finder, or by dropping files onto the app
// either in the Finder or (more likely) onto the Dock icon. It is also called
// after launching if the user double-clicks a file in the Finder or drops
// a file onto the app icon and the application is already launched.
- (BOOL)application:(NSApplication *)sender openFile:(NSString *)filename {
    open_empty_window_ = false;  // LoadAndCreateWindow() always opens a window
    LoadAndCreateWindow(filename.UTF8String);
    return YES;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    // The app will terminate after this function exits. This will result
    // in the Application object in main() getting destructed, but it still
    // thinks it is running. So tell Application to quit, which will post
    // the required events to the event loop to properly clean up.
    open3d::gui::Application::GetInstance().OnTerminate();
}
@end

// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    // If we double-clicked the app, the CWD gets set to /, so change that
    // to the user's home directory so that file dialogs act reasonably.
    if (open3d::utility::filesystem::GetWorkingDirectory() == "/") {
        std::string homedir = NSHomeDirectory().UTF8String;
        open3d::utility::filesystem::ChangeWorkingDirectory(homedir);
    }

    open3d::gui::Application::GetInstance().Initialize(argc, argv);

    // Note: if NSApp is created (which happens in +sharedApplication)
    //       then GLFW will use our NSApp with our delegate instead of its
    //       own delegate that doesn't have the openFile and terminate
    //       selectors.

    // Ideally we could do the following:
    //@autoreleasepool {
    //    AppDelegate *delegate = [[AppDelegate alloc] init];
    //    NSApplication *application = [NSApplication sharedApplication];
    //    [application setDelegate:delegate];
    //    [NSApp run];
    //}
    // But somewhere along the line GLFW seems to clean up the autorelease
    // pool, which then causes a crash when [NSApp run] finishes and the
    // autorelease pool cleans up at the '}'. To avoid that, we will not
    // autorelease things. That creates a memory leak, but we're exiting
    // after that so it does not matter.
    AppDelegate *delegate = [[AppDelegate alloc] init];
    NSApplication *application = [NSApplication sharedApplication];
    [application setDelegate:delegate];
    // ---- [NSApp run] equivalent ----
    // https://www.cocoawithlove.com/2009/01/demystifying-nsapplication-by.html
    [NSApp finishLaunching];
    open3d::gui::Application::GetInstance().Run();
    // ----
}

#endif // __APPLE__
