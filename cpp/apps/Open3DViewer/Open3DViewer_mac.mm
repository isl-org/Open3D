// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef __APPLE__

#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>

#include "Open3DViewer.h"

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Dialog.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/Native.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/visualizer/GuiVisualizer.h"

// ----------------------------------------------------------------------------
using namespace open3d::visualization::gui;

class Open3DVisualizer : public open3d::visualization::GuiVisualizer {
    using Super = GuiVisualizer;
public:
    Open3DVisualizer()
        : open3d::visualization::GuiVisualizer("Open3D", WIDTH, HEIGHT) {
        AddItemsToAppMenu({{"Make Default 3D Viewer", MAC_MAKE_DEFAULT_APP}});
    }

protected:
    static constexpr Menu::ItemId MAC_MAKE_DEFAULT_APP = 100;

    void OnMenuItemSelected(Menu::ItemId item_id) override {
        if (item_id == MAC_MAKE_DEFAULT_APP) {
            auto em = GetTheme().font_size;
            auto dlg = std::make_shared<Dialog>("Make Open3D default");

            auto cancel = std::make_shared<Button>("Cancel");
            cancel->SetOnClicked([this]() { this->CloseDialog(); });

            auto ok = std::make_shared<Button>("Make Default");
            ok->SetOnClicked([this]() {
                // This will set the users personal default to use Open3D for
                // the file types below. THIS SHOULD ONLY BE CALLED
                // AFTER THE USER EXPLICITLY CONFIRMS THAT THEY WANT TO DO THIS!
                CFStringRef open3dBundleId = (__bridge CFStringRef)@"com.isl-org.open3d.Open3D";
                // The UTIs should match what we declare in Info.plist
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.gl-transmission-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.gl-binary-transmission-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.geometry-definition-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.object-file-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.point-cloud-library-file",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.polygon-file-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.3d-points-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.standard-tesselated-geometry-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.xyz-points-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.xyzn-points-format",
                    kLSRolesAll, open3dBundleId);
                LSSetDefaultRoleHandlerForContentType(
                    (__bridge CFStringRef)@"public.xyzrgb-points-format",
                    kLSRolesAll, open3dBundleId);

                this->CloseDialog();
            });

            auto vert = std::make_shared<Vert>(0, Margins(em));
            vert->AddChild(std::make_shared<Label>(
                "This will make Open3D the default application for the "
                "following file types:"));
            vert->AddFixed(em);
            auto table = std::make_shared<VGrid>(2, 0, Margins(em, 0, 0, 0));
            table->AddChild(std::make_shared<Label>("Mesh:"));
            table->AddChild(std::make_shared<Label>(".gltf, .glb, .obj, .off, .ply, .stl"));
            table->AddChild(std::make_shared<Label>("Point clouds:"));
            table->AddChild(std::make_shared<Label>(".pcd, .ply, .pts, .xyz, .xyzn, .xyzrgb"));
            vert->AddChild(table);
            vert->AddFixed(em);
            auto buttons = std::make_shared<Horiz>(0.5 * em);
            buttons->AddStretch();
            buttons->AddChild(cancel);
            buttons->AddChild(ok);
            vert->AddChild(buttons);
            dlg->AddChild(vert);
            ShowDialog(dlg);
        } else {
            Super::OnMenuItemSelected(item_id);
        }
    }
};

constexpr Menu::ItemId Open3DVisualizer::MAC_MAKE_DEFAULT_APP;  // for Xcode

// ----------------------------------------------------------------------------
static void LoadAndCreateWindow(const char *path) {
    auto vis = std::make_shared<Open3DVisualizer>();
    bool is_path_valid = (path && path[0] != '\0');
    if (is_path_valid) {
        vis->LoadGeometry(path);
    }
    Application::GetInstance().AddWindow(vis);
}

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
    // -application:openFile: runs before applicationDidFinishLaunching: so we
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
    Application::GetInstance().OnTerminate();
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

    Application::GetInstance().Initialize(argc, argv);

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
    Application::GetInstance().Run();
    // ----
}

#endif // __APPLE__
