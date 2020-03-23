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

#include "Application.h"

#include "Button.h"
#include "Events.h"
#include "Label.h"
#include "Layout.h"
#include "Native.h"
#include "Theme.h"
#include "Window.h"

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"

#include <GLFW/glfw3.h>

#include <chrono>
#include <thread>
#include <unordered_set>

namespace {

const double RUNLOOP_DELAY_SEC = 0.010;

std::string findResourcePath(int argc, const char *argv[]) {
    std::string argv0;
    if (argc != 0 && argv) {
        argv0 = argv[0];
    }

    // Convert backslash (Windows) to forward slash
    for (auto &c : argv0) {
        if (c == '\\') {
            c = '/';
        }
    }

    // Chop off the process name
    auto lastSlash = argv0.rfind("/");
    auto path = argv0.substr(0, lastSlash);

    if (argv0[0] == '/' ||
        (argv0.size() > 3 && argv0[1] == ':' && argv0[2] == '/')) {
        // is absolute path, we're done
    } else {
        // relative path:  prepend working directory
        auto cwd = open3d::utility::filesystem::GetWorkingDirectory();
        path = cwd + "/" + path;
    }

#ifdef __APPLE__
    if (path.rfind("MacOS") == path.size() - 5) {  // path is in a bundle
        return path.substr(0, path.size() - 5) + "Resources";
    }
#endif  // __APPLE__

    auto rsrcPath = path + "/resources";
    if (!open3d::utility::filesystem::DirectoryExists(rsrcPath)) {
        return path + "/../resources";  // building with Xcode
    }
    return rsrcPath;
}

}  // namespace

namespace open3d {
namespace gui {

/*void PostWindowEvent(Window *w, SDL_WindowEventID type) {
    SDL_Event e;
    e.type = SDL_WINDOWEVENT;
    e.window.windowID = w->GetID();
    e.window.event = type;
    SDL_PushEvent(&e);
}
*/
struct Application::Impl {
    std::string resourcePath;
    Theme theme;
    bool isGLFWinitalized = false;
    bool isRunning = false;
    bool shouldQuit = false;

    std::shared_ptr<Menu> menubar;
    std::unordered_set<std::shared_ptr<Window>> windows;

    void InitGFLW() {
        if (this->isGLFWinitalized) {
            return;
        }

        glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE); // no auto-create menubar
        glfwInit();
        this->isGLFWinitalized = true;
    }
};

Application &Application::GetInstance() {
    static Application gApp;
    return gApp;
}

void Application::ShowMessageBox(const char *title, const char *message) {
    utility::LogInfo("%s", message);

    auto alert = std::make_shared<Window>("Alert", Window::FLAG_TOPMOST);
    auto em = alert->GetTheme().fontSize;
    auto layout = std::make_shared<Vert>(em, Margins(em));
    auto msg = std::make_shared<Label>(message);
    auto ok = std::make_shared<Button>("Ok");
    ok->SetOnClicked([alert = alert.get() /*avoid shared_ptr cycle*/]() {
        Application::GetInstance().RemoveWindow(alert);
    });
    layout->AddChild(Horiz::MakeCentered(msg));
    layout->AddChild(Horiz::MakeCentered(ok));
    alert->AddChild(layout);
    Application::GetInstance().AddWindow(alert);
}

Application::Application() : impl_(new Application::Impl()) {
    Color highlightColor(0.5, 0.5, 0.5);

    // Note that any values here need to be scaled by the scale factor in Window
    impl_->theme.fontPath =
            "Roboto-Medium.ttf";     // full path will be added in Initialize()
    impl_->theme.fontSize = 16;      // 1 em (font size is em in digital type)
    impl_->theme.defaultMargin = 8;  // 0.5 * em
    impl_->theme.defaultLayoutSpacing = 6;  // 0.333 * em

    impl_->theme.backgroundColor = Color(0.175, 0.175, 0.175);
    impl_->theme.textColor = Color(0.875, 0.875, 0.875);
    impl_->theme.borderWidth = 1;
    impl_->theme.borderRadius = 3;
    impl_->theme.borderColor = Color(0.5, 0.5, 0.5);
    impl_->theme.menubarBorderColor = Color(0.25, 0.25, 0.25);
    impl_->theme.buttonColor = Color(0.4, 0.4, 0.4);
    impl_->theme.buttonHoverColor = Color(0.6, 0.6, 0.6);
    impl_->theme.buttonActiveColor = Color(0.5, 0.5, 0.5);
    impl_->theme.checkboxBackgroundOffColor = Color(0.333, 0.333, .333);
    impl_->theme.checkboxBackgroundOnColor = highlightColor;
    impl_->theme.checkboxBackgroundHoverOffColor = Color(0.5, 0.5, 0.5);
    impl_->theme.checkboxBackgroundHoverOnColor =
            highlightColor.Lightened(0.15);
    impl_->theme.checkboxCheckColor = Color(1, 1, 1);
    impl_->theme.comboboxBackgroundColor = Color(0.4, 0.4, 0.4);
    impl_->theme.comboboxHoverColor = Color(0.5, 0.5, 0.5);
    impl_->theme.comboboxArrowBackgroundColor = highlightColor;
    impl_->theme.sliderGrabColor = Color(0.666, 0.666, 0.666);
    impl_->theme.textEditBackgroundColor = Color(0.25, 0.25, 0.25);
    impl_->theme.tabInactiveColor = impl_->theme.buttonColor;
    impl_->theme.tabHoverColor = impl_->theme.buttonHoverColor;
    impl_->theme.tabActiveColor = impl_->theme.buttonActiveColor;
    impl_->theme.dialogBorderWidth = 1;
    impl_->theme.dialogBorderRadius = 10;

    visualization::EngineInstance::SelectBackend(
            filament::backend::Backend::OPENGL);

    // Init GLFW here so that we can create windows before running
    impl_->InitGFLW();
}

Application::~Application() {}

void Application::Initialize() {
    // We don't have a great way of getting the process name, so let's hope that
    // the current directory is where the resources are located. This is a
    // safe assumption when running on macOS and Windows normally.
    auto path = open3d::utility::filesystem::GetWorkingDirectory();
    // Copy to C string, as some implementations of std::string::c_str()
    // return a very temporary pointer.
    char *argv = strdup(path.c_str());
    Initialize(1, (const char **)&argv);
    free(argv);
}

void Application::Initialize(int argc, const char *argv[]) {
    impl_->resourcePath = findResourcePath(argc, argv);
    impl_->theme.fontPath = impl_->resourcePath + "/" + impl_->theme.fontPath;
}

std::shared_ptr<Menu> Application::GetMenubar() const { return impl_->menubar; }

void Application::SetMenubar(std::shared_ptr<Menu> menubar) {
    auto old = impl_->menubar;
    impl_->menubar = menubar;
    // If added or removed menubar, the size of the window's content region
    // may have changed (in not on macOS), so need to relayout.
    if ((!old && menubar) || (old && !menubar)) {
        for (auto w : impl_->windows) {
            w->OnResize();
        }
    }

#if defined(__APPLE__)
    auto *native = menubar->GetNativePointer();
    if (native) {
        SetNativeMenubar(native);
    }
#endif  // __APPLE__
}

void Application::AddWindow(std::shared_ptr<Window> window) {
    window->OnResize();  // so we get an initial resize
    window->Show();
    impl_->windows.insert(window);
}

void Application::RemoveWindow(Window *window) {
    for (auto it = impl_->windows.begin(); it != impl_->windows.end(); ++it) {
        if (it->get() == window) {
            impl_->windows.erase(it);
            break;
        }
    }

    if (impl_->windows.empty()) {
        impl_->shouldQuit = true;
    }
}

void Application::Quit() {
    while (!impl_->windows.empty()) {
        RemoveWindow(impl_->windows.begin()->get());
    }
}

void Application::OnMenuItemSelected(Menu::ItemId itemId) {
    for (auto w : impl_->windows) {
        if (w->IsActiveWindow()) {
            w->OnMenuItemSelected(itemId);
            // This is a menu selection that came from a native menu.
            // We need to draw twice to ensure that any new dialog
            // that the menu item may have displayed is properly laid out.
            // (ImGUI can take up to two iterations to fully layout)
            // If we post two expose events they get coalesced, but
            // setting needsLayout forces two (for the reason given above).
            w->SetNeedsLayout();
            Window::UpdateAfterEvent(w.get());
            return;
        }
    }
}

void Application::Run() {
    while (RunOneTick())
        ;
}

bool Application::RunOneTick() {
    // Initialize if we have not started yet
    if (!impl_->isRunning) {
        // Verify that the resource path is valid. If it is not, display a
        // message box (std::cerr may not be visible to the user, if we were run
        // as app).
        if (impl_->resourcePath.empty()) {
            ShowNativeAlert(
                    "Internal error: Application::Initialize() was not called");
            return false;
        }
        if (!utility::filesystem::DirectoryExists(impl_->resourcePath)) {
            std::stringstream err;
            err << "Could not find resource directory:\n'"
                << impl_->resourcePath << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }
        if (!utility::filesystem::FileExists(impl_->theme.fontPath)) {
            std::stringstream err;
            err << "Could not load UI font:\n'" << impl_->theme.fontPath
                << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }

        // We already called this in the constructor, but it is possible
        // (but unlikely) that the run loop finished and is starting again.
        impl_->InitGFLW();

        impl_->isRunning = true;
    }

    // Process the events that have queued up
    auto status = ProcessQueuedEvents();

    // Cleanup if we are done
    if (status == RunStatus::DONE) {
        glfwTerminate();
        impl_->isGLFWinitalized = false;
        impl_->isRunning = false;
    }

    return impl_->isRunning;
}

Application::RunStatus Application::ProcessQueuedEvents() {
    glfwWaitEventsTimeout(RUNLOOP_DELAY_SEC);
    if (impl_->shouldQuit) {
        return RunStatus::DONE;
    }
    return RunStatus::CONTINUE;
}

const char *Application::GetResourcePath() const {
    return impl_->resourcePath.c_str();
}

const Theme &Application::GetTheme() const { return impl_->theme; }

}  // namespace gui
}  // namespace open3d
