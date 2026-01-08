// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Application.h"

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // so APIENTRY gets defined and GLFW doesn't define it
#endif                // _MSC_VER

#include <algorithm>
#include <chrono>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "open3d/geometry/Image.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/GLFWWindowSystem.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/Native.h"
#include "open3d/visualization/gui/Task.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderToBuffer.h"
#include "open3d/visualization/utility/GLHelper.h"

namespace {

const double RUNLOOP_DELAY_SEC = 0.010;

std::string FindResourcePath(int argc, const char *argv[]) {
    namespace o3dfs = open3d::utility::filesystem;
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
    auto last_slash = argv0.rfind("/");
    auto path = argv0.substr(0, last_slash);

    if (argv0[0] == '/' ||
        (argv0.size() > 3 && argv0[1] == ':' && argv0[2] == '/')) {
        // is absolute path, we're done
    } else {
        // relative path:  prepend working directory
        auto cwd = o3dfs::GetWorkingDirectory();
#ifdef __APPLE__
        // When running an app from the command line with the full relative
        // path (e.g. `bin/Open3D.app/Contents/MacOS/Open3D`), the working
        // directory can be set to the resources directory, in which case
        // a) we are done, and b) cwd + / + argv0 is wrong.
        if (cwd.rfind("/Contents/Resources") == cwd.size() - 19) {
            return cwd;
        }
#endif  // __APPLE__
        path = cwd + "/" + path;
    }

#ifdef __APPLE__
    if (path.rfind("MacOS") == path.size() - 5) {  // path is in a bundle
        return path.substr(0, path.size() - 5) + "Resources";
    }
#endif  // __APPLE__

    for (auto &subpath :
         {"/resources", "/../resources" /*building with Xcode */,
          "/share/resources" /* GNU */, "/share/Open3D/resources" /* GNU */}) {
        open3d::utility::LogInfo("Checking for resources in {}",
                                 path + subpath);
        if (o3dfs::DirectoryExists(path + subpath)) {
            return path + subpath;
        }
    }
    open3d::utility::LogError("Could not find resource directory.");
    return "";
}

}  // namespace

namespace open3d {
namespace visualization {
namespace gui {

struct Application::Impl {
    bool is_initialized_ = false;
    std::shared_ptr<WindowSystem> window_system_;
    std::vector<FontDescription> fonts_;
    Theme theme_;
    double last_time_ = 0.0;
    bool is_ws_initialized_ = false;
    bool is_running_ = false;
    bool should_quit_ = false;

    std::shared_ptr<Menu> menubar_;
    std::unordered_set<std::shared_ptr<Window>> windows_;
    std::unordered_set<std::shared_ptr<Window>> windows_to_be_destroyed_;

    std::list<Task> running_tasks_;  // always accessed from main thread
    // ----
    struct Posted {
        Window *window;
        std::function<void()> f;

        Posted(Window *w, std::function<void()> func) : window(w), f(func) {}
    };
    std::mutex posted_lock_;
    std::vector<Posted> posted_;
    // ----

    void InitWindowSystem() {
        if (!window_system_) {
            window_system_ = std::make_shared<GLFWWindowSystem>();
        }

        if (!is_ws_initialized_) {
            window_system_->Initialize();
            is_ws_initialized_ = true;
        }
    }

    void PrepareForRunning() {
        // We already called this in the constructor, but it is possible
        // that the run loop finished and is starting again.
        InitWindowSystem();

        // Initialize rendering
        // NOTE: This is where we used to manually select a backend for Filament
        // but we feel comfortable now allowing Filament to select the best
        // backend for the platform so there's no code here anymore
    }

    void CleanupAfterRunning() {
        // Aside from general tidiness in shutting down rendering,
        // failure to do this causes the Python module to hang on
        // Windows. (Specifically, if a widget is has been assigned a
        // Python function as a callback, the Python interpreter will
        // not delete the objects, the Window's destructor will not be
        // called, and the Filament threads will not stop, causing the
        // Python process to remain running even after execution of the
        // script finishes.
        visualization::rendering::EngineInstance::DestroyInstance();

        if (window_system_) {
            window_system_->Uninitialize();
        }
        is_ws_initialized_ = false;
    }
};

constexpr FontId Application::DEFAULT_FONT_ID;  // already assigned in header

Application &Application::GetInstance() {
    static Application g_app;
    return g_app;
}

void Application::ShowMessageBox(const char *title, const char *message) {
    utility::LogInfo("{}", message);

    auto alert = std::make_shared<Window>((title ? title : "Alert"),
                                          Window::FLAG_TOPMOST);
    auto em = alert->GetTheme().font_size;
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
    Color highlight_color(0.5, 0.5, 0.5);

    // Note that any values here need to be scaled by the scale factor in Window
    impl_->theme_.font_path =
            "Roboto-Medium.ttf";  // full path will be added in Initialize()
    impl_->theme_.font_bold_path = "Roboto-Bold.ttf";
    impl_->theme_.font_italic_path = "Roboto-MediumItalic.ttf";
    impl_->theme_.font_bold_italic_path = "Roboto-BoldItalic.ttf";
    impl_->theme_.font_mono_path = "RobotoMono-Medium.ttf";
    impl_->theme_.font_size = 16;      // 1 em (font size is em in digital type)
    impl_->theme_.default_margin = 8;  // 0.5 * em
    impl_->theme_.default_layout_spacing = 6;  // 0.333 * em

    impl_->theme_.background_color = Color(0.175f, 0.175f, 0.175f);
    impl_->theme_.text_color = Color(0.875f, 0.875f, 0.875f);
    impl_->theme_.border_width = 1;
    impl_->theme_.border_radius = 3;
    impl_->theme_.border_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.menubar_border_color = Color(0.25f, 0.25f, 0.25f);
    impl_->theme_.button_color = Color(0.4f, 0.4f, 0.4f);
    impl_->theme_.button_hover_color = Color(0.6f, 0.6f, 0.6f);
    impl_->theme_.button_active_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.button_on_color = Color(0.7f, 0.7f, 0.7f);
    impl_->theme_.button_on_hover_color = Color(0.9f, 0.9f, 0.9f);
    impl_->theme_.button_on_active_color = Color(0.8f, 0.8f, 0.8f);
    impl_->theme_.button_on_text_color = Color(0, 0, 0);
    impl_->theme_.checkbox_background_off_color = Color(0.333f, 0.333f, .333f);
    impl_->theme_.checkbox_background_on_color = highlight_color;
    impl_->theme_.checkbox_background_hover_off_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.checkbox_background_hover_on_color =
            highlight_color.Lightened(0.15f);
    impl_->theme_.checkbox_check_color = Color(0.9f, 0.9f, 0.9f);
    impl_->theme_.radiobtn_background_off_color = Color(0.333f, 0.333f, .333f);
    impl_->theme_.radiobtn_background_on_color = highlight_color;
    impl_->theme_.radiobtn_background_hover_off_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.radiobtn_background_hover_on_color =
            highlight_color.Lightened(0.15f);
    impl_->theme_.toggle_background_off_color =
            impl_->theme_.checkbox_background_off_color;
    impl_->theme_.toggle_background_on_color = Color(0.666f, 0.666f, 0.666f);
    impl_->theme_.toggle_background_hover_off_color =
            impl_->theme_.checkbox_background_hover_off_color;
    impl_->theme_.toggle_background_hover_on_color =
            impl_->theme_.toggle_background_on_color.Lightened(0.15f);
    impl_->theme_.toggle_thumb_color = Color(1, 1, 1);
    impl_->theme_.combobox_background_color = Color(0.4f, 0.4f, 0.4f);
    impl_->theme_.combobox_hover_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.combobox_arrow_background_color = highlight_color;
    impl_->theme_.slider_grab_color = Color(0.666f, 0.666f, 0.666f);
    impl_->theme_.text_edit_background_color = Color(0.1f, 0.1f, 0.1f);
    impl_->theme_.list_background_color = Color(0.1f, 0.1f, 0.1f);
    impl_->theme_.list_hover_color = Color(0.6f, 0.6f, 0.6f);
    impl_->theme_.list_selected_color = Color(0.5f, 0.5f, 0.5f);
    impl_->theme_.tree_background_color = impl_->theme_.list_background_color;
    impl_->theme_.tree_selected_color = impl_->theme_.list_selected_color;
    impl_->theme_.tab_inactive_color = impl_->theme_.button_color;
    impl_->theme_.tab_hover_color = impl_->theme_.button_hover_color;
    impl_->theme_.tab_active_color = impl_->theme_.button_active_color;
    impl_->theme_.dialog_border_width = 1;
    impl_->theme_.dialog_border_radius = 10;
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
    Initialize(FindResourcePath(argc, argv).c_str());
}

void Application::Initialize(const char *resource_path) {
    // Prepare for running so that we can create windows. Note that although
    // Application may be initialized, GLFW/Filament may not be, if we finished
    // Run() and are calling again.
    impl_->PrepareForRunning();

    if (impl_->is_initialized_) {
        return;
    }

    rendering::EngineInstance::SetResourcePath(resource_path);
    std::string uiblit_path = std::string(resource_path) + "/ui_blit.filamat";
    if (!utility::filesystem::FileExists(uiblit_path)) {
        utility::LogError(
                "Resource directory does not have Open3D resources: {}",
                resource_path);
    }

    impl_->theme_.font_path = std::string(resource_path) + std::string("/") +
                              impl_->theme_.font_path;
    impl_->theme_.font_bold_path = std::string(resource_path) +
                                   std::string("/") +
                                   impl_->theme_.font_bold_path;
    impl_->theme_.font_italic_path = std::string(resource_path) +
                                     std::string("/") +
                                     impl_->theme_.font_italic_path;
    impl_->theme_.font_bold_italic_path = std::string(resource_path) +
                                          std::string("/") +
                                          impl_->theme_.font_bold_italic_path;
    impl_->theme_.font_mono_path = std::string(resource_path) +
                                   std::string("/") +
                                   impl_->theme_.font_mono_path;
    if (impl_->fonts_.empty()) {
        AddFont(FontDescription(FontDescription::SANS_SERIF,
                                FontStyle::NORMAL));
    }
    impl_->is_initialized_ = true;
}

void Application::VerifyIsInitialized() {
    if (impl_->is_initialized_) {
        return;
    }

    // Call LogWarning() first because it is easier to visually parse than the
    // error message.
    utility::LogWarning("gui::Initialize() was not called");

    // It would be nice to make this LogWarning() and then call Initialize(),
    // but Python scripts requires a different heuristic for finding the
    // resource path than C++.
    utility::LogError(
            "gui::Initialize() must be called before creating a window or UI "
            "element.");
}

bool Application::UsingNativeWindows() const {
    auto os_ws =
            std::dynamic_pointer_cast<GLFWWindowSystem>(impl_->window_system_);
    return (os_ws != nullptr);
}

WindowSystem &Application::GetWindowSystem() const {
    return *impl_->window_system_;
}

void Application::SetWindowSystem(std::shared_ptr<WindowSystem> ws) {
    if (impl_->window_system_ != nullptr) {
        utility::LogError("Cannot set WindowSystem. It is already set.");
    }
    impl_->window_system_ = ws;
    impl_->is_ws_initialized_ = false;
}

FontId Application::AddFont(const FontDescription &fd) {
    FontId id = impl_->fonts_.size();
    impl_->fonts_.push_back(fd);
    SetFont(id, fd);  // make sure paths get update properly
    return id;
}

void Application::SetFont(FontId id, const FontDescription &fd) {
    auto GetSansSerifPath = [this](FontStyle style) {
        switch (style) {
            case FontStyle::BOLD:
                return impl_->theme_.font_bold_path;
            case FontStyle::ITALIC:
                return impl_->theme_.font_italic_path;
            case FontStyle::BOLD_ITALIC:
                return impl_->theme_.font_bold_italic_path;
            default:
                return impl_->theme_.font_path;
        }
    };

    auto GetStyleName = [](FontStyle style) {
        switch (style) {
            case FontStyle::BOLD:
                return "BOLD";
            case FontStyle::ITALIC:
                return "ITALIC";
            case FontStyle::BOLD_ITALIC:
                return "BOLD_ITALIC";
            default:
                return "NORMAL";
        }
    };

    impl_->fonts_[id] = fd;
    auto style = impl_->fonts_[id].style_;
    for (auto &range : impl_->fonts_[id].ranges_) {
        // Substitute proper paths for default CSS-style virtual fonts
        if (range.path == FontDescription::SANS_SERIF) {
            range.path = GetSansSerifPath(style);
        } else if (range.path == FontDescription::MONOSPACE) {
            range.path = impl_->theme_.font_mono_path;
        }
        // Get the actual path
        auto path = FindFontPath(range.path, style);
        if (!path.empty()) {
            range.path = path;
        } else {
            // If we can't find the requested style, try to at least find the
            // typeface.
            auto fallback = FindFontPath(range.path, FontStyle::NORMAL);
            if (fallback.empty()) {
                // But if that doesn't work, fall back to styled sans-serif.
                fallback = GetSansSerifPath(style);
            }
            utility::LogWarning("Could not find font '{}' with style {}",
                                range.path, GetStyleName(style));
            range.path = fallback;
        }
    }

    if (id == DEFAULT_FONT_ID && fd.point_size_ > 0) {
        impl_->theme_.font_size = fd.point_size_;
    }
}

const std::vector<FontDescription> &Application::GetFontDescriptions() const {
    return impl_->fonts_;
}

double Application::Now() const {
    static auto g_tzero = std::chrono::steady_clock::now();
    std::chrono::duration<double> t =
            std::chrono::steady_clock::now() - g_tzero;
    return t.count();
}

std::shared_ptr<Menu> Application::GetMenubar() const {
    return impl_->menubar_;
}

void Application::SetMenubar(std::shared_ptr<Menu> menubar) {
    auto old = impl_->menubar_;
    impl_->menubar_ = menubar;
    // If added or removed menubar, the size of the window's content region
    // may have changed (in not on macOS), so need to relayout.
    if ((!old && menubar) || (old && !menubar)) {
        for (auto w : impl_->windows_) {
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
    impl_->windows_.insert(window);
}

void Application::RemoveWindow(Window *window) {
    if (impl_->should_quit_) {
        return;
    }

    for (auto it = impl_->windows_.begin(); it != impl_->windows_.end(); ++it) {
        if (it->get() == window) {
            impl_->windows_to_be_destroyed_.insert(*it);
            impl_->windows_.erase(it);
            if (impl_->windows_.empty()) {
                impl_->should_quit_ = true;
            }
            break;
        }
    }
    window->Show(false);
}

void Application::Quit() {
    while (!impl_->windows_.empty()) {
        RemoveWindow(impl_->windows_.begin()->get());
    }
}

void Application::OnTerminate() {
    // Note: if you need to modify this function, you should test that
    // the following still work:
    //  1) on macOS, quit by right-clicking on the dock icon and
    //     selecting Quit.
    //  2) run a Python script that creates a window and exits cleanly.
    //  3) run a Python script that creates a window and throws a
    //     fatal exception.

    // This function should work even if called after a successful cleanup
    // (e.g. after Run() successfully finished, either due to closing the
    // last window or Quit() called).

    Quit();
    // If we are in exit() already (e.g. an exception occurred in a
    // Python callback and the interpreter is exiting) just clearing
    // the shared_ptr may not be sufficient to destroy the object.
    // We need to clean up filament to avoid a crash, but we will
    // hang if the window still exists.
    for (auto w : impl_->windows_to_be_destroyed_) {
        w->DestroyWindow();
    }
    impl_->windows_to_be_destroyed_.clear();
    impl_->CleanupAfterRunning();
}

void Application::OnMenuItemSelected(Menu::ItemId itemId) {
    for (auto w : impl_->windows_) {
        if (w->IsActiveWindow()) {
            w->OnMenuItemSelected(itemId);
            // This is a menu selection that came from a native menu.
            // We need to draw twice to ensure that any new dialog
            // that the menu item may have displayed is properly laid out.
            // (ImGUI can take up to two iterations to fully layout)
            // If we post two expose events they get coalesced, but
            // setting needsLayout forces two (for the reason given above).
            w->SetNeedsLayout();
            w->PostRedraw();
            return;
        }
    }
}

void Application::Run() {
    EnvUnlocker noop;  // containing env is C++
    while (RunOneTick(noop));
}

bool Application::RunOneTick(EnvUnlocker &unlocker,
                             bool cleanup_if_no_windows /*=true*/) {
    // Initialize if we have not started yet
    if (!impl_->is_running_) {
        // Verify that the resource path is valid. If it is not, display a
        // message box (std::cerr may not be visible to the user, if we were run
        // as app).
        if (!impl_->is_initialized_) {
            ShowNativeAlert(
                    "Internal error: Application::Initialize() was not called");
            return false;
        }
        auto resource_path = rendering::EngineInstance::GetResourcePath();
        if (!utility::filesystem::DirectoryExists(resource_path)) {
            std::stringstream err;
            err << "Could not find resource directory:\n'" << resource_path
                << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }
        if (!utility::filesystem::FileExists(impl_->theme_.font_path)) {
            std::stringstream err;
            err << "Could not load UI font:\n'" << impl_->theme_.font_path
                << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }

        impl_->PrepareForRunning();
        impl_->is_running_ = true;
    }

    // Process the events that have queued up
    auto status = ProcessQueuedEvents(unlocker);

    // Cleanup if we are done
    if (status == RunStatus::DONE) {
        if (cleanup_if_no_windows) {
            // Clear all the running tasks. The destructor will wait for them to
            // finish.
            for (auto it = impl_->running_tasks_.begin();
                 it != impl_->running_tasks_.end(); ++it) {
                auto current = it;
                ++it;
                impl_->running_tasks_.erase(current);  // calls join()
            }

            impl_->is_running_ = false;
            impl_->CleanupAfterRunning();
        }
        // reset, otherwise we will be done next time, too.
        impl_->should_quit_ = false;
    }

    return (status == RunStatus::CONTINUE);
}

Application::RunStatus Application::ProcessQueuedEvents(EnvUnlocker &unlocker) {
    unlocker.unlock();  // don't want to be locked while we wait
    impl_->window_system_->WaitEventsTimeout(RUNLOOP_DELAY_SEC);
    unlocker.relock();  // need to relock in case we call any callbacks to
                        // functions in the containing (e.g. Python) environment

    // Handle tick messages.
    double now = Now();
    if (now - impl_->last_time_ >= 0.95 * RUNLOOP_DELAY_SEC) {
        for (auto w : impl_->windows_) {
            w->OnTickEvent(TickEvent());
        }
        impl_->last_time_ = now;
    }

    // Run any posted functions
    // To avoid deadlock while PostToMainThread is called in Posted, the
    // posted_lock_ should not be locked in its invoking.
    decltype(impl_->posted_) posted;
    {
        // The only other place posted_lock_ is used is PostToMainThread.
        // If pybind is posting a Python function, it acquires posted_lock_,
        // then locks the GIL. Since we are locked at this point, we (can)
        // deadlock. (So far only observed on macOS, within about 10 runs)
        unlocker.unlock();
        std::lock_guard<std::mutex> lock(impl_->posted_lock_);
        unlocker.relock();
        posted = std::move(impl_->posted_);
    }

    for (auto &p : posted) {
        // Make sure this window still exists. Unfortunately, p.window
        // is a pointer but impl_->windows_ is a shared_ptr, so we can't
        // use find.
        if (p.window) {
            bool found = false;
            for (auto w : impl_->windows_) {
                if (w.get() == p.window) {
                    found = true;
                }
            }
            if (!found) {
                continue;
            }
        }

        void *old = nullptr;
        if (p.window) {
            old = p.window->MakeDrawContextCurrent();
        }
        p.f();
        if (p.window) {
            p.window->RestoreDrawContext(old);
            p.window->PostRedraw();
        }
    }

    // Clear any tasks that have finished
    impl_->running_tasks_.remove_if(
            [](const Task &t) { return t.IsFinished(); });

    // We can't destroy a GLFW window in a callback, so we need to do it here.
    // Since these are the only copy of the shared pointers, this will cause
    // the Window destructor to be called.
    impl_->windows_to_be_destroyed_.clear();

    if (impl_->should_quit_) {
        return RunStatus::DONE;
    }
    return RunStatus::CONTINUE;
}

void Application::RunInThread(std::function<void()> f) {
    // We need to be on the main thread here.
    impl_->running_tasks_.emplace_back(f);
    impl_->running_tasks_.back().Run();
}

void Application::PostToMainThread(Window *window, std::function<void()> f) {
    std::lock_guard<std::mutex> lock(impl_->posted_lock_);
    impl_->posted_.emplace_back(window, f);
}

const char *Application::GetResourcePath() const {
    return rendering::EngineInstance::GetResourcePath().c_str();
}

const Theme &Application::GetTheme() const { return impl_->theme_; }

std::shared_ptr<geometry::Image> Application::RenderToImage(
        rendering::Renderer &renderer,
        rendering::View *view,
        rendering::Scene *scene,
        int width,
        int height) {
    std::shared_ptr<geometry::Image> img;
    auto callback = [&img](std::shared_ptr<geometry::Image> _img) {
        img = _img;
    };

    // Despite the fact that Renderer is created with a width/height, it is
    // the View's viewport that actually controls the size when rendering to
    // an image. Set the viewport here, rather than in the pybinds so that
    // C++ callers do not need to know do this themselves.
    view->SetViewport(0, 0, width, height);

    renderer.RenderToImage(view, scene, callback);
    renderer.BeginFrame();
    renderer.EndFrame();

    return img;
}

std::shared_ptr<geometry::Image> Application::RenderToDepthImage(
        rendering::Renderer &renderer,
        rendering::View *view,
        rendering::Scene *scene,
        int width,
        int height,
        bool z_in_view_space /* =false */) {
    std::shared_ptr<geometry::Image> img;
    auto callback = [&img](std::shared_ptr<geometry::Image> _img) {
        img = _img;
    };

    // Despite the fact that Renderer is created with a width/height, it is
    // the View's viewport that actually controls the size when rendering to
    // an image. Set the viewport here, rather than in the pybinds so that
    // C++ callers do not need to know do this themselves.
    view->SetViewport(0, 0, width, height);

    renderer.RenderToDepthImage(view, scene, callback, z_in_view_space);
    renderer.BeginFrame();
    renderer.EndFrame();

    return img;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
