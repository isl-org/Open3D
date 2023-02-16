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

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "open3d/visualization/gui/Font.h"
#include "open3d/visualization/gui/Menu.h"

namespace open3d {

namespace geometry {
class Image;
}

namespace visualization {

namespace rendering {
class Renderer;
class View;
class Scene;
}  // namespace rendering

namespace gui {

struct Theme;
class Window;
class WindowSystem;

class Application {
public:
    static Application &GetInstance();

    virtual ~Application();

    /// Initializes the application, and in particular, finds the path for
    /// the resources. If you can provide the argc/argv arguments it is more
    /// reliable.
    void Initialize();
    /// Initializes the application, and in particular, finds the path for
    /// the resources. If you can provide the argc/argv arguments it is more
    /// reliable.
    void Initialize(int argc, const char *argv[]);
    /// Initializes the application, with a specific path to the resources.
    void Initialize(const char *resource_path);

    /// Identifier for font used by default for all UI elements
    static constexpr FontId DEFAULT_FONT_ID = 0;
    /// Adds a font. Must be called after Initialize() and before a window is
    /// created.
    FontId AddFont(const FontDescription &fd);
    /// Replaces font. Must be called after Initialize() and before a window is
    /// created.
    void SetFont(FontId id, const FontDescription &fd);

    /// Does not return until the UI is completely finished.
    void Run();
    /// Closes all the windows, which exits as a result
    void Quit();

    /// Runs \param f in a separate thread. Do NOT call UI functions in
    /// \p f; if you have a long running function that needs to call UI
    /// functions (e.g. updating a progress bar), have your function call
    /// PostToMainThread() with code that will do the UI (note: your function
    /// may finish before the code given to PostToMainThread will run, so if
    /// using lambdas, capture by copy and make sure whatever you use will
    /// still be alive).
    void RunInThread(std::function<void()> f);
    /// Runs \p f on the main thread at some point in the near future.
    /// Proper context will be setup for \p window. \p f will block the
    /// UI, so it should run quickly. If you need to do something slow
    /// (e.g. load a file) consider using RunInThread() and have the function
    /// pass off UI calls to PostToMainThread().
    void PostToMainThread(Window *window, std::function<void()> f);

    std::shared_ptr<Menu> GetMenubar() const;
    void SetMenubar(std::shared_ptr<Menu> menubar);

    /// Must be called on the same thread that calls Run()
    void AddWindow(std::shared_ptr<Window> window);
    /// Must be called on the same thread that calls Run(). This is normally
    /// called from Window::Close() and should not need to be called in user
    /// code.
    void RemoveWindow(Window *window);

    /// Creates a message box window the next time the event loop processes.
    /// This message box will be a separate window and not associated with any
    /// of the other windows shown with AddWindow().
    ///     THIS FUNCTION SHOULD BE USED ONLY AS A LAST RESORT!
    /// If you have a window, you should use Window::ShowMessageBox() so that
    /// the message box will be modal to that window. If you do not have a
    /// window it is better to use ShowNativeAlert(). If the platform does not
    /// have an alert (like Linux), then this can be used as a last resort.
    void ShowMessageBox(const char *title, const char *message);

    WindowSystem &GetWindowSystem() const;

    // (std::string not good in interfaces for ABI reasons)
    const char *GetResourcePath() const;

    /// This is primarily intended for use by the Window class. Any size-related
    /// fields (for example, fontSize) should be accessed through
    /// Window::GetTheme() as they are updated to reflect the pixel scaling
    /// on the monitor where the Window is displayed.
    const Theme &GetTheme() const;

    /// Returns high-resolution counter value (in seconds). Not valid
    /// until Initialize() is called.
    double Now() const;

    /// Delivers the itemId to the active window. Used internally.
    void OnMenuItemSelected(Menu::ItemId itemId);

    /// Cleanup everything right now. An example of usage is Cocoa's
    /// -applicationWillTermiate: AppDelegate message. Using Quit would result
    /// in a crash (and an unsightly message from macOS) due to destructing
    /// the windows at the wrong time.
    void OnTerminate();

    class EnvUnlocker {
    public:
        EnvUnlocker() {}
        virtual ~EnvUnlocker() {}
        virtual void unlock() {}
        virtual void relock() {}
    };
    /// For internal use. Returns true if the run loop has not finished, and
    /// false if the last window has closed or Quit() has been called.
    /// EnvUnlocker allows an external environment to provide
    /// a way to unlock the environment while we wait for the next event.
    /// This is useful to release the Python GIL, for example. Callers of
    /// of Open3D's GUI from languages such as scripting languages which do
    /// not expect the author to need to clean up after themselves may want to
    /// write their own Run() function that calls RunOneTick() with
    /// cleanup_if_no_windows=false and schedule a call to OnTerminate() with
    /// atexit().
    bool RunOneTick(EnvUnlocker &unlocker, bool cleanup_if_no_windows = true);

    /// Sets the WindowSystem to given object. Can be used to override the
    /// default GLFWWindowSystem (e.g. BitmapWindowSystem). Must be called
    /// before creating any Windows.
    void SetWindowSystem(std::shared_ptr<WindowSystem> ws);

    /// Returns true if using the native OS window system, false otherwise.
    /// This is useful for choosing to display some features that are only
    /// useful with native windows. For instance, when embedded in a Jupyter
    /// notebook, a "Close Window" menu item is not necessary.
    bool UsingNativeWindows() const;

    /// Verifies that Initialize() has been called, printing out an error and
    /// exiting if not.
    void VerifyIsInitialized();

    const std::vector<FontDescription> &GetFontDescriptions() const;

    /// Returns the scene rendered to an image. This MUST NOT be called while
    /// in Run(). It is intended for use when no windows are shown. If you
    /// need to render from a GUI, use Scene::RenderToImage().
    std::shared_ptr<geometry::Image> RenderToImage(
            rendering::Renderer &renderer,
            rendering::View *view,
            rendering::Scene *scene,
            int width,
            int height);

    // Same as RenderToImage(), but returns the depth values in a float image.
    std::shared_ptr<geometry::Image> RenderToDepthImage(
            rendering::Renderer &renderer,
            rendering::View *view,
            rendering::Scene *scene,
            int width,
            int height,
            bool z_in_view_space = false);

private:
    Application();

    enum class RunStatus { CONTINUE, DONE };
    RunStatus ProcessQueuedEvents(EnvUnlocker &unlocker);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
