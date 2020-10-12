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

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Gui.h"
#include "open3d/visualization/gui/Menu.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/rendering/Renderer.h"

struct GLFWwindow;

namespace open3d {
namespace visualization {
namespace gui {

class Dialog;
class Menu;
class Renderer;
struct Theme;

class Window {
    friend class Application;
    friend class Renderer;

public:
    static const int FLAG_HIDDEN;
    static const int FLAG_TOPMOST;

    /// Creates a Window that is auto-sized and centered. Window creation is
    /// NOT thread-safe. Window must be created on the same thread that
    /// calls Application::Run().
    explicit Window(const std::string& title, int flags = 0);

    /// Creates a Window that is centered. Window creation is
    /// NOT thread-safe. Window must be created on the same thread that
    /// calls Application::Run().
    Window(const std::string& title,
           int width,
           int height,
           int flags = 0);  // centered

    /// Creates a Window. Window creation is
    /// NOT thread-safe. Window must be created on the same thread that
    /// calls Application::Run().
    Window(const std::string& title,
           int x,
           int y,
           int width,
           int height,
           int flags = 0);
    virtual ~Window();

    const Theme& GetTheme() const;
    visualization::rendering::Renderer& GetRenderer() const;

    /// Gets the window's size and position in OS pixels, not actual
    /// device pixels.
    Rect GetOSFrame() const;
    /// Sets the window's size and position in OS pixels, not actual
    /// device pixels.
    void SetOSFrame(const Rect& r);

    const char* GetTitle() const;
    void SetTitle(const char* title);

    /// Auto-sizes the window to the results of CalcPreferredSize(),
    /// which by default is the size that the layouts of the window want.
    void SizeToFit();

    /// Sets the size of the window in pixels. Includes menubar on Linux.
    void SetSize(const Size& size);
    /// Returns the total interior size of window, in pixels. On Linux this
    /// includes the menubar.
    Size GetSize() const;
    /// Returns the rectangle that is available to widgets to use;
    /// excludes the menubar.
    Rect GetContentRect() const;
    /// Returns the scaling factor from OS pixels to device pixels
    float GetScaling() const;
    /// Returns the global point (in OS pixels) in window local coordinates.
    Point GlobalToWindowCoord(int global_x, int global_y);

    bool IsVisible() const;
    void Show(bool vis = true);
    /// Closes the window and destroys it.
    /// (Same as calling Application::RemoveWindow())
    void Close();

    /// Instructs the window to relayout before the next draw.
    void SetNeedsLayout();
    /// Sends a draw event to the window through the operating system's
    /// event queue.
    void PostRedraw();

    void SetTopmost(bool topmost);
    void RaiseToTop() const;

    bool IsActiveWindow() const;

    /// Sets \param w as widget with text focus.
    void SetFocusWidget(Widget* w);

    void AddChild(std::shared_ptr<Widget> w);

    /// Sets a callback for menu items. If you inherit from Window you can
    /// also override OnMenuItemSelected(); however, you should choose one or
    /// the other, but don't use both.
    void SetOnMenuItemActivated(Menu::ItemId item_id,
                                std::function<void()> callback);

    void SetOnTickEvent(std::function<bool()> callback);

    /// Shows the dialog. If a dialog is currently being shown it will be
    /// closed.
    void ShowDialog(std::shared_ptr<Dialog> dlg);
    /// Closes the dialog.
    void CloseDialog();

    void ShowMessageBox(const char* title, const char* message);

    /// This is for internal use in rare circumstances when the destructor
    /// will not be called in a timely fashion.
    void DestroyWindow();

protected:
    /// Returns the preferred size of the window. The window is not
    /// obligated to honor this size. If all children of the window
    /// are layouts, this function does not need to be overridden.
    /// This function can only be called after MakeDrawContextCurrent()
    /// has been called.
    virtual Size CalcPreferredSize();

    /// Lays out all the widgets in the window. If all children
    /// of the window are layouts, this function does not need to
    /// be overriden.
    virtual void Layout(const Theme& theme);

    // Override to handle menu items
    virtual void OnMenuItemSelected(Menu::ItemId item_id);

    // Override to handle drag and drop on the windows.
    virtual void OnDragDropped(const char* path);

    const std::vector<std::shared_ptr<Widget>>& GetChildren() const;

private:
    enum DrawResult { NONE, REDRAW };
    DrawResult OnDraw();
    Widget::DrawResult DrawOnce(bool is_layout_pass);
    void ForceRedrawSceneWidget();
    void OnResize();
    void OnMouseEvent(const MouseEvent& e);
    void OnKeyEvent(const KeyEvent& e);
    void OnTextInput(const TextInputEvent& e);
    bool OnTickEvent(const TickEvent& e);
    void* MakeDrawContextCurrent() const;
    void RestoreDrawContext(void* old_context) const;
    void* GetNativeDrawable() const;

    static void DrawCallback(GLFWwindow* window);
    static void ResizeCallback(GLFWwindow* window, int os_width, int os_height);
    static void WindowMovedCallback(GLFWwindow* window, int os_x, int os_y);
    static void RescaleCallback(GLFWwindow* window, float xscale, float yscale);
    static void MouseMoveCallback(GLFWwindow* window, double x, double y);
    static void MouseButtonCallback(GLFWwindow* window,
                                    int button,
                                    int action,
                                    int mods);
    static void MouseScrollCallback(GLFWwindow* window, double dx, double dy);
    static void KeyCallback(
            GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CharCallback(GLFWwindow* window, unsigned int utf32char);
    static void DragDropCallback(GLFWwindow*, int count, const char* paths[]);
    static void CloseCallback(GLFWwindow* window);
    static void UpdateAfterEvent(Window* w);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
