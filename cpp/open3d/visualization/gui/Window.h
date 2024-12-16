// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Gui.h"
#include "open3d/visualization/gui/Menu.h"
#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/gui/WindowSystem.h"
#include "open3d/visualization/rendering/Renderer.h"

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
    /// Closes the window and destroys it (unless the close callback cancels
    /// the close)
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

    /// Sets a callback that will be called on every UI tick (about 10 msec).
    /// Callback should return true if a redraw is required (i.e. the UI or
    /// a 3D scene has changed), false otherwise.
    void SetOnTickEvent(std::function<bool()> callback);

    /// Sets a callback that will be called immediately before the window
    /// is closed. Callback should return true if the window should continue
    /// closing or false to cancel the close.
    void SetOnClose(std::function<bool()> callback);

    /// Sets a callback that will intercept key event dispatching to focused
    /// widget. Callback should return true to stop more dispatching or false
    /// to dispatch to focused widget.
    void SetOnKeyEvent(std::function<bool(const KeyEvent&)> callback);

    /// Shows the dialog. If a dialog is currently being shown it will be
    /// closed.
    void ShowDialog(std::shared_ptr<Dialog> dlg);
    /// Closes the dialog.
    void CloseDialog();

    void ShowMessageBox(const char* title, const char* message);

    /// This is for internal use in rare circumstances when the destructor
    /// will not be called in a timely fashion.
    void DestroyWindow();

    // Override to handle menu items
    virtual void OnMenuItemSelected(Menu::ItemId item_id);

    // Override to handle drag and drop on the windows.
    virtual void OnDragDropped(const char* path);

    // Shows or hides the menubar, except on macOS when using real windows.
    // This is intended to be used when using HeadlessWindowSystem but may
    // be useful in other circumstances.
    void ShowMenu(bool show);

    int GetMouseMods() const;  // internal, for WindowSystem

    /// Returns the window's unique identifier when WebRTCWindowSystem is in
    /// use. Returns "window_undefined" if the window system is not
    /// WebRTCWindowSystem.
    std::string GetWebRTCUID() const;

protected:
    /// Returns the preferred size of the window. The window is not
    /// obligated to honor this size. If all children of the window
    /// are layouts, this function does not need to be overridden.
    /// This function can only be called after MakeDrawContextCurrent()
    /// has been called.
    virtual Size CalcPreferredSize();

    /// Lays out all the widgets in the window. If all children
    /// of the window are layouts, this function does not need to
    /// be overridden.
    virtual void Layout(const LayoutContext& context);

    LayoutContext GetLayoutContext();

    const std::vector<std::shared_ptr<Widget>>& GetChildren() const;

public:
    // these are intended for internal delivery of events
    void OnDraw();
    void OnResize();
    void OnMouseEvent(const MouseEvent& e);
    void OnKeyEvent(const KeyEvent& e);
    void OnTextInput(const TextInputEvent& e);
    void OnTickEvent(const TickEvent& e);

    WindowSystem::OSWindow GetOSWindow() const;

private:
    void CreateRenderer();
    Widget::DrawResult DrawOnce(bool is_layout_pass);
    void* MakeDrawContextCurrent() const;
    void RestoreDrawContext(void* old_context) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
