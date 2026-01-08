// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>

#include "open3d/visualization/gui/WindowSystem.h"

namespace open3d {

namespace core {
class Tensor;
}

namespace visualization {
namespace gui {

struct MouseEvent;
struct KeyEvent;
struct TextInputEvent;

class BitmapWindowSystem : public WindowSystem {
public:
    // Not needed anymore - may be removed
    enum class Rendering {
        NORMAL,   // normal OpenGL rendering, requires X11, Win32, or Cocoa
        HEADLESS  // uses EGL, does not require the OS to have a window system.
                  // (Linux only)
    };
    BitmapWindowSystem(Rendering mode = Rendering::NORMAL);
    ~BitmapWindowSystem();

    void Initialize() override;
    void Uninitialize() override;

    using OnDrawCallback =
            std::function<void(Window*, std::shared_ptr<core::Tensor>)>;
    void SetOnWindowDraw(OnDrawCallback callback);

    void WaitEventsTimeout(double timeout_secs) override;

    OSWindow CreateOSWindow(Window* o3d_window,
                            int width,
                            int height,
                            const char* title,
                            int flags) override;
    void DestroyWindow(OSWindow w) override;

    Size GetScreenSize(OSWindow w) override;

    void PostRedrawEvent(OSWindow w) override;
    void PostMouseEvent(OSWindow w, const MouseEvent& e);
    void PostKeyEvent(OSWindow w, const KeyEvent& e);
    void PostTextInputEvent(OSWindow w, const TextInputEvent& e);

    bool GetWindowIsVisible(OSWindow w) const override;
    void ShowWindow(OSWindow w, bool show) override;

    void RaiseWindowToTop(OSWindow w) override;
    bool IsActiveWindow(OSWindow w) const override;

    Point GetWindowPos(OSWindow w) const override;
    void SetWindowPos(OSWindow w, int x, int y) override;

    Size GetWindowSize(OSWindow w) const override;
    void SetWindowSize(OSWindow w, int width, int height) override;

    Size GetWindowSizePixels(OSWindow w) const override;
    void SetWindowSizePixels(OSWindow w, const Size& size) override;

    float GetWindowScaleFactor(OSWindow w) const override;
    float GetUIScaleFactor(OSWindow w) const override;

    void SetWindowTitle(OSWindow w, const char* title) override;

    Point GetMousePosInWindow(OSWindow w) const override;
    int GetMouseButtons(OSWindow w) const override;

    void CancelUserClose(OSWindow w) override;

    void* GetNativeDrawable(OSWindow w) override;

    rendering::FilamentRenderer* CreateRenderer(OSWindow w) override;

    void ResizeRenderer(OSWindow w,
                        rendering::FilamentRenderer* renderer) override;

    MenuBase* CreateOSMenu() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
