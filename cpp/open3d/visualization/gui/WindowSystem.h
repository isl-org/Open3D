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

#include "open3d/visualization/gui/Gui.h"

namespace open3d {
namespace visualization {

namespace rendering {
class FilamentRenderer;
}  // namespace rendering

namespace gui {

class MenuBase;
class Window;

/// WindowSystem (and its derived classes) are internal to Open3D and not
/// intended to be used directly. Internal users should get the WindowSystem
/// object using Application::GetInstance().GetWindowSystem().
class WindowSystem {
public:
    using OSWindow = void*;

    virtual ~WindowSystem(){};

    virtual void Initialize() = 0;
    virtual void Uninitialize() = 0;

    virtual void WaitEventsTimeout(double timeout_secs) = 0;

    virtual Size GetScreenSize(OSWindow w) = 0;

    static constexpr int FLAG_VISIBLE = 0;
    static constexpr int FLAG_HIDDEN = (1 << 0);
    static constexpr int FLAG_TOPMOST = (1 << 1);

    virtual OSWindow CreateOSWindow(Window* o3d_window,
                                    int width,
                                    int height,
                                    const char* title,
                                    int flags) = 0;
    virtual void DestroyWindow(OSWindow w) = 0;

    virtual void PostRedrawEvent(OSWindow w) = 0;

    virtual bool GetWindowIsVisible(OSWindow w) const = 0;
    virtual void ShowWindow(OSWindow w, bool show) = 0;

    virtual void RaiseWindowToTop(OSWindow w) = 0;
    virtual bool IsActiveWindow(OSWindow w) const = 0;

    virtual Point GetWindowPos(OSWindow w) const = 0;
    virtual void SetWindowPos(OSWindow w, int x, int y) = 0;

    virtual Size GetWindowSize(OSWindow w) const = 0;
    virtual void SetWindowSize(OSWindow w, int width, int height) = 0;

    virtual Size GetWindowSizePixels(OSWindow w) const = 0;
    virtual void SetWindowSizePixels(OSWindow w, const Size& size) = 0;

    virtual float GetWindowScaleFactor(OSWindow w) const = 0;
    virtual float GetUIScaleFactor(OSWindow w) const = 0;

    virtual void SetWindowTitle(OSWindow w, const char* title) = 0;

    virtual Point GetMousePosInWindow(OSWindow w) const = 0;
    virtual int GetMouseButtons(OSWindow w) const = 0;

    virtual void CancelUserClose(OSWindow w) = 0;

    virtual void* GetNativeDrawable(OSWindow w) = 0;

    // Creates an appropriate renderer with new(). It is the caller's
    // responsibility to delete it.
    virtual rendering::FilamentRenderer* CreateRenderer(OSWindow w) = 0;

    virtual void ResizeRenderer(OSWindow w,
                                rendering::FilamentRenderer* renderer) = 0;

    virtual MenuBase* CreateOSMenu() = 0;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
