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

#include "open3d/visualization/gui/WindowSystem.h"

struct GLFWwindow;

namespace open3d {
namespace visualization {
namespace gui {

class GLFWWindowSystem : public WindowSystem {
public:
    GLFWWindowSystem();
    ~GLFWWindowSystem();

    void Initialize() override;
    void Uninitialize() override;

    void WaitEventsTimeout(double timeout_secs) override;

    Size GetScreenSize(OSWindow w) override;

    OSWindow CreateOSWindow(Window* o3d_window,
                            int width,
                            int height,
                            const char* title,
                            int flags) override;
    void DestroyWindow(OSWindow w) override;

    void PostRedrawEvent(OSWindow w) override;

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
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
