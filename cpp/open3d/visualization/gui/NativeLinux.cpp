// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <GLFW/glfw3.h>

#include "Application.h"
#include "Native.h"
#define GLFW_EXPOSE_NATIVE_X11 1
#include <GLFW/glfw3native.h>
#include <memory.h>

namespace open3d {
namespace visualization {
namespace gui {

void* GetNativeDrawable(GLFWwindow* glfw_window) {
    return (void*)glfwGetX11Window(glfw_window);
}

void PostNativeExposeEvent(GLFWwindow* glfw_window) {
    Display* d = glfwGetX11Display();
    auto x11win = glfwGetX11Window(glfw_window);

    XEvent e;
    memset(&e, 0, sizeof(e));
    e.type = Expose;
    e.xexpose.window = x11win;

    XSendEvent(d, x11win, False, ExposureMask, &e);
    XFlush(d);
}

void ShowNativeAlert(const char* message) {
    // Linux doesn't have a native alert
    Application::GetInstance().ShowMessageBox("Alert", message);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
