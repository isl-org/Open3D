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

#include "Native.h"

#include "Application.h"

#include <GLFW/glfw3.h>
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
