// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <GLFW/glfw3.h>

#include "Native.h"
#define GLFW_EXPOSE_NATIVE_WIN32 1
#include <GLFW/glfw3native.h>
#include <winuser.h>

namespace open3d {
namespace visualization {
namespace gui {

void* GetNativeDrawable(GLFWwindow* glfw_window) {
    return glfwGetWin32Window(glfw_window);
}

void PostNativeExposeEvent(GLFWwindow* glfw_window) {
    InvalidateRect(glfwGetWin32Window(glfw_window), NULL, FALSE);
    // InvalidateRect() does not actually post an event to the message queue.
    // The way paint events work on Windows is that the window gets marked
    // as dirty, then the next time GetMessage() is called and there isn't
    // an actual event and the window is dirty, then a paint event is
    // synthesized and the wndproc called. For some reason, a paint event
    // is never actually generated. I suspect it is because Filament's
    // render thread finishes and presumably buffer swap validates the
    // window, erasing the dirty flag, before the event queue has time to
    // notice that the window was marked as dirty. So force an update.
    // Unfortunately, this draws *now*, so we have to wait until we are
    // done drawing, which needs to be done at a higher level.
    UpdateWindow(glfwGetWin32Window(glfw_window));
}

void ShowNativeAlert(const char* message) {
    MessageBox(NULL, "Alert", message, MB_OK | MB_ICONEXCLAMATION);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
