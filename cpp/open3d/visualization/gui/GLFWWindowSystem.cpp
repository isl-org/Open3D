// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/GLFWWindowSystem.h"

#include <GLFW/glfw3.h>

#include <iostream>
#include <unordered_map>

#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/MenuImgui.h"
#ifdef __APPLE__
#include "open3d/visualization/gui/MenuMacOS.h"
#endif
#include "open3d/visualization/gui/Native.h"
#include "open3d/visualization/gui/Window.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static constexpr int FALLBACK_MONITOR_WIDTH = 1024;
static constexpr int FALLBACK_MONITOR_HEIGHT = 768;

// GLFW doesn't provide double-click messages, nor will it read the default
// values from the OS, so we need to do it ourselves.
static constexpr double DOUBLE_CLICK_TIME = 0.300;  // 300 ms is a typical value

// These are used in the GLFW callbacks, which are global functions, and it's
// not worth creating a wrapper around Window just for this.
double g_last_button_down_time = 0.0;
MouseButton g_last_button_down = MouseButton::NONE;

int MouseButtonFromGLFW(int button) {
    switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            return int(MouseButton::LEFT);
        case GLFW_MOUSE_BUTTON_RIGHT:
            return int(MouseButton::RIGHT);
        case GLFW_MOUSE_BUTTON_MIDDLE:
            return int(MouseButton::MIDDLE);
        case GLFW_MOUSE_BUTTON_4:
            return int(MouseButton::BUTTON4);
        case GLFW_MOUSE_BUTTON_5:
            return int(MouseButton::BUTTON5);
        default:
            return int(MouseButton::NONE);
    }
}

int KeymodsFromGLFW(int glfw_mods) {
    int keymods = 0;
    if (glfw_mods & GLFW_MOD_SHIFT) {
        keymods |= int(KeyModifier::SHIFT);
    }
    if (glfw_mods & GLFW_MOD_CONTROL) {
#if __APPLE__
        keymods |= int(KeyModifier::ALT);
#else
        keymods |= int(KeyModifier::CTRL);
#endif  // __APPLE__
    }
    if (glfw_mods & GLFW_MOD_ALT) {
#if __APPLE__
        keymods |= int(KeyModifier::META);
#else
        keymods |= int(KeyModifier::ALT);
#endif  // __APPLE__
    }
    if (glfw_mods & GLFW_MOD_SUPER) {
#if __APPLE__
        keymods |= int(KeyModifier::CTRL);
#else
        keymods |= int(KeyModifier::META);
#endif  // __APPLE__
    }
    return keymods;
}

float CallGLFWGetWindowContentScale(GLFWwindow* w) {
    float xscale, yscale;
    glfwGetWindowContentScale(w, &xscale, &yscale);
    return std::min(xscale, yscale);
}

}  // namespace

GLFWWindowSystem::GLFWWindowSystem() {}

GLFWWindowSystem::~GLFWWindowSystem() {}

void GLFWWindowSystem::Initialize() {
#if __APPLE__
    // If we are running from Python we might not be running from a bundle
    // and would therefore not be a Proper app yet.
    MacTransformIntoApp();

    glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);  // no auto-create menubar
    // Don't change directory to resource directory in bundle (which is awkward
    // if using a framework version of Python).
    glfwInitHint(GLFW_COCOA_CHDIR_RESOURCES, GLFW_FALSE);
#endif
    glfwInit();
}

void GLFWWindowSystem::Uninitialize() { glfwTerminate(); }

void GLFWWindowSystem::WaitEventsTimeout(double timeout_secs) {
    glfwWaitEventsTimeout(timeout_secs);
    const char* err;
    if (glfwGetError(&err) != GLFW_NO_ERROR) {
        std::cerr << "[error] GLFW error: " << err << std::endl;
    }
}

Size GLFWWindowSystem::GetScreenSize(OSWindow w) {
    int screen_width = FALLBACK_MONITOR_WIDTH;
    int screen_height = FALLBACK_MONITOR_HEIGHT;
    auto* monitor = glfwGetWindowMonitor((GLFWwindow*)w);
    if (!monitor) {
        monitor = glfwGetPrimaryMonitor();
    }
    if (monitor) {
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (mode) {
            screen_width = mode->width;
            screen_height = mode->height;
        }
        // TODO: if we can update GLFW we can replace the above with this
        //       Also, see below.
        // int xpos, ypos;
        // glfwGetMonitorWorkarea(monitor, &xpos, &ypos,
        //                       &screen_width, &screen_height);
    }

    return Size(screen_width, screen_height);
}

GLFWWindowSystem::OSWindow GLFWWindowSystem::CreateOSWindow(Window* o3d_window,
                                                            int width,
                                                            int height,
                                                            const char* title,
                                                            int flags) {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // // NOTE: Setting alpha and stencil bits to match GLX standard default
    // // values. GLFW sets these internally to 8 and 8 respectively if not
    // // specified which causes problems with Filament on Linux with Nvidia
    // binary
    // // driver
    // glfwWindowHint(GLFW_ALPHA_BITS, 0);
    // glfwWindowHint(GLFW_STENCIL_BITS, 0);

#if __APPLE__
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
#endif
    bool visible = !(flags & FLAG_HIDDEN);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING,
                   ((flags & FLAG_TOPMOST) != 0 ? GLFW_TRUE : GLFW_FALSE));

    auto* glfw_window = glfwCreateWindow(width, height, title, NULL, NULL);

    glfwSetWindowUserPointer(glfw_window, o3d_window);
    glfwSetWindowSizeCallback(glfw_window, ResizeCallback);
    glfwSetWindowPosCallback(glfw_window, WindowMovedCallback);
    glfwSetWindowRefreshCallback(glfw_window, DrawCallback);
    glfwSetCursorPosCallback(glfw_window, MouseMoveCallback);
    glfwSetMouseButtonCallback(glfw_window, MouseButtonCallback);
    glfwSetScrollCallback(glfw_window, MouseScrollCallback);
    glfwSetKeyCallback(glfw_window, KeyCallback);
    glfwSetCharCallback(glfw_window, CharCallback);
    glfwSetDropCallback(glfw_window, DragDropCallback);
    glfwSetWindowCloseCallback(glfw_window, CloseCallback);

    return glfw_window;
}

void GLFWWindowSystem::DestroyWindow(OSWindow w) {
    glfwDestroyWindow((GLFWwindow*)w);
}

void GLFWWindowSystem::PostRedrawEvent(OSWindow w) {
    PostNativeExposeEvent((GLFWwindow*)w);
}

bool GLFWWindowSystem::GetWindowIsVisible(OSWindow w) const {
    return glfwGetWindowAttrib((GLFWwindow*)w, GLFW_VISIBLE);
}

void GLFWWindowSystem::ShowWindow(OSWindow w, bool show) {
    if (show) {
        glfwShowWindow((GLFWwindow*)w);
    } else {
        glfwHideWindow((GLFWwindow*)w);
    }
}

void GLFWWindowSystem::RaiseWindowToTop(OSWindow w) {
    glfwFocusWindow((GLFWwindow*)w);
}

bool GLFWWindowSystem::IsActiveWindow(OSWindow w) const {
    return glfwGetWindowAttrib((GLFWwindow*)w, GLFW_FOCUSED);
}

Point GLFWWindowSystem::GetWindowPos(OSWindow w) const {
    int x, y;
    glfwGetWindowPos((GLFWwindow*)w, &x, &y);
    return Point(x, y);
}

void GLFWWindowSystem::SetWindowPos(OSWindow w, int x, int y) {
    glfwSetWindowPos((GLFWwindow*)w, x, y);
}

Size GLFWWindowSystem::GetWindowSize(OSWindow w) const {
    int width, height;
    glfwGetWindowSize((GLFWwindow*)w, &width, &height);
    return Size(width, height);
}

void GLFWWindowSystem::SetWindowSize(OSWindow w, int width, int height) {
    glfwSetWindowSize((GLFWwindow*)w, width, height);
}

Size GLFWWindowSystem::GetWindowSizePixels(OSWindow w) const {
    uint32_t width, height;
    glfwGetFramebufferSize((GLFWwindow*)w, (int*)&width, (int*)&height);
    return Size(width, height);
}

void GLFWWindowSystem::SetWindowSizePixels(OSWindow w, const Size& size) {
    std::cout << "[o3d] TODO: implement GLFWWindowSystem::SetWindowSizePixels()"
              << std::endl;
}

float GLFWWindowSystem::GetWindowScaleFactor(OSWindow w) const {
    // This function returns the number of device pixels per OS distance-unit.
    // Windows and Linux keep one pixel equal to one real pixel, whereas
    // macOS keeps the unit of measurement the same (1 pt = 1/72 inch) and
    // changes the number pixels in one "virtual pixel". This function returns
    // the scale factor as macOS thinks of it. This function should be used
    // in converting to/from OS coordinates (e.g. mouse events), but not for
    // sizing user interface elements like fonts.
#if __APPLE__
    return CallGLFWGetWindowContentScale((GLFWwindow*)w);
#else
    return 1.0f;
#endif  // __APPLE__
}

float GLFWWindowSystem::GetUIScaleFactor(OSWindow w) const {
    // This function returns the scale factor needed to have appropriately
    // sized user interface elements.
    return CallGLFWGetWindowContentScale((GLFWwindow*)w);
}

void GLFWWindowSystem::SetWindowTitle(OSWindow w, const char* title) {
    glfwSetWindowTitle((GLFWwindow*)w, title);
}

Point GLFWWindowSystem::GetMousePosInWindow(OSWindow w) const {
    double mx, my;
    glfwGetCursorPos((GLFWwindow*)w, &mx, &my);
    auto scaling = GetWindowScaleFactor((GLFWwindow*)w);
    return Point(int(float(mx) * scaling), int(float(my) * scaling));
}

int GLFWWindowSystem::GetMouseButtons(OSWindow w) const {
    GLFWwindow* gw = (GLFWwindow*)w;
    int buttons = 0;
    if (glfwGetMouseButton(gw, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        buttons |= int(MouseButton::LEFT);
    }
    if (glfwGetMouseButton(gw, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        buttons |= int(MouseButton::RIGHT);
    }
    if (glfwGetMouseButton(gw, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        buttons |= int(MouseButton::MIDDLE);
    }
    return buttons;
}

void GLFWWindowSystem::CancelUserClose(OSWindow w) {
    glfwSetWindowShouldClose((GLFWwindow*)w, 0);
}

// ----------------------------------------------------------------------------
void GLFWWindowSystem::DrawCallback(GLFWwindow* window) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->OnDraw();
}

void GLFWWindowSystem::ResizeCallback(GLFWwindow* window,
                                      int os_width,
                                      int os_height) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->OnResize();
}

void GLFWWindowSystem::WindowMovedCallback(GLFWwindow* window,
                                           int os_x,
                                           int os_y) {
#ifdef __APPLE__
    // On macOS we need to recreate the swap chain if the window changes
    // size OR MOVES!
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->OnResize();
#endif
}

void GLFWWindowSystem::RescaleCallback(GLFWwindow* window,
                                       float xscale,
                                       float yscale) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->OnResize();
}

void GLFWWindowSystem::MouseMoveCallback(GLFWwindow* window,
                                         double x,
                                         double y) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    int buttons = 0;
    for (int b = GLFW_MOUSE_BUTTON_1; b < GLFW_MOUSE_BUTTON_5; ++b) {
        if (glfwGetMouseButton(window, b) == GLFW_PRESS) {
            buttons |= MouseButtonFromGLFW(b);
        }
    }
    float scaling =
            Application::GetInstance().GetWindowSystem().GetWindowScaleFactor(
                    window);
    int ix = int(std::ceil(x * scaling));
    int iy = int(std::ceil(y * scaling));

    auto type = (buttons == 0 ? MouseEvent::MOVE : MouseEvent::DRAG);
    MouseEvent me = MouseEvent::MakeButtonEvent(type, ix, iy, w->GetMouseMods(),
                                                MouseButton(buttons), 1);

    w->OnMouseEvent(me);
}

void GLFWWindowSystem::MouseButtonCallback(GLFWwindow* window,
                                           int button,
                                           int action,
                                           int mods) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto type = (action == GLFW_PRESS ? MouseEvent::BUTTON_DOWN
                                      : MouseEvent::BUTTON_UP);
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    float scaling =
            Application::GetInstance().GetWindowSystem().GetWindowScaleFactor(
                    window);
    int ix = int(std::ceil(mx * scaling));
    int iy = int(std::ceil(my * scaling));

    MouseEvent me = MouseEvent::MakeButtonEvent(
            type, ix, iy, KeymodsFromGLFW(mods),
            MouseButton(MouseButtonFromGLFW(button)), 1);

    double now = Application::GetInstance().Now();
    if (g_last_button_down == me.button.button) {
        double dt = now - g_last_button_down_time;
        if (dt > 0.0 && dt < DOUBLE_CLICK_TIME) {
            me.button.count += 1;
        }
    }
    if (type == MouseEvent::BUTTON_DOWN) {
        g_last_button_down = me.button.button;
        g_last_button_down_time = now;
    }
    w->OnMouseEvent(me);
}

void GLFWWindowSystem::MouseScrollCallback(GLFWwindow* window,
                                           double dx,
                                           double dy) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    float scaling =
            Application::GetInstance().GetWindowSystem().GetWindowScaleFactor(
                    window);
    int ix = int(std::ceil(mx * scaling));
    int iy = int(std::ceil(my * scaling));

    // GLFW doesn't give us any information about whether this scroll event
    // came from a mousewheel or a trackpad two-finger scroll.
#if __APPLE__
    bool isTrackpad = true;
#else
    bool isTrackpad = false;
#endif  // __APPLE__

    // Note that although pixels are integers, the trackpad value needs to
    // be a float, since macOS trackpads produce fractional values when
    // scrolling slowly. These fractional values need to be passed all the way
    // down to the MatrixInteractorLogic::Dolly() in order for dollying to
    // feel buttery smooth with the trackpad.
    MouseEvent me = MouseEvent::MakeWheelEvent(
            MouseEvent::WHEEL, ix, iy, w->GetMouseMods(), dx, dy, isTrackpad);

    w->OnMouseEvent(me);
}

void GLFWWindowSystem::KeyCallback(
        GLFWwindow* window, int key, int scancode, int action, int mods) {
    static std::unordered_map<int, uint32_t> g_GLFW2Key = {
            {GLFW_KEY_BACKSPACE, KEY_BACKSPACE},
            {GLFW_KEY_TAB, KEY_TAB},
            {GLFW_KEY_ENTER, KEY_ENTER},
            {GLFW_KEY_ESCAPE, KEY_ESCAPE},
            {GLFW_KEY_DELETE, KEY_DELETE},
            {GLFW_KEY_LEFT_SHIFT, KEY_LSHIFT},
            {GLFW_KEY_RIGHT_SHIFT, KEY_RSHIFT},
            {GLFW_KEY_LEFT_CONTROL, KEY_LCTRL},
            {GLFW_KEY_RIGHT_CONTROL, KEY_RCTRL},
            {GLFW_KEY_LEFT_ALT, KEY_ALT},
            {GLFW_KEY_RIGHT_ALT, KEY_ALT},
            {GLFW_KEY_LEFT_SUPER, KEY_META},
            {GLFW_KEY_RIGHT_SUPER, KEY_META},
            {GLFW_KEY_CAPS_LOCK, KEY_CAPSLOCK},
            {GLFW_KEY_LEFT, KEY_LEFT},
            {GLFW_KEY_RIGHT, KEY_RIGHT},
            {GLFW_KEY_UP, KEY_UP},
            {GLFW_KEY_DOWN, KEY_DOWN},
            {GLFW_KEY_INSERT, KEY_INSERT},
            {GLFW_KEY_HOME, KEY_HOME},
            {GLFW_KEY_END, KEY_END},
            {GLFW_KEY_PAGE_UP, KEY_PAGEUP},
            {GLFW_KEY_PAGE_DOWN, KEY_PAGEDOWN},
    };
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));

    auto type = (action == GLFW_RELEASE ? KeyEvent::Type::UP
                                        : KeyEvent::Type::DOWN);

    uint32_t k = key;
    if (key >= 'A' && key <= 'Z') {
        k += 32;  // GLFW gives uppercase for letters, convert to lowercase
    } else {
        auto it = g_GLFW2Key.find(key);
        if (it != g_GLFW2Key.end()) {
            k = it->second;
        }
    }
    KeyEvent e = {type, k, (action == GLFW_REPEAT)};

    w->OnKeyEvent(e);
}

void GLFWWindowSystem::CharCallback(GLFWwindow* window,
                                    unsigned int utf32char) {
    // Convert utf-32 to utf8
    // From https://stackoverflow.com/a/42013433/218226
    // Note: This code handles all characters, but non-European characters
    //       won't draw unless we will include them in the ImGUI font (which
    //       is prohibitively large for hanzi/kanji)
    char utf8[5];
    if (utf32char <= 0x7f) {
        utf8[0] = utf32char;
        utf8[1] = '\0';
    } else if (utf32char <= 0x7ff) {
        utf8[0] = 0xc0 | (utf32char >> 6);
        utf8[1] = 0x80 | (utf32char & 0x3f);
        utf8[2] = '\0';
    } else if (utf32char <= 0xffff) {
        utf8[0] = 0xe0 | (utf32char >> 12);
        utf8[1] = 0x80 | ((utf32char >> 6) & 0x3f);
        utf8[2] = 0x80 | (utf32char & 0x3f);
        utf8[3] = '\0';
    } else if (utf32char <= 0x10ffff) {
        utf8[0] = 0xf0 | (utf32char >> 18);
        utf8[1] = 0x80 | ((utf32char >> 12) & 0x3f);
        utf8[2] = 0x80 | ((utf32char >> 6) & 0x3f);
        utf8[3] = 0x80 | (utf32char & 0x3f);
        utf8[4] = '\0';
    } else {
        // These characters are supposed to be forbidden, but just in case
        utf8[0] = '?';
        utf8[1] = '\0';
    }

    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->OnTextInput(TextInputEvent{utf8});
}

void GLFWWindowSystem::DragDropCallback(GLFWwindow* window,
                                        int count,
                                        const char* paths[]) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    for (int i = 0; i < count; ++i) {
        w->OnDragDropped(paths[i]);
    }
}

void GLFWWindowSystem::CloseCallback(GLFWwindow* window) {
    Window* w = static_cast<Window*>(glfwGetWindowUserPointer(window));
    w->Close();
}

void* GLFWWindowSystem::GetNativeDrawable(OSWindow w) {
    return gui::GetNativeDrawable((GLFWwindow*)w);
}

rendering::FilamentRenderer* GLFWWindowSystem::CreateRenderer(OSWindow w) {
    return new rendering::FilamentRenderer(
            rendering::EngineInstance::GetInstance(), GetNativeDrawable(w),
            rendering::EngineInstance::GetResourceManager());
}

void GLFWWindowSystem::ResizeRenderer(OSWindow w,
                                      rendering::FilamentRenderer* renderer) {
#if __APPLE__
    // We need to recreate the swap chain after resizing a window on macOS
    // otherwise things look very wrong. SwapChain does not need to be resized
    // on other platforms.
    renderer->UpdateSwapChain();
#endif  // __APPLE__
}

MenuBase* GLFWWindowSystem::CreateOSMenu() {
#ifdef __APPLE__
    return new MenuMacOS();
#else
    return new MenuImgui();
#endif
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
