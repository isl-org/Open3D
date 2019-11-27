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

#include "Application.h"

#include "Theme.h"
#include "Window.h"

#include <SDL.h>

#include <chrono>
#include <thread>
#include <unordered_map>

#include <sys/stat.h>
#include <sys/types.h>
#if !defined(WIN32)
#    include <unistd.h>
#else
#    include <io.h>
#endif

#ifdef WIN32
#define getcwd _getcwd
#endif // WIN32

namespace {

const int RUNLOOP_DELAY_MSEC = 10;

bool isDirectory(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0)
        return false;
    return S_ISDIR(statbuf.st_mode);
}

std::string findResourcePath(int argc, const char *argv[]) {
    std::string argv0;
    if (argc != 0 && argv) {
        argv0 = argv[0];
    }

    // Convert backslash (Windows) to forward slash
    for (auto &c : argv0) {
        if (c == '\\') {
            c = '/';
        }
    }

    // Chop off the process name
    auto lastSlash = argv0.rfind("/");
    auto path = argv0.substr(0, lastSlash);

    if (argv0[0] == '/' || (argv0.size() > 3 && argv0[1] == ':' && argv0[2] == '/')) {
        // is absolute path, we're done
    } else {
        // relative path:  prepend working directory
        char *cwd = getcwd(nullptr, 0); // will malloc()
        path = std::string(cwd) + "/" + path;
        free(cwd);
    }

#ifdef __APPLE__
    if (path.rfind("MacOS") == path.size() - 5) {  // path is in a bundle
        return path.substr(0, path.size() - 5) + "Resources";
    }
#endif // __APPLE__

    auto rsrcPath = path + "/resources";
    if (!isDirectory(rsrcPath)) {
        return path + "/../resources";  // building with Xcode
    }
    return rsrcPath;
}

}

namespace open3d {
namespace gui {

struct Application::Impl {
    std::string resourcePath;
    std::unordered_map<uint32_t, std::shared_ptr<Window>> windows;
    Theme theme;
};

Application& Application::GetInstance() {
    static Application gApp;
    return gApp;
}

Application::Application()
: impl_(new Application::Impl()) {
    impl_->theme.backgroundColor = Color(0.25, 0.25, 0.25);
    impl_->theme.fontPath = "Roboto-Medium.ttf";  // full path will be added in Initialize()
    impl_->theme.fontSize = 16;
    impl_->theme.textColor = Color(0.9, 0.9, 0.9);
    impl_->theme.borderWidth = 1;
    impl_->theme.borderRadius = 3;
    impl_->theme.borderColor = Color(0.5, 0.5, 0.5);
}

Application::~Application() {
}

void Application::Initialize(int argc, const char *argv[]) {
    impl_->resourcePath = findResourcePath(argc, argv);
    impl_->theme.fontPath = impl_->resourcePath + "/" + impl_->theme.fontPath;
}

void Application::AddWindow(std::shared_ptr<Window> window) {
    window->OnResize();  // so we get an initial resize
    impl_->windows[window->GetID()] = window;
}

void Application::Run() {
    SDL_Init(SDL_INIT_EVENTS);

/*    ImGuiIO& io = ImGui::GetIO();
#ifdef WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window->getSDLWindow(), &wmInfo);
    io.ImeWindowHandle = wmInfo.info.win.window;
#endif
    io.KeyMap[ImGuiKey_Tab] = SDL_SCANCODE_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = SDL_SCANCODE_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = SDL_SCANCODE_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = SDL_SCANCODE_UP;
    io.KeyMap[ImGuiKey_DownArrow] = SDL_SCANCODE_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = SDL_SCANCODE_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown] = SDL_SCANCODE_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home] = SDL_SCANCODE_HOME;
    io.KeyMap[ImGuiKey_End] = SDL_SCANCODE_END;
    io.KeyMap[ImGuiKey_Insert] = SDL_SCANCODE_INSERT;
    io.KeyMap[ImGuiKey_Delete] = SDL_SCANCODE_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = SDL_SCANCODE_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = SDL_SCANCODE_SPACE;
    io.KeyMap[ImGuiKey_Enter] = SDL_SCANCODE_RETURN;
    io.KeyMap[ImGuiKey_Escape] = SDL_SCANCODE_ESCAPE;
    io.KeyMap[ImGuiKey_A] = SDL_SCANCODE_A;
    io.KeyMap[ImGuiKey_C] = SDL_SCANCODE_C;
    io.KeyMap[ImGuiKey_V] = SDL_SCANCODE_V;
    io.KeyMap[ImGuiKey_X] = SDL_SCANCODE_X;
    io.KeyMap[ImGuiKey_Y] = SDL_SCANCODE_Y;
    io.KeyMap[ImGuiKey_Z] = SDL_SCANCODE_Z;
    io.SetClipboardTextFn = [](void*, const char* text) {
        SDL_SetClipboardText(text);
    };
    io.GetClipboardTextFn = [](void*) -> const char* {
        return SDL_GetClipboardText();
    };
    io.ClipboardUserData = nullptr;
*/
    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);

    bool done = false;
    while (!done) {
//        SDL_Window* sdlWindow = window->getSDLWindow();
//        if (mWindowTitle != SDL_GetWindowTitle(sdlWindow)) {
//            SDL_SetWindowTitle(sdlWindow, mWindowTitle.c_str());
//        }

//        if (!UTILS_HAS_THREADING) {
//            mEngine->execute();
//        }

        constexpr int kMaxEvents = 16;
        SDL_Event events[kMaxEvents];
        int nevents = 0;
        while (nevents < kMaxEvents && SDL_PollEvent(&events[nevents]) != 0) {
//            ImGuiIO& io = ImGui::GetIO();
            SDL_Event* event = &events[nevents];
            switch (event->type) {
                case SDL_QUIT:   // sent after last window closed
                    done = true;
                    break;
                case SDL_MOUSEMOTION: {
                    auto &e = event->motion;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        win->OnMouseMove(MouseMoveEvent{ int(std::ceil(float(e.x) * scaling)),
                                                         int(std::ceil(float(e.y) * scaling)) });
                    }
                    break;
                }
                case SDL_MOUSEWHEEL: {
                    auto &e = event->wheel;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        win->OnMouseWheel(MouseWheelEvent{ int(std::ceil(float(e.x) * scaling)),
                                                           int(std::ceil(float(e.y) * scaling)) });
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP:
                {
                    auto &e = event->button;
                    MouseButton button = MouseButton::NONE;
                    switch (e.button) {
                        case SDL_BUTTON_LEFT: button = MouseButton::LEFT; break;
                        case SDL_BUTTON_RIGHT: button = MouseButton::RIGHT; break;
                        case SDL_BUTTON_MIDDLE: button = MouseButton::MIDDLE; break;
                        case SDL_BUTTON_X1: button = MouseButton::BUTTON4; break;
                        case SDL_BUTTON_X2: button = MouseButton::BUTTON5; break;
                    }
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto type = (event->type == SDL_MOUSEBUTTONDOWN
                                         ? MouseButtonEvent::DOWN
                                         : MouseButtonEvent::UP);
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        win->OnMouseButton(MouseButtonEvent{
                                                type,
                                                int(std::ceil(float(e.x) * scaling)),
                                                int(std::ceil(float(e.y) * scaling)),
                                                button, });
                    }
                    break;
                }
                case SDL_TEXTINPUT: {
                    auto &e = event->text;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        win->OnTextInput(TextInputEvent{ e.text });
                    }
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
//                    int key = event->key.keysym.scancode;
//                    IM_ASSERT(key >= 0 && key < IM_ARRAYSIZE(io.KeysDown));
                    break;
                }
                case SDL_DROPFILE: {
                    SDL_free(event->drop.file);
                    break;
                }
                case SDL_WINDOWEVENT: {
                    auto &e = event->window;
                    auto wIt = impl_->windows.find(e.windowID);
                    if (wIt == impl_->windows.end()) {
                        break;
                    }
                    auto window = wIt->second;
                    switch (e.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            window->OnResize();
                            break;
                        case SDL_WINDOWEVENT_CLOSE:
                            CloseWindow(window);
                            break;
                        default:
                            break;
                    }
                    break;
                }
                default:
                    break;
            }
            nevents++;
        }

        for (auto &kv : impl_->windows) {
            auto w = kv.second;
            if (w->IsVisible()) {
                w->OnDraw(float(RUNLOOP_DELAY_MSEC) / 1000.0);
            }
        }

        SDL_Delay(RUNLOOP_DELAY_MSEC);
    }

    SDL_Quit();
}

void Application::CloseWindow(std::shared_ptr<Window> window) {
    impl_->windows.erase(window->GetID());
}

const char* Application::GetResourcePath() const {
    return impl_->resourcePath.c_str();
}

const Theme& Application::GetTheme() const {
    return impl_->theme;
}

} // gui
} // open3d
