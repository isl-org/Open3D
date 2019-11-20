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

#include <iostream> // debugging

namespace {

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
};

Application& Application::GetInstance() {
    static Application gApp;
    return gApp;
}

Application::Application()
    : impl_(new Application::Impl()) {
}

Application::~Application() {
}

void Application::Initialize(int argc, const char *argv[]) {
    impl_->resourcePath = findResourcePath(argc, argv);
}

void Application::AddWindow(std::shared_ptr<Window> window) {
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

        // Loop over fresh events twice: first stash them and let ImGui process them, then allow
        // the app to process the stashed events. This is done because ImGui might wish to block
        // certain events from the app (e.g., when dragging the mouse over an obscuring window).
        constexpr int kMaxEvents = 16;
        SDL_Event events[kMaxEvents];
        int nevents = 0;
        while (nevents < kMaxEvents && SDL_PollEvent(&events[nevents]) != 0) {
//            ImGuiIO& io = ImGui::GetIO();
            SDL_Event* event = &events[nevents];
            switch (event->type) {
                case SDL_MOUSEWHEEL: {
//                    if (event->wheel.x > 0) io.MouseWheelH += 1;
//                    if (event->wheel.x < 0) io.MouseWheelH -= 1;
//                    if (event->wheel.y > 0) io.MouseWheel += 1;
//                    if (event->wheel.y < 0) io.MouseWheel -= 1;
                    break;
                }
                case SDL_MOUSEBUTTONDOWN: {
//                    if (event->button.button == SDL_BUTTON_LEFT) mousePressed[0] = true;
//                    if (event->button.button == SDL_BUTTON_RIGHT) mousePressed[1] = true;
//                    if (event->button.button == SDL_BUTTON_MIDDLE) mousePressed[2] = true;
                    break;
                }
                case SDL_TEXTINPUT: {
//                    io.AddInputCharactersUTF8(event->text.text);
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
//                    int key = event->key.keysym.scancode;
//                    IM_ASSERT(key >= 0 && key < IM_ARRAYSIZE(io.KeysDown));
//                    io.KeysDown[key] = (event->type == SDL_KEYDOWN);
//                    io.KeyShift = ((SDL_GetModState() & KMOD_SHIFT) != 0);
//                    io.KeyAlt = ((SDL_GetModState() & KMOD_ALT) != 0);
//                    io.KeyCtrl = ((SDL_GetModState() & KMOD_CTRL) != 0);
//                    io.KeySuper = ((SDL_GetModState() & KMOD_GUI) != 0);
                    break;
                }
            }
            nevents++;
        }

        // Now, loop over the events a second time for app-side processing.
        for (int i = 0; i < nevents; i++) {
            const SDL_Event& event = events[i];
//            ImGuiIO* io = mImGuiHelper ? &ImGui::GetIO() : nullptr;
            switch (event.type) {
                case SDL_QUIT:   // sent after last window closed
                    done = true;
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) {
//                        mClosed = true;
                    }
                    break;
                case SDL_MOUSEWHEEL:
//                    if (!io || !io->WantCaptureMouse)
//                        window->mouseWheel(event.wheel.y);
                    break;
                case SDL_MOUSEBUTTONDOWN:
//                    if (!io || !io->WantCaptureMouse)
//                        window->mouseDown(event.button.button, event.button.x, event.button.y);
                    break;
                case SDL_MOUSEBUTTONUP:
//                    if (!io || !io->WantCaptureMouse)
//                        window->mouseUp(event.button.x, event.button.y);
                    break;
                case SDL_MOUSEMOTION:
//                    if (!io || !io->WantCaptureMouse)
//                        window->mouseMoved(event.motion.x, event.motion.y);
                    break;
                case SDL_DROPFILE:
//                    if (mDropHandler) {
//                        mDropHandler(event.drop.file);
//                    }
                    SDL_free(event.drop.file);
                    break;
                case SDL_WINDOWEVENT: {
                    auto wIt = impl_->windows.find(event.window.windowID);
                    if (wIt == impl_->windows.end()) {
                        break;
                    }
                    auto window = wIt->second;
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            window->OnResize();
                            break;
                        case SDL_WINDOWEVENT_CLOSE:  // sent if not last window
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
        }

        for (auto &kv : impl_->windows) {
            kv.second->OnDraw();
        }

        SDL_Delay(10); // millisec
    }

    SDL_Quit();
}

void Application::CloseWindow(std::shared_ptr<Window> window) {
    impl_->windows.erase(window->GetID());
}

const char* Application::GetResourcePath() const {
    return impl_->resourcePath.c_str();
}

} // gui
} // open3d
