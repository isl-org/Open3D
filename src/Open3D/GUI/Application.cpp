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

#include "Events.h"
#include "Theme.h"
#include "Window.h"

#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"

#include <SDL.h>

#include <chrono>
#include <thread>
#include <unordered_map>

#include <sys/stat.h>
#include <sys/types.h>
#if !defined(WIN32)
#include <unistd.h>
#else
#include <direct.h>
#include <io.h>

// Copy-paste from UNIX <sys/stat.h> sources
#define S_ISDIR(mask) ((mask & S_IFMT) == S_IFDIR)
#endif

#ifdef WIN32
#define getcwd _getcwd
#endif  // WIN32

namespace {

const int RUNLOOP_DELAY_MSEC = 10;

std::unordered_map<int, uint32_t> SCANCODE2KEY = {
        {SDL_SCANCODE_BACKSPACE, open3d::gui::KEY_BACKSPACE},
        {SDL_SCANCODE_TAB, open3d::gui::KEY_TAB},
        {SDL_SCANCODE_RETURN, open3d::gui::KEY_ENTER},
        {SDL_SCANCODE_ESCAPE, open3d::gui::KEY_ESCAPE},
        {SDL_SCANCODE_DELETE, open3d::gui::KEY_DELETE},
        {SDL_SCANCODE_SPACE, ' '},
        {SDL_SCANCODE_0, '0'},
        {SDL_SCANCODE_1, '1'},
        {SDL_SCANCODE_2, '2'},
        {SDL_SCANCODE_3, '3'},
        {SDL_SCANCODE_4, '4'},
        {SDL_SCANCODE_5, '5'},
        {SDL_SCANCODE_6, '6'},
        {SDL_SCANCODE_7, '7'},
        {SDL_SCANCODE_8, '8'},
        {SDL_SCANCODE_9, '9'},
        {SDL_SCANCODE_A, 'a'},
        {SDL_SCANCODE_B, 'b'},
        {SDL_SCANCODE_C, 'c'},
        {SDL_SCANCODE_D, 'd'},
        {SDL_SCANCODE_E, 'e'},
        {SDL_SCANCODE_F, 'f'},
        {SDL_SCANCODE_G, 'g'},
        {SDL_SCANCODE_H, 'h'},
        {SDL_SCANCODE_I, 'i'},
        {SDL_SCANCODE_J, 'j'},
        {SDL_SCANCODE_K, 'k'},
        {SDL_SCANCODE_L, 'l'},
        {SDL_SCANCODE_M, 'm'},
        {SDL_SCANCODE_N, 'n'},
        {SDL_SCANCODE_O, 'o'},
        {SDL_SCANCODE_P, 'p'},
        {SDL_SCANCODE_Q, 'q'},
        {SDL_SCANCODE_R, 'r'},
        {SDL_SCANCODE_S, 's'},
        {SDL_SCANCODE_T, 't'},
        {SDL_SCANCODE_U, 'u'},
        {SDL_SCANCODE_V, 'v'},
        {SDL_SCANCODE_W, 'w'},
        {SDL_SCANCODE_X, 'x'},
        {SDL_SCANCODE_Y, 'y'},
        {SDL_SCANCODE_Z, 'z'},
        {SDL_SCANCODE_LEFTBRACKET, '['},
        {SDL_SCANCODE_RIGHTBRACKET, ']'},
        {SDL_SCANCODE_BACKSLASH, '\\'},
        {SDL_SCANCODE_SEMICOLON, ';'},
        {SDL_SCANCODE_APOSTROPHE, '\''},
        {SDL_SCANCODE_COMMA, ','},
        {SDL_SCANCODE_PERIOD, '.'},
        {SDL_SCANCODE_SLASH, '/'},
        {SDL_SCANCODE_LEFT, open3d::gui::KEY_LEFT},
        {SDL_SCANCODE_RIGHT, open3d::gui::KEY_RIGHT},
        {SDL_SCANCODE_UP, open3d::gui::KEY_UP},
        {SDL_SCANCODE_DOWN, open3d::gui::KEY_DOWN},
        {SDL_SCANCODE_INSERT, open3d::gui::KEY_INSERT},
        {SDL_SCANCODE_HOME, open3d::gui::KEY_HOME},
        {SDL_SCANCODE_END, open3d::gui::KEY_END},
        {SDL_SCANCODE_PAGEUP, open3d::gui::KEY_PAGEUP},
        {SDL_SCANCODE_PAGEDOWN, open3d::gui::KEY_PAGEDOWN}};

bool isDirectory(const std::string &path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) return false;

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

    if (argv0[0] == '/' ||
        (argv0.size() > 3 && argv0[1] == ':' && argv0[2] == '/')) {
        // is absolute path, we're done
    } else {
        // relative path:  prepend working directory
        char *cwd = getcwd(nullptr, 0);  // will malloc()
        path = std::string(cwd) + "/" + path;
        free(cwd);
    }

#ifdef __APPLE__
    if (path.rfind("MacOS") == path.size() - 5) {  // path is in a bundle
        return path.substr(0, path.size() - 5) + "Resources";
    }
#endif  // __APPLE__

    auto rsrcPath = path + "/resources";
    if (!isDirectory(rsrcPath)) {
        return path + "/../resources";  // building with Xcode
    }
    return rsrcPath;
}

}  // namespace

namespace open3d {
namespace gui {

struct Application::Impl {
    std::string resourcePath;
    std::unordered_map<uint32_t, std::shared_ptr<Window>> windows;
    Theme theme;
};

Application &Application::GetInstance() {
    static Application gApp;
    return gApp;
}

Application::Application()
: impl_(new Application::Impl()) {
    Color highlightColor(0.5, 0.5, 0.5);

    // Note that any values here need to be scaled by the scale factor in Window
    impl_->theme.fontPath = "Roboto-Medium.ttf";  // full path will be added in Initialize()
    impl_->theme.fontSize = 16; // 1 em (font size is em in digital type)
    impl_->theme.defaultMargin = 8; // 0.5 * em
    impl_->theme.defaultLayoutSpacing = 6; // 0.333 * em

    impl_->theme.backgroundColor = Color(0.175, 0.175, 0.175);
    impl_->theme.textColor = Color(0.875, 0.875, 0.875);
    impl_->theme.borderWidth = 1;
    impl_->theme.borderRadius = 3;
    impl_->theme.borderColor = Color(0.5, 0.5, 0.5);
    impl_->theme.menubarBorderColor = Color(0.25, 0.25, 0.25);
    impl_->theme.buttonColor = Color(0.4, 0.4, 0.4);
    impl_->theme.buttonHoverColor = Color(0.6, 0.6, 0.6);
    impl_->theme.buttonActiveColor = Color(0.5, 0.5, 0.5);
    impl_->theme.checkboxBackgroundOffColor = Color(0.333, 0.333, .333);
    impl_->theme.checkboxBackgroundOnColor = highlightColor;
    impl_->theme.checkboxBackgroundHoverOffColor = Color(0.5, 0.5, 0.5);
    impl_->theme.checkboxBackgroundHoverOnColor =
            highlightColor.Lightened(0.15);
    impl_->theme.checkboxCheckColor = Color(1, 1, 1);
    impl_->theme.comboboxBackgroundColor = Color(0.4, 0.4, 0.4);
    impl_->theme.comboboxHoverColor = Color(0.5, 0.5, 0.5);
    impl_->theme.comboboxArrowBackgroundColor = highlightColor;
    impl_->theme.sliderGrabColor = Color(0.666, 0.666, 0.666);
    impl_->theme.textEditBackgroundColor = Color(0.25, 0.25, 0.25);
    impl_->theme.tabInactiveColor = impl_->theme.buttonColor;
    impl_->theme.tabHoverColor = impl_->theme.buttonHoverColor;
    impl_->theme.tabActiveColor = impl_->theme.buttonActiveColor;

    visualization::EngineInstance::SelectBackend(
            filament::backend::Backend::OPENGL);
}

Application::~Application() {}

void Application::Initialize() {
    // We don't have a great way of getting the process name, so let's hope that
    // the current directory is where the resources are located. This is a
    // safe assumption when running on macOS and Windows normally.
    char *path = getcwd(NULL, 4096 /* ignored, but make it large just in case */);
    Initialize(1, (const char **)&path);
    free(path);
}

void Application::Initialize(int argc, const char *argv[]) {
    impl_->resourcePath = findResourcePath(argc, argv);
    impl_->theme.fontPath = impl_->resourcePath + "/" + impl_->theme.fontPath;
}

void Application::AddWindow(std::shared_ptr<Window> window) {
    window->OnResize();  // so we get an initial resize
    impl_->windows[window->GetID()] = window;
}

void Application::RemoveWindow(Window *window) {
    // SDL_DestroyWindow doesn't send SDL_WINDOWEVENT_CLOSED or SDL_QUIT
    // messages, so we have to do them ourselves.
    int nWindows = impl_->windows.size();

    SDL_Event e;
    e.type = SDL_WINDOWEVENT;
    e.window.windowID = window->GetID();
    e.window.event = SDL_WINDOWEVENT_CLOSE;
    SDL_PushEvent(&e);

    if (nWindows == 1) {
        SDL_Event quit;
        quit.type = SDL_QUIT;
        SDL_PushEvent(&quit);
    }
}

void Application::Run() {
    SDL_Init(SDL_INIT_EVENTS);

    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);

    bool done = false;
    std::unordered_map<Window *, int> eventCounts;
    while (!done) {
        //        SDL_Window* sdlWindow = window->getSDLWindow();
        //        if (mWindowTitle != SDL_GetWindowTitle(sdlWindow)) {
        //            SDL_SetWindowTitle(sdlWindow, mWindowTitle.c_str());
        //        }

        //        if (!UTILS_HAS_THREADING) {
        //            mEngine->execute();
        //        }

        eventCounts.clear();
        constexpr int kMaxEvents = 16;
        SDL_Event events[kMaxEvents];
        int nevents = 0;
        while (nevents < kMaxEvents && SDL_PollEvent(&events[nevents]) != 0) {
            SDL_Event* event = &events[nevents];
            switch (event->type) {
                case SDL_QUIT:  // sent after last window closed
                    done = true;
                    break;
                case SDL_MOUSEMOTION: {
                    auto &e = event->motion;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        auto type = (e.state == 0 ? MouseEvent::MOVE
                                                  : MouseEvent::DRAG);
                        int x = int(std::ceil(float(e.x) * scaling));
                        int y = int(std::ceil(float(e.y) * scaling));
                        int buttons = 0;
                        if (e.state & SDL_BUTTON_LEFT) {
                            buttons |= int(MouseButton::LEFT);
                        }
                        if (e.state & SDL_BUTTON_RIGHT) {
                            buttons |= int(MouseButton::RIGHT);
                        }
                        if (e.state & SDL_BUTTON_MIDDLE) {
                            buttons |= int(MouseButton::MIDDLE);
                        }
                        if (e.state & SDL_BUTTON_X1) {
                            buttons |= int(MouseButton::BUTTON4);
                        }
                        if (e.state & SDL_BUTTON_X2) {
                            buttons |= int(MouseButton::BUTTON5);
                        }
                        MouseEvent me = { type, x, y };
                        me.move = {buttons};

                        win->OnMouseEvent(me);
                        eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_MOUSEWHEEL: {
                    auto &e = event->wheel;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        int mx, my;
                        SDL_GetGlobalMouseState(&mx, &my);
                        auto pos = win->GlobalToWindowCoord(mx, my);
                        int dx = int(std::ceil(float(e.x) * scaling));
                        int dy = int(std::ceil(float(e.y) * scaling));
                        MouseEvent me = { MouseEvent::WHEEL, pos.x, pos.y };
                        me.wheel = {dx, dy};

                        win->OnMouseEvent(me);
                        eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_MOUSEBUTTONDOWN:
                case SDL_MOUSEBUTTONUP: {
                    auto &e = event->button;
                    MouseButton button = MouseButton::NONE;
                    switch (e.button) {
                        case SDL_BUTTON_LEFT:
                            button = MouseButton::LEFT;
                            break;
                        case SDL_BUTTON_RIGHT:
                            button = MouseButton::RIGHT;
                            break;
                        case SDL_BUTTON_MIDDLE:
                            button = MouseButton::MIDDLE;
                            break;
                        case SDL_BUTTON_X1:
                            button = MouseButton::BUTTON4;
                            break;
                        case SDL_BUTTON_X2:
                            button = MouseButton::BUTTON5;
                            break;
                    }
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto type = (event->type == SDL_MOUSEBUTTONDOWN
                                         ? MouseEvent::BUTTON_DOWN
                                         : MouseEvent::BUTTON_UP);
                        auto &win = it->second;
                        auto scaling = win->GetScaling();
                        int x = int(std::ceil(float(e.x) * scaling));
                        int y = int(std::ceil(float(e.y) * scaling));
                        MouseEvent me = { type, x, y };
                        me.button = { button };

                        win->OnMouseEvent(me);
                        eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_TEXTINPUT: {
                    auto &e = event->text;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        win->OnTextInput(TextInputEvent{e.text});
                        eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    auto &e = event->key;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        auto type = (event->type == SDL_KEYDOWN ? KeyEvent::DOWN
                                                                : KeyEvent::UP);
                        uint32_t key = KEY_UNKNOWN;
                        auto it = SCANCODE2KEY.find(e.keysym.scancode);
                        if (it != SCANCODE2KEY.end()) {
                            key = it->second;
                        }
                        win->OnKeyEvent(KeyEvent{ type, key, (e.repeat != 0) });
                        eventCounts[win.get()] += 1;
                    }
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
                            impl_->windows.erase(window->GetID());
                            break;
                        default:
                            break;
                    }
                    eventCounts[window.get()] += 1;
                    break;
                }
                default:
                    break;
            }
            nevents++;
        }

        for (auto &kv : impl_->windows) {
            auto w = kv.second;
            bool gotEvents = (eventCounts.find(w.get()) != eventCounts.end());
            if (w->IsVisible() && gotEvents) {
                if (w->DrawOnce(float(RUNLOOP_DELAY_MSEC) / 1000.0) ==
                    Window::REDRAW) {
                    SDL_Event expose;
                    expose.type = SDL_WINDOWEVENT;
                    expose.window.windowID = w->GetID();
                    expose.window.event = SDL_WINDOWEVENT_EXPOSED;
                    SDL_PushEvent(&expose);
                }
            }
        }

        SDL_Delay(RUNLOOP_DELAY_MSEC);
    }

    SDL_Quit();
}

const char *Application::GetResourcePath() const {
    return impl_->resourcePath.c_str();
}

const Theme &Application::GetTheme() const { return impl_->theme; }

}  // namespace gui
}  // namespace open3d
