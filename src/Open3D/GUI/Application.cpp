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

#include "Button.h"
#include "Events.h"
#include "Label.h"
#include "Layout.h"
#include "Native.h"
#include "Theme.h"
#include "Window.h"

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"

#include <SDL.h>

#include <chrono>
#include <thread>
#include <unordered_map>

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
        {SDL_SCANCODE_PAGEDOWN, open3d::gui::KEY_PAGEDOWN},
        {SDL_SCANCODE_LSHIFT, open3d::gui::KEY_LSHIFT},
        {SDL_SCANCODE_RSHIFT, open3d::gui::KEY_RSHIFT},
        {SDL_SCANCODE_LCTRL, open3d::gui::KEY_LCTRL},
        {SDL_SCANCODE_RCTRL, open3d::gui::KEY_RCTRL},
        {SDL_SCANCODE_LALT, open3d::gui::KEY_ALT},
        {SDL_SCANCODE_RALT, open3d::gui::KEY_ALT},
        {SDL_SCANCODE_LGUI, open3d::gui::KEY_META},
        {SDL_SCANCODE_RGUI, open3d::gui::KEY_META}};

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
        auto cwd = open3d::utility::filesystem::GetWorkingDirectory();
        path = cwd + "/" + path;
    }

#ifdef __APPLE__
    if (path.rfind("MacOS") == path.size() - 5) {  // path is in a bundle
        return path.substr(0, path.size() - 5) + "Resources";
    }
#endif  // __APPLE__

    auto rsrcPath = path + "/resources";
    if (!open3d::utility::filesystem::DirectoryExists(rsrcPath)) {
        return path + "/../resources";  // building with Xcode
    }
    return rsrcPath;
}

int keyModsRightNow() {  // requires SDL is initialized
    int keyMods = 0;
    auto sdlMods = SDL_GetModState();
    if ((sdlMods & KMOD_LSHIFT) || (sdlMods & KMOD_RSHIFT)) {
        keyMods |= int(open3d::gui::KeyModifier::SHIFT);
    }
    if ((sdlMods & KMOD_LCTRL) || (sdlMods & KMOD_RCTRL)) {
        keyMods |= int(open3d::gui::KeyModifier::CTRL);
    }
    if ((sdlMods & KMOD_LALT) || (sdlMods & KMOD_RALT)) {
        keyMods |= int(open3d::gui::KeyModifier::ALT);
    }
    if ((sdlMods & KMOD_LGUI) || (sdlMods & KMOD_RGUI)) {
        keyMods |= int(open3d::gui::KeyModifier::META);
    }
    return keyMods;
}

}  // namespace

namespace open3d {
namespace gui {

struct Application::Impl {
    std::string resourcePath;
    Theme theme;
    bool isRunning = false;

    std::unordered_map<uint32_t, std::shared_ptr<Window>> windows;
    std::unordered_map<Window *, int> eventCounts;  // don't recreate each draw

    // We keep track of our own key states becase SDL_GetModState()
    // gets the instantaneous state, whereas we need the state at the
    // time the event happened, which may not be the same, since we
    // process events in batches.
    struct {
        bool lShift = false;
        bool rShift = false;
        bool lCtrl = false;
        bool rCtrl = false;
        bool lAlt = false;
        bool rAlt = false;
        bool lMeta = false;
        bool rMeta = false;
    } keyStates;
    int keyMods = 0;
};

Application &Application::GetInstance() {
    static Application gApp;
    return gApp;
}

void Application::ShowMessageBox(const char *title, const char *message) {
    utility::LogInfo("%s", message);

    auto alert = std::make_shared<Window>("Alert", Window::FLAG_TOPMOST);
    auto em = alert->GetTheme().fontSize;
    auto layout = std::make_shared<Vert>(em, Margins(em));
    auto msg = std::make_shared<Label>(message);
    auto ok = std::make_shared<Button>("Ok");
    ok->SetOnClicked([alert = alert.get() /*avoid shared_ptr cycle*/]() {
        Application::GetInstance().RemoveWindow(alert);
    });
    layout->AddChild(Horiz::MakeCentered(msg));
    layout->AddChild(Horiz::MakeCentered(ok));
    alert->AddChild(layout);
    Application::GetInstance().AddWindow(alert);
}

Application::Application() : impl_(new Application::Impl()) {
    Color highlightColor(0.5, 0.5, 0.5);

    // Note that any values here need to be scaled by the scale factor in Window
    impl_->theme.fontPath =
            "Roboto-Medium.ttf";     // full path will be added in Initialize()
    impl_->theme.fontSize = 16;      // 1 em (font size is em in digital type)
    impl_->theme.defaultMargin = 8;  // 0.5 * em
    impl_->theme.defaultLayoutSpacing = 6;  // 0.333 * em

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
    impl_->theme.dialogBorderWidth = 1;
    impl_->theme.dialogBorderRadius = 10;

    visualization::EngineInstance::SelectBackend(
            filament::backend::Backend::OPENGL);
}

Application::~Application() {}

void Application::Initialize() {
    // We don't have a great way of getting the process name, so let's hope that
    // the current directory is where the resources are located. This is a
    // safe assumption when running on macOS and Windows normally.
    auto path = open3d::utility::filesystem::GetWorkingDirectory();
    // Copy to C string, as some implementations of std::string::c_str()
    // return a very temporary pointer.
    char *argv = strdup(path.c_str());
    Initialize(1, (const char **)&argv);
    free(argv);
}

void Application::Initialize(int argc, const char *argv[]) {
    impl_->resourcePath = findResourcePath(argc, argv);
    impl_->theme.fontPath = impl_->resourcePath + "/" + impl_->theme.fontPath;
}

void Application::AddWindow(std::shared_ptr<Window> window) {
    window->OnResize();  // so we get an initial resize
    window->Show();
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
    while (RunOneTick()) {
        SDL_Delay(RUNLOOP_DELAY_MSEC);
    }
}

bool Application::RunOneTick() {
    // Initialize if we have not started yet
    if (!impl_->isRunning) {
        // Verify that the resource path is valid. If it is not, display a
        // message box (std::cerr may not be visible to the user, if we were run
        // as app).
        if (impl_->resourcePath.empty()) {
            ShowNativeAlert(
                    "Internal error: Application::Initialize() was not called");
            return false;
        }
        if (!utility::filesystem::DirectoryExists(impl_->resourcePath)) {
            std::stringstream err;
            err << "Could not find resource directory:\n'"
                << impl_->resourcePath << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }
        if (!utility::filesystem::FileExists(impl_->theme.fontPath)) {
            std::stringstream err;
            err << "Could not load UI font:\n'" << impl_->theme.fontPath
                << "' does not exist";
            ShowNativeAlert(err.str().c_str());
            return false;
        }

        SDL_Init(SDL_INIT_EVENTS);
        SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
        impl_->keyMods = keyModsRightNow();
        impl_->isRunning = true;
    }

    // Process the events that have queued up
    auto status = ProcessQueuedEvents();

    // Cleanup if we are done
    if (status == RunStatus::DONE) {
        SDL_Quit();
        impl_->isRunning = false;
    }

    return impl_->isRunning;
}

Application::RunStatus Application::ProcessQueuedEvents() {
    auto status = RunStatus::CONTINUE;

    impl_->eventCounts.clear();
    constexpr int kMaxEvents = 16;
    SDL_Event events[kMaxEvents];
    int nevents = 0;
    {
        while (nevents < kMaxEvents && SDL_PollEvent(&events[nevents]) != 0) {
            SDL_Event *event = &events[nevents];
            switch (event->type) {
                case SDL_QUIT:  // sent after last window closed
                    status = RunStatus::DONE;
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

                        MouseEvent me = {type, x, y, impl_->keyMods};
                        // GCC complains when trying to initialize inline above
                        me.move.buttons = buttons;

                        win->OnMouseEvent(me);
                        impl_->eventCounts[win.get()] += 1;
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

                        MouseEvent me = {MouseEvent::WHEEL, pos.x, pos.y,
                                         impl_->keyMods};
                        me.wheel.dx = dx;
                        me.wheel.dy = dy;
#if __APPLE__
                        // SDL is supposed to set e.which to the mouse button or
                        // to SDL_TOUCH_MOUSEID, but it doesn't, so we have no
                        // way of knowing which one the user is using.
                        me.wheel.isTrackpad = true;
#else
                        me.wheel.isTrackpad = (e.which == SDL_TOUCH_MOUSEID);
#endif  // __APPLE__

                        win->OnMouseEvent(me);
                        impl_->eventCounts[win.get()] += 1;
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

                        MouseEvent me = {type, x, y, impl_->keyMods};
                        me.button.button = button;

                        win->OnMouseEvent(me);
                        impl_->eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_TEXTINPUT: {
                    auto &e = event->text;
                    auto it = impl_->windows.find(e.windowID);
                    if (it != impl_->windows.end()) {
                        auto &win = it->second;
                        win->OnTextInput(TextInputEvent{e.text});
                        impl_->eventCounts[win.get()] += 1;
                    }
                    break;
                }
                case SDL_KEYDOWN:
                case SDL_KEYUP: {
                    auto &e = event->key;

                    // Update modifier keys. We compare (type != keyup) rather
                    // than (type == keydown) in case SDL adds SDL_KEYREPEAT
                    // in the future.
                    auto& keyStates = impl_->keyStates; // helps line lengths
                    switch (e.keysym.scancode) {
                        case SDL_SCANCODE_LSHIFT:
                            keyStates.lShift = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_RSHIFT:
                            keyStates.rShift = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_LCTRL:
                            keyStates.lCtrl = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_RCTRL:
                            keyStates.rCtrl = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_LALT:
                            keyStates.lAlt = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_RALT:
                            keyStates.rAlt = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_LGUI:
                            keyStates.lMeta = (event->type != SDL_KEYUP);
                            break;
                        case SDL_SCANCODE_RGUI:
                            keyStates.rMeta = (event->type != SDL_KEYUP);
                            break;
                        default:
                            break;
                    }
                    int keyMods = 0;
                    if (keyStates.lShift || keyStates.rShift) {
                        keyMods |= int(KeyModifier::SHIFT);
                    }
#ifdef __APPLE__
                    if (keyStates.lCtrl || keyStates.rCtrl) {
                        keyMods |= int(KeyModifier::ALT);
                    }
                    if (keyStates.lAlt || keyStates.rAlt) {
                        keyMods |= int(KeyModifier::META);
                    }
                    if (keyStates.lMeta || keyStates.rMeta) {
                        keyMods |= int(KeyModifier::CTRL);
                    }
#else
                    if (keyStates.lCtrl || keyStates.rCtrl) {
                        keyMods |= int(KeyModifier::CTRL);
                    }
                    if (keyStates.lAlt || keyStates.rAlt) {
                        keyMods |= int(KeyModifier::ALT);
                    }
                    if (keyStates.lMeta || keyStates.rMeta) {
                        keyMods |= int(KeyModifier::META);
                    }
#endif // __APPLE__
                    impl_->keyMods = keyMods;

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
                        win->OnKeyEvent(KeyEvent{type, key, (e.repeat != 0)});
                        impl_->eventCounts[win.get()] += 1;
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
                        case SDL_WINDOWEVENT_FOCUS_GAINED:
                            // The user might have pressed or unpressed a key
                            // while another window was focused, (most notably
                            // Alt-Tab), so recalc mods.
                            impl_->keyMods = keyModsRightNow();
                            break;
                        default:
                            break;
                    }
                    impl_->eventCounts[window.get()] += 1;
                    break;
                }
                default:
                    break;
            }
            nevents++;
        }

        for (auto &kv : impl_->windows) {
            auto w = kv.second;
            bool gotEvents = (impl_->eventCounts.find(w.get()) !=
                              impl_->eventCounts.end());
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
    }

    return status;
}

const char *Application::GetResourcePath() const {
    return impl_->resourcePath.c_str();
}

const Theme &Application::GetTheme() const { return impl_->theme; }

}  // namespace gui
}  // namespace open3d
