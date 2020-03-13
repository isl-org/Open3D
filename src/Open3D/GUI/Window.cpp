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

#include "Window.h"

#include "Application.h"
#include "Button.h"
#include "Dialog.h"
#include "ImguiFilamentBridge.h"
#include "Label.h"
#include "Layout.h"
#include "Menu.h"
#include "Native.h"
#include "Theme.h"
#include "Util.h"
#include "Widget.h"

#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentRenderer.h"

#include <SDL.h>
#include <filament/Engine.h>
#include <imgui.h>
#include <imgui_internal.h>

#include <cmath>
#include <queue>
#include <vector>

#ifdef WIN32
#include <SDL_syswm.h>
#endif

using namespace open3d::gui::util;

// ----------------------------------------------------------------------------
namespace open3d {
namespace gui {

namespace {

static constexpr int CENTERED_X = -10000;
static constexpr int CENTERED_Y = -10000;
static constexpr int AUTOSIZE_WIDTH = 0;
static constexpr int AUTOSIZE_HEIGHT = 0;

// Assumes the correct ImGuiContext is current
void updateImGuiForScaling(float newScaling) {
    ImGuiStyle& style = ImGui::GetStyle();
    // FrameBorderSize is not adjusted (we want minimal borders)
    style.FrameRounding *= newScaling;
}

}  // namespace

const int Window::FLAG_TOPMOST = (1 << 0);

struct Window::Impl {
    SDL_Window* window = nullptr;
    Theme theme;  // so that the font size can be different based on scaling
    visualization::FilamentRenderer* renderer;
    struct {
        std::unique_ptr<ImguiFilamentBridge> imguiBridge = nullptr;
        ImGuiContext* context;
        ImFont* systemFont;  // is a reference; owned by imguiContext
        float scaling = 1.0;
    } imgui;
    std::vector<std::shared_ptr<Widget>> children;

    // Active dialog is owned here. It is not put in the children because
    // we are going to add it and take it out during draw (since that's
    // how an immediate mode GUI works) and that involves changing the
    // children while iterating over it. Also, conceptually it is not a
    // child, it is a child window, and needs to be on top, which we cannot
    // guarantee if it is a child widget.
    std::shared_ptr<Dialog> activeDialog;

    std::queue<std::function<void()>> deferredUntilBeforeDraw;
    std::queue<std::function<void()>> deferredUntilDraw;
    Widget* focusWidget =
            nullptr;  // only used if ImGUI isn't taking keystrokes
    double lastFrameTime = 0.0;
    int nSkippedFrames = 0;
    bool wantsAutoSizeAndCenter = false;
    bool needsLayout = true;
};

Window::Window(const std::string& title, int flags /*= 0*/)
    : Window(title, CENTERED_X, CENTERED_Y, AUTOSIZE_WIDTH, AUTOSIZE_HEIGHT) {}

Window::Window(const std::string& title,
               int width,
               int height,
               int flags /*= 0*/)
    : Window(title, CENTERED_X, CENTERED_Y, width, height) {}

Window::Window(const std::string& title,
               int x,
               int y,
               int width,
               int height,
               int flags /*= 0*/)
    : impl_(new Window::Impl()) {
    if (x == CENTERED_X) {
        x = SDL_WINDOWPOS_CENTERED;
    }
    if (y == CENTERED_Y) {
        y = SDL_WINDOWPOS_CENTERED;
    }
    if (width == AUTOSIZE_WIDTH || height == AUTOSIZE_HEIGHT) {
        impl_->wantsAutoSizeAndCenter = true;
    }
    uint32_t sdlflags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    if (sdlflags & FLAG_TOPMOST) {
        sdlflags |= SDL_WINDOW_ALWAYS_ON_TOP;
    }
    impl_->window =
            SDL_CreateWindow(title.c_str(), x, y, width, height, sdlflags);

    // On single-threaded platforms, Filament's OpenGL context must be current,
    // not SDL's context, so create the renderer after the window.

    // ImGUI creates a bitmap atlas from a font, so we need to have the correct
    // size when we create it, because we can't change the bitmap without
    // reloading the whole thing (expensive).
    float scaling = GetScaling();
    impl_->theme = Application::GetInstance().GetTheme();
    impl_->theme.fontSize *= scaling;
    impl_->theme.defaultMargin *= scaling;
    impl_->theme.defaultLayoutSpacing *= scaling;

    auto& engineInstance = visualization::EngineInstance::GetInstance();
    auto& resourceManager = visualization::EngineInstance::GetResourceManager();

    impl_->renderer = new visualization::FilamentRenderer(
            engineInstance, GetNativeDrawable(), resourceManager);

    auto& theme = impl_->theme;  // shorter alias
    impl_->imgui.context = ImGui::CreateContext();
    auto oldContext = MakeCurrent();

    impl_->imgui.imguiBridge =
            std::make_unique<ImguiFilamentBridge>(impl_->renderer, GetSize());

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(0, 0);
    style.WindowRounding = 0;
    style.WindowBorderSize = 0;
    style.FrameBorderSize = theme.borderWidth;
    style.FrameRounding = theme.borderRadius;
    style.Colors[ImGuiCol_WindowBg] = colorToImgui(theme.backgroundColor);
    style.Colors[ImGuiCol_Text] = colorToImgui(theme.textColor);
    style.Colors[ImGuiCol_Border] = colorToImgui(theme.borderColor);
    style.Colors[ImGuiCol_Button] = colorToImgui(theme.buttonColor);
    style.Colors[ImGuiCol_ButtonHovered] = colorToImgui(theme.buttonHoverColor);
    style.Colors[ImGuiCol_ButtonActive] = colorToImgui(theme.buttonActiveColor);
    style.Colors[ImGuiCol_CheckMark] = colorToImgui(theme.checkboxCheckColor);
    style.Colors[ImGuiCol_FrameBg] =
            colorToImgui(theme.comboboxBackgroundColor);
    style.Colors[ImGuiCol_FrameBgHovered] =
            colorToImgui(theme.comboboxHoverColor);
    style.Colors[ImGuiCol_FrameBgActive] =
            style.Colors[ImGuiCol_FrameBgHovered];
    style.Colors[ImGuiCol_SliderGrab] = colorToImgui(theme.sliderGrabColor);
    style.Colors[ImGuiCol_SliderGrabActive] =
            colorToImgui(theme.sliderGrabColor);
    style.Colors[ImGuiCol_Tab] = colorToImgui(theme.tabInactiveColor);
    style.Colors[ImGuiCol_TabHovered] = colorToImgui(theme.tabHoverColor);
    style.Colors[ImGuiCol_TabActive] = colorToImgui(theme.tabActiveColor);

    // If the given font path is invalid, ImGui will silently fall back to
    // proggy, which is a tiny "pixel art" texture that is compiled into the
    // library.
    if (!theme.fontPath.empty()) {
        ImGuiIO& io = ImGui::GetIO();
        impl_->imgui.systemFont = io.Fonts->AddFontFromFileTTF(
                theme.fontPath.c_str(), theme.fontSize);
        /*static*/ unsigned char* pixels;
        int textureW, textureH, bytesPerPx;
        io.Fonts->GetTexDataAsAlpha8(&pixels, &textureW, &textureH,
                                     &bytesPerPx);
        impl_->imgui.imguiBridge->createAtlasTextureAlpha8(
                pixels, textureW, textureH, bytesPerPx);
    }

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
#ifdef WIN32
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(impl_->window, &wmInfo);
    io.ImeWindowHandle = wmInfo.info.win.window;
#endif
    // ImGUI's io.KeysDown is indexed by our scan codes, and we fill out
    // io.KeyMap to map from our code to ImGui's code.
    io.KeyMap[ImGuiKey_Tab] = KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = KEY_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown] = KEY_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home] = KEY_HOME;
    io.KeyMap[ImGuiKey_End] = KEY_END;
    io.KeyMap[ImGuiKey_Insert] = KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete] = KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = ' ';
    io.KeyMap[ImGuiKey_Enter] = KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = KEY_ESCAPE;
    io.KeyMap[ImGuiKey_A] = 'a';
    io.KeyMap[ImGuiKey_C] = 'c';
    io.KeyMap[ImGuiKey_V] = 'v';
    io.KeyMap[ImGuiKey_X] = 'x';
    io.KeyMap[ImGuiKey_Y] = 'y';
    io.KeyMap[ImGuiKey_Z] = 'z';
    io.SetClipboardTextFn = [](void*, const char* text) {
        SDL_SetClipboardText(text);
    };
    io.GetClipboardTextFn = [](void*) -> const char* {
        return SDL_GetClipboardText();
    };
    io.ClipboardUserData = nullptr;

    // Restore the context, in case we are creating a window during a draw.
    // (This is quite likely, since ImGUI only handles things like button
    // presses during draw. A file open dialog is likely to create a window
    // after pressing "Open".)
    RestoreCurrent(oldContext);
}

Window::~Window() {
    impl_->children.clear();  // needs to happen before deleting renderer
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGui::DestroyContext();
    delete impl_->renderer;
    SDL_DestroyWindow(impl_->window);
}

void* Window::MakeCurrent() const {
    auto oldContext = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(impl_->imgui.context);
    return oldContext;
}

void Window::RestoreCurrent(void* oldContext) const {
    ImGui::SetCurrentContext((ImGuiContext*)oldContext);
}

void* Window::GetNativeDrawable() const {
    return open3d::gui::GetNativeDrawable(impl_->window);
}

uint32_t Window::GetID() const { return SDL_GetWindowID(impl_->window); }

const Theme& Window::GetTheme() const { return impl_->theme; }

visualization::Renderer& Window::GetRenderer() const {
    return *impl_->renderer;
}

Rect Window::GetFrame() const {
    int x, y, w, h;
    SDL_GetWindowPosition(impl_->window, &x, &y);
    SDL_GetWindowSize(impl_->window, &w, &h);
    return Rect(x, y, w, h);
}

void Window::SetFrame(const Rect& r) {
    SDL_SetWindowPosition(impl_->window, r.x, r.y);
    SDL_SetWindowSize(impl_->window, r.width, r.height);
}

const char* Window::GetTitle() const {
    return SDL_GetWindowTitle(impl_->window);
}

void Window::SetTitle(const char* title) {
    SDL_SetWindowTitle(impl_->window, title);
}

// Note: this can only be called during draw!
Size Window::CalcPreferredSize() {
    Rect bbox(0, 0, 0, 0);
    for (auto& child : impl_->children) {
        auto pref = child->CalcPreferredSize(GetTheme());
        Rect r(child->GetFrame().x, child->GetFrame().y, pref.width,
               pref.height);
        bbox = bbox.UnionedWith(r);
    }

    // Note: we are doing (bbox.GetRight() - 0) NOT (bbox.GetRight() - bbox.x)
    //       (and likewise for height) because the origin of the window is
    //       (0, 0) and anything up/left is clipped.
    return Size(bbox.GetRight(), bbox.GetBottom());
}

void Window::SizeToFit() {
    // CalcPreferredSize() can only be called during draw, but we probably
    // aren't calling this in a draw, we are probably setting up the window.
    auto autoSize = [this]() {
        auto pref = CalcPreferredSize();
        SetSize(Size(pref.width / this->impl_->imgui.scaling,
                     pref.height / this->impl_->imgui.scaling));
    };
    impl_->deferredUntilDraw.push(autoSize);
}

void Window::SetSize(const Size& size) {
    // Make sure we do the resize outside of a draw, to avoid unsightly
    // errors if we happen to do this in the middle of a draw.
    auto resize = [this, size /*copy*/]() {
        SDL_SetWindowSize(this->impl_->window, size.width, size.height);
        // SDL_SetWindowSize() doesn't generate an event, so we need to update
        // the size ourselves
        this->OnResize();
    };
    impl_->deferredUntilBeforeDraw.push(resize);
}

Size Window::GetSize() const {
    uint32_t w, h;
    SDL_GL_GetDrawableSize(impl_->window, (int*)&w, (int*)&h);
    return Size(w, h);
}

Rect Window::GetContentRect() const {
    auto size = GetSize();
    int menuHeight = 0;
#if !(GUI_USE_NATIVE_MENUS && defined(__APPLE__))
    MakeCurrent();
    auto menubar = Application::GetInstance().GetMenubar();
    if (menubar) {
        menuHeight = menubar->CalcHeight(GetTheme());
    }
#endif

    return Rect(0, menuHeight, size.width, size.height - menuHeight);
}

float Window::GetScaling() const {
    uint32_t wPx, hPx;
    SDL_GL_GetDrawableSize(impl_->window, (int*)&wPx, (int*)&hPx);
    int wVpx, hVpx;
    SDL_GetWindowSize(impl_->window, &wVpx, &hVpx);
    return (float(wPx) / float(wVpx));
}

Point Window::GlobalToWindowCoord(int globalX, int globalY) {
    int wx, wy;
    SDL_GetWindowPosition(impl_->window, &wx, &wy);
    return Point(globalX - wx, globalY - wy);
}

bool Window::IsVisible() const {
    return (SDL_GetWindowFlags(impl_->window) & SDL_WINDOW_SHOWN);
}

void Window::Show(bool vis /*= true*/) {
    if (vis) {
        SDL_ShowWindow(impl_->window);
    } else {
        SDL_HideWindow(impl_->window);
    }
}

void Window::Close() { Application::GetInstance().RemoveWindow(this); }

void Window::SetNeedsLayout() { impl_->needsLayout = true; }

void Window::RaiseToTop() const { SDL_RaiseWindow(impl_->window); }

bool Window::IsActiveWindow() const {
    return (SDL_GetWindowFlags(impl_->window) & SDL_WINDOW_INPUT_FOCUS);
}

void Window::AddChild(std::shared_ptr<Widget> w) {
    impl_->children.push_back(w);
    impl_->needsLayout = true;
}

void Window::ShowDialog(std::shared_ptr<Dialog> dlg) {
    if (impl_->activeDialog) {
        CloseDialog();
    }
    impl_->activeDialog = dlg;
    dlg->OnWillShow();

    auto winSize = GetSize();
    auto pref = dlg->CalcPreferredSize(GetTheme());
    int w = dlg->GetFrame().width;
    int h = dlg->GetFrame().height;
    if (w == 0) {
        w = pref.width;
    }
    if (h == 0) {
        h = pref.height;
    }
    w = std::min(w, int(std::round(0.8 * winSize.width)));
    h = std::min(h, int(std::round(0.8 * winSize.height)));
    dlg->SetFrame(
            gui::Rect((winSize.width - w) / 2, (winSize.height - h) / 2, w, h));
    dlg->Layout(GetTheme());
}

void Window::CloseDialog() { impl_->activeDialog.reset(); }

void Window::ShowMessageBox(const char* title, const char* message) {
    auto em = GetTheme().fontSize;
    auto margins = Margins(GetTheme().defaultMargin);
    auto dlg = std::make_shared<Dialog>(title);
    auto layout = std::make_shared<Vert>(em, margins);
    layout->AddChild(std::make_shared<Label>(message));
    auto ok = std::make_shared<Button>("Ok");
    ok->SetOnClicked([this]() { this->CloseDialog(); });
    layout->AddChild(Horiz::MakeCentered(ok));
    dlg->AddChild(layout);
    ShowDialog(dlg);
}

double Window::GetLastFrameTimeSeconds() const { return impl_->lastFrameTime; }

void Window::Layout(const Theme& theme) {
    if (impl_->children.size() == 1) {
        auto r = GetContentRect();
        impl_->children[0]->SetFrame(r);
        impl_->children[0]->Layout(theme);
    } else {
        for (auto& child : impl_->children) {
            child->Layout(theme);
        }
    }
}

void Window::OnMenuItemSelected(Menu::ItemId itemId) {}

namespace {
enum Mode { NORMAL, DIALOG, NO_INPUT };

Widget::DrawResult DrawChild(DrawContext& dc,
                             const char* name,
                             std::shared_ptr<Widget> child,
                             Mode mode) {
    // Note: ImGUI's concept of a "window" is really a moveable child of the
    //       OS window. We want a child to act like a child of the OS window,
    //       like native UI toolkits, Qt, etc. So the top-level widgets of
    //       a window are drawn using ImGui windows whose frame is specified
    //       and which have no title bar, resizability, etc.

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse;
    // Q: When we want no input, why not use ImGui::BeginPopupModal(),
    //    which takes care of blocking input for us, since a modal popup
    //    is the most likely use case for wanting no input?
    // A: It animates an overlay, which would require us to constantly
    //    redraw, otherwise it only animates when the mouse moves. But
    //    we don't need constant animation for anything else, so that would
    //    be a waste of CPU and battery (and really annoys people like me).
    if (mode == NO_INPUT) {
        flags |= ImGuiWindowFlags_NoInputs;
    }
    auto frame = child->GetFrame();
    bool bgColorNotDefault = !child->IsDefaultBackgroundColor();
    auto isContainer = !child->GetChildren().empty();
    if (isContainer) {
        dc.uiOffsetX = frame.x;
        dc.uiOffsetY = frame.y;
        ImGui::SetNextWindowPos(ImVec2(frame.x, frame.y));
        ImGui::SetNextWindowSize(ImVec2(frame.width, frame.height));
        if (bgColorNotDefault) {
            auto& bgColor = child->GetBackgroundColor();
            ImGui::PushStyleColor(ImGuiCol_WindowBg,
                                  util::colorToImgui(bgColor));
        }
        ImGui::Begin(name, nullptr, flags);
    } else {
        dc.uiOffsetX = 0;
        dc.uiOffsetY = 0;
    }

    Widget::DrawResult result;
    result = child->Draw(dc);

    if (isContainer) {
        ImGui::End();
        if (bgColorNotDefault) {
            ImGui::PopStyleColor();
        }
    }

    return result;
}
}  // namespace

Widget::DrawResult Window::OnDraw(float dtSec) {
    // These are here to provide fast unique window names. If you find yourself
    // needing more than a handful, you should probably be using a container
    // of some sort (see Layout.h).
    static const char* winNames[] = {
            "win1",  "win2",  "win3",  "win4",  "win5",  "win6",  "win7",
            "win8",  "win9",  "win10", "win11", "win12", "win13", "win14",
            "win15", "win16", "win17", "win18", "win19", "win20"};

    bool needsLayout = false;
    bool needsRedraw = false;

    // Run the deferred callbacks that need to happen outside a draw
    while (!impl_->deferredUntilBeforeDraw.empty()) {
        impl_->deferredUntilBeforeDraw.front()();
        impl_->deferredUntilBeforeDraw.pop();
    }

    impl_->renderer->BeginFrame();  // this can return false if Filament wants
                                    // to skip a frame

    // Set current context
    MakeCurrent();  // make sure our ImGUI context is active
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = dtSec;

    // Set mouse information
    int mx, my;
    Uint32 buttons = SDL_GetGlobalMouseState(&mx, &my);
    auto mousePos = GlobalToWindowCoord(mx, my);
    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
    io.MouseDown[0] = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    io.MouseDown[1] = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    io.MouseDown[2] = (buttons & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
    if ((SDL_GetWindowFlags(impl_->window) & SDL_WINDOW_INPUT_FOCUS) != 0) {
        auto scaling = GetScaling();
        io.MousePos = ImVec2((float)mousePos.x * scaling,
                             (float)mousePos.y * scaling);
    }

    // Set key information
    io.KeyShift = ((SDL_GetModState() & KMOD_SHIFT) != 0);
    io.KeyAlt = ((SDL_GetModState() & KMOD_ALT) != 0);
    io.KeyCtrl = ((SDL_GetModState() & KMOD_CTRL) != 0);
    io.KeySuper = ((SDL_GetModState() & KMOD_GUI) != 0);

    // Begin ImGUI frame
    ImGui::NewFrame();
    ImGui::PushFont(impl_->imgui.systemFont);

    // Run the deferred callbacks that need to happen inside a draw
    // In particular, text sizing with ImGUI seems to require being
    // in a frame, otherwise there isn't an GL texture info and we crash.
    while (!impl_->deferredUntilDraw.empty()) {
        impl_->deferredUntilDraw.front()();
        impl_->deferredUntilDraw.pop();
    }

    // Layout if necessary.  This must happen within ImGui setup so that widgets
    // can query font information.
    auto& theme = impl_->theme;
    if (impl_->needsLayout) {
        Layout(theme);
        impl_->needsLayout = false;
    }

    auto size = GetSize();
    int em = theme.fontSize;  // em = font size in digital type (from Wikipedia)
    DrawContext dc{theme, 0, 0, size.width, size.height, em, dtSec};

    // Draw all the widgets. These will get recorded by ImGui.
    int winIdx = 0;
    Mode drawMode = (impl_->activeDialog ? NO_INPUT : NORMAL);
    for (auto& child : this->impl_->children) {
        if (!child->IsVisible()) {
            continue;
        }
        auto result = DrawChild(dc, winNames[winIdx++], child, drawMode);
        if (result != Widget::DrawResult::NONE) {
            needsRedraw = true;
        }
        if (result == Widget::DrawResult::RELAYOUT) {
            needsLayout = true;
        }
    }

    // Draw menubar after the children so it is always on top (although it
    // shouldn't matter, as there shouldn't be anything under it)
    auto menubar = Application::GetInstance().GetMenubar();
    if (menubar) {
        auto id = menubar->DrawMenuBar(dc, !impl_->activeDialog);
        if (id != Menu::NO_ITEM) {
            OnMenuItemSelected(id);
            needsRedraw = true;
        }
    }

    // Draw any active dialog
    if (impl_->activeDialog) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize,
                            theme.dialogBorderWidth);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,
                            theme.dialogBorderRadius);
        if (DrawChild(dc, "dialog", impl_->activeDialog, DIALOG) !=
            Widget::DrawResult::NONE) {
            needsRedraw = true;
        }
        ImGui::PopStyleVar(2);
    }

    // Finish frame and generate the commands
    ImGui::PopFont();
    ImGui::EndFrame();
    ImGui::Render();  // creates the draw data (i.e. Render()s to data)

    // Draw the ImGui commands
    impl_->imgui.imguiBridge->update(ImGui::GetDrawData());

    impl_->renderer->Draw();

    impl_->renderer->EndFrame();

    if (needsLayout) {
        return Widget::DrawResult::RELAYOUT;
    } else if (needsRedraw) {
        return Widget::DrawResult::REDRAW;
    } else {
        return Widget::DrawResult::NONE;
    }
}

Window::DrawResult Window::DrawOnce(float dtSec) {
    auto t0 = SDL_GetPerformanceCounter();

    bool neededLayout = impl_->needsLayout;

    auto result = OnDraw(dtSec);
    if (result == Widget::DrawResult::RELAYOUT) {
        impl_->needsLayout = true;
    }

    // ImGUI can take two frames to do its layout, so if we did a layout
    // redraw a second time. This helps prevent a brief red flash when the
    // window first appears, as well as corrupted images if the
    // window initially appears underneath the mouse.
    if (neededLayout || impl_->needsLayout) {
        OnDraw(0.001);
    }

    auto t1 = SDL_GetPerformanceCounter();
    double freq = double(SDL_GetPerformanceFrequency());
    impl_->lastFrameTime = double(t1 - t0) / freq;

    return (result == Widget::DrawResult::NONE ? NONE : REDRAW);
}

void Window::OnResize() {
    impl_->needsLayout = true;

#if __APPLE__
    // We need to recreate the swap chain after resizing a window on macOS
    // otherwise things look very wrong.
    impl_->renderer->UpdateSwapChain();
#endif  // __APPLE__

    impl_->imgui.imguiBridge->onWindowResized(*this);

    auto size = GetSize();
    auto scaling = GetScaling();

    auto oldContext = MakeCurrent();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(size.width, size.height);
    if (impl_->imgui.scaling != scaling) {
        updateImGuiForScaling(1.0 / impl_->imgui.scaling);  // undo previous
        updateImGuiForScaling(scaling);
        impl_->imgui.scaling = scaling;
    }
    io.DisplayFramebufferScale.x = 1.0f;
    io.DisplayFramebufferScale.y = 1.0f;

    if (impl_->wantsAutoSizeAndCenter) {
        impl_->wantsAutoSizeAndCenter = false;
        ImGui::NewFrame();
        ImGui::PushFont(impl_->imgui.systemFont);
        auto pref = CalcPreferredSize();
        Size size(pref.width / this->impl_->imgui.scaling,
                  pref.height / this->impl_->imgui.scaling);
        SDL_SetWindowSize(impl_->window, size.width, size.height);
        SDL_SetWindowPosition(impl_->window, SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED);
        ImGui::PopFont();
        ImGui::EndFrame();
        OnResize();
    }

    RestoreCurrent(oldContext);
}

void Window::OnMouseEvent(const MouseEvent& e) {
    MakeCurrent();
    switch (e.type) {
        case MouseEvent::MOVE:
        case MouseEvent::BUTTON_DOWN:
        case MouseEvent::DRAG:
        case MouseEvent::BUTTON_UP:
            break;
        case MouseEvent::WHEEL: {
            ImGuiIO& io = ImGui::GetIO();
            float dx = 0.0, dy = 0.0;
            if (e.wheel.dx != 0) {
                dx = e.wheel.dx / std::abs(e.wheel.dx);
            }
            if (e.wheel.dy != 0) {
                dy = e.wheel.dy / std::abs(e.wheel.dy);
            }
            // Note: ImGUI's documentation says that 1 unit of wheel movement
            //       is about 5 lines of text scrolling.
            if (e.wheel.isTrackpad) {
                io.MouseWheelH += dx * 0.25;
                io.MouseWheel += dy * 0.25;
            } else {
                io.MouseWheelH += dx;
                io.MouseWheel += dy;
            }
            break;
        }
    }

    // Some ImGUI widgets have popup windows, in particular, the color
    // picker, which creates a popup window when you click on the color
    // patch. Since these aren't gui::Widgets, we don't know about them,
    // and will deliver mouse events to something below them. So find any
    // that would use the mouse, and if it isn't a toplevel child, then
    // eat the event for it.
    if (e.type == MouseEvent::BUTTON_DOWN || e.type == MouseEvent::BUTTON_UP) {
        ImGuiContext* context = ImGui::GetCurrentContext();
        for (auto* w : context->Windows) {
            if (!w->Hidden && w->Flags & ImGuiWindowFlags_Popup) {
                Rect r(w->Pos.x, w->Pos.y, w->Size.x, w->Size.y);
                if (r.Contains(e.x, e.y)) {
                    bool weKnowThis = false;
                    for (auto child : impl_->children) {
                        if (child->GetFrame() == r) {
                            weKnowThis = true;
                            break;
                        }
                    }
                    if (!weKnowThis) {
                        // This is not a rect that is one of our children,
                        // must be an ImGUI internal popup. Eat event.
                        return;
                    }
                }
            }
        }
    }

    // Iterate backwards so that we send mouse events from the top down.
    auto handleMouseForChild = [this](const MouseEvent& e,
                                      std::shared_ptr<Widget> child) -> bool {
        if (child->GetFrame().Contains(e.x, e.y) && child->IsVisible()) {
            if (e.type == MouseEvent::BUTTON_DOWN) {
                impl_->focusWidget = child.get();
            }
            child->Mouse(e);
            return true;
        }
        return false;
    };
    if (impl_->activeDialog) {
        handleMouseForChild(e, impl_->activeDialog);
    } else {
        std::vector<std::shared_ptr<Widget>>& children = impl_->children;
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            if (handleMouseForChild(e, *it)) {
                break;
            }
        }
    }
}

void Window::OnKeyEvent(const KeyEvent& e) {
    ImGuiIO& io = ImGui::GetIO();
    if (e.key < IM_ARRAYSIZE(io.KeysDown)) {
        io.KeysDown[e.key] = (e.type == KeyEvent::DOWN);
    }

    // If an ImGUI widget is not getting keystrokes, we can send them to
    // non-ImGUI widgets
    if (ImGui::GetCurrentContext()->ActiveId == 0 && impl_->focusWidget) {
        impl_->focusWidget->Key(e);
    }
}

void Window::OnTextInput(const TextInputEvent& e) {
    MakeCurrent();
    ImGuiIO& io = ImGui::GetIO();
    io.AddInputCharactersUTF8(e.utf8);
}

void Window::OnDragDropped(const char* path) {}

}  // namespace gui
}  // namespace open3d
