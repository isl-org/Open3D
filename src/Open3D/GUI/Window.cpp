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
#include "Native.h"
#include "Renderer.h"
#include "Theme.h"
#include "Widget.h"

#include <imgui.h>
#include <SDL.h>

#include <vector>

// ----------------------------------------------------------------------------
namespace open3d {
namespace gui {

namespace {
ImVec4 colorToImgui(const Color& color) {
    return ImVec4(color.GetRed(), color.GetGreen(), color.GetBlue(),
                  color.GetAlpha());
}

// Assumes the correct ImGuiContext is current
void updateImGuiForScaling(float newScaling) {
    ImGuiStyle &style = ImGui::GetStyle();
    // FrameBorderSize is not adjusted (we want minimal borders)
    style.FrameRounding *= newScaling;
}

} // (anonymous)

struct Window::Impl
{
    SDL_Window *window = nullptr;
    Theme theme;  // so that the font size can be different based on scaling
    Renderer *renderer;
    struct {
        ImGuiContext *context;
        ImFont *systemFont;  // is a reference; owned by imguiContext
        float scaling = 1.0;
    } imgui;
    std::vector<std::shared_ptr<Widget>> children;
    bool needsLayout = true;
    int nSkippedFrames = 0;
};

Window::Window(const std::string& title, int width, int height)
: impl_(new Window::Impl()) {
    const int x = SDL_WINDOWPOS_CENTERED;
    const int y = SDL_WINDOWPOS_CENTERED;
    uint32_t flags = SDL_WINDOW_SHOWN |  // so SDL's context gets created
                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    impl_->window = SDL_CreateWindow(title.c_str(), x, y, width, height, flags);

    // On single-threaded platforms, Filament's OpenGL context must be current,
    // not SDL's context, so create the renderer after the window.

    // ImGUI creates a bitmap atlas from a font, so we need to have the correct
    // size when we create it, because we can't change the bitmap without
    // reloading the whole thing (expensive).
    impl_->theme = Application::GetInstance().GetTheme();
    impl_->theme.fontSize *= GetScaling();

    impl_->renderer = new Renderer(*this, impl_->theme);

    auto &theme = impl_->theme;  // shorter alias
    impl_->imgui.context = ImGui::CreateContext();
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(0, 0);
    style.WindowRounding = 0;
    style.Colors[ImGuiCol_WindowBg] = colorToImgui(theme.backgroundColor);
    style.Colors[ImGuiCol_Text] = colorToImgui(theme.textColor);
    style.FrameBorderSize = theme.borderWidth;
    style.FrameRounding = theme.borderRadius;
    style.Colors[ImGuiCol_Border] = colorToImgui(theme.borderColor);

    // If the given font path is invalid, ImGui will silently fall back to proggy, which is a
    // tiny "pixel art" texture that is compiled into the library.
    if (!theme.fontPath.empty()) {
        ImGuiIO &io = ImGui::GetIO();
        impl_->imgui.systemFont = io.Fonts->AddFontFromFileTTF(theme.fontPath.c_str(), theme.fontSize);
        /*static*/ unsigned char* pixels;
        int width, height, bytesPerPx;
        io.Fonts->GetTexDataAsAlpha8(&pixels, &width, &height, &bytesPerPx);
        impl_->renderer->AddFontTextureAtlasAlpha8(pixels, width, height, bytesPerPx);
    }
}

Window::~Window() {
    impl_->children.clear();  // needs to happen before deleting renderer
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGui::DestroyContext();
    delete impl_->renderer;
    SDL_DestroyWindow(impl_->window);
}

void* Window::GetNativeDrawable() const {
    return open3d::gui::GetNativeDrawable(impl_->window);
}

uint32_t Window::GetID() const {
    return SDL_GetWindowID(impl_->window);
}

Renderer& Window::GetRenderer() {
    return *impl_->renderer;
}

Size Window::GetSize() const {
    uint32_t w, h;
    SDL_GL_GetDrawableSize(impl_->window, (int*) &w, (int*) &h);
    return Size(w, h);
}

float Window::GetScaling() const {
    uint32_t wPx, hPx;
    SDL_GL_GetDrawableSize(impl_->window, (int*)&wPx, (int*)&hPx);
    int wVpx, hVpx;
    SDL_GetWindowSize(impl_->window, &wVpx, &hVpx);
    return (float(wPx) / float(wVpx));
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

void Window::AddChild(std::shared_ptr<Widget> w) {
    impl_->children.push_back(w);
    impl_->needsLayout = true;
}

void Window::Layout(const Theme& theme) {
    for (auto &child : impl_->children) {
        child->Layout(theme);
    }
}

void Window::OnDraw(float dtSec) {
    // These are here to provide fast unique window names. If you find yourself
    // needing more than a handful, you should probably be using a container
    // of some sort (see Layout.h).
    static const char* winNames[] = { "win1", "win2", "win3", "win4", "win5",
                                      "win6", "win7", "win8", "win9", "win10",
                                      "win11", "win12", "win13", "win14", "win15",
                                      "win16", "win17", "win18", "win19", "win20" };

    impl_->renderer->BeginFrame();  // this can return false if Filament wants to skip a frame

    // Set current context
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = dtSec;

    // Set mouse information
    int mx, my;
    Uint32 buttons = SDL_GetMouseState(&mx, &my);
    io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
    io.MouseDown[0] = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    io.MouseDown[1] = (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    io.MouseDown[2] = (buttons & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
    // TODO: use SDL_CaptureMouse() to retrieve mouse coordinates
    // outside of the client area; see the imgui SDL example.
    if ((SDL_GetWindowFlags(impl_->window) & SDL_WINDOW_INPUT_FOCUS) != 0) {
        auto scaling = GetScaling();
        io.MousePos = ImVec2((float)mx * scaling, (float)my * scaling);
    }

    // Set key information
    io.KeyShift = ((SDL_GetModState() & KMOD_SHIFT) != 0);
    io.KeyAlt = ((SDL_GetModState() & KMOD_ALT) != 0);
    io.KeyCtrl = ((SDL_GetModState() & KMOD_CTRL) != 0);
    io.KeySuper = ((SDL_GetModState() & KMOD_GUI) != 0);

    // Begin ImGUI frame
    ImGui::NewFrame();
    ImGui::PushFont(impl_->imgui.systemFont);

    // Layout if necessary.  This must happen within ImGui setup so that widgets
    // can query font information.
    auto &theme = this->impl_->theme;
    if (this->impl_->needsLayout) {
        this->Layout(theme);
        this->impl_->needsLayout = false;
    }

    DrawContext dc{ theme };

    // Draw the 3D widgets first (in case the UI wants to be transparent
    // on top). These will actually get drawn now. Since these are not
    // ImGui objects, nothing will happen as far as ImGui is concerned,
    // but Filament will issue the appropriate rendering commands.
    for (auto &child : this->impl_->children) {
        if (child->Is3D()) {
            child->Draw(dc);
        }
    }

    // Now draw all the 2D widgets. These will get recorded by ImGui.
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse;
    int winIdx = 0;
    for (auto &child : this->impl_->children) {
        if (!child->Is3D()) {
            auto frame = child->GetFrame();
            auto isContainer = !child->GetChildren().empty();
            if (isContainer) {
                ImGui::SetNextWindowPos(ImVec2(frame.x, frame.y));
                ImGui::SetNextWindowSize(ImVec2(frame.width, frame.height));
                ImGui::Begin(winNames[winIdx++], nullptr, flags);
            }
            child->Draw(dc);
            if (isContainer) {
                ImGui::End();
            }
        }
    }

    // Finish frame and generate the commands
    ImGui::PopFont();
    ImGui::EndFrame();
    ImGui::Render(); // creates the draw data (i.e. Render()s to data)

    // Draw the ImGui commands
    impl_->renderer->RenderImgui(ImGui::GetDrawData());

    impl_->renderer->EndFrame();
}

void Window::OnResize() {
    impl_->needsLayout = true;

    auto size = GetSize();
    auto scaling = GetScaling();

    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(size.width, size.height);
    if (impl_->imgui.scaling != scaling) {
        updateImGuiForScaling(1.0 / impl_->imgui.scaling);  // undo previous
        updateImGuiForScaling(scaling);
        impl_->imgui.scaling = scaling;
    }
    io.DisplayFramebufferScale.x = 1.0f;
    io.DisplayFramebufferScale.y = 1.0f;
}

void Window::OnMouseMove(const MouseMoveEvent& e) {
    ImGui::SetCurrentContext(impl_->imgui.context);
}

void Window::OnMouseButton(const MouseButtonEvent& e) {
    ImGui::SetCurrentContext(impl_->imgui.context);
}

void Window::OnMouseWheel(const MouseWheelEvent& e) {
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (e.x > 0 ? 1 : -1);
    io.MouseWheel  += (e.y > 0 ? 1 : -1);
}

void Window::OnTextInput(const TextInputEvent& e) {
    ImGui::SetCurrentContext(impl_->imgui.context);
    ImGuiIO& io = ImGui::GetIO();
    io.AddInputCharactersUTF8(e.utf8);
}

} // gui
} // opend3d
