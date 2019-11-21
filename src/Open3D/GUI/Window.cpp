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

#include "Native.h"
#include "Renderer.h"
#include "Widget.h"

#include <SDL.h>

#include <vector>

// ----------------------------------------------------------------------------
namespace open3d {
namespace gui {

struct Window::Impl
{
    SDL_Window *window = nullptr;
    Renderer *renderer;
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
    impl_->renderer = new Renderer(*this);
}

Window::~Window() {
    impl_->children.clear();  // needs to happen before deleting renderer
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

void Window::Layout() {
}

void Window::OnDraw() {
    if (impl_->needsLayout) {
        Layout();
        impl_->needsLayout = false;
    }

    DrawContext dc;
    if (impl_->renderer->BeginFrame()) {
        for (auto &child : impl_->children) {
            child->Draw(dc);
        }

        impl_->renderer->EndFrame();
    } else {
        ++impl_->nSkippedFrames;
    }
}

void Window::OnResize() {
    impl_->needsLayout = true;
}

} // gui
} // opend3d
