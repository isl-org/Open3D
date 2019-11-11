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

#include "Renderer.h"

#include <SDL.h>

namespace {
// Makes sure that SDL_Quit gets called after the last window is destroyed
class SDLLibrary {
public:
    static SDLLibrary& instance() {
        static SDLLibrary *lib = nullptr;
        if (!lib) {
            lib = new SDLLibrary();
        }
        return *lib;
    }

    void init() {
        if (count_ == 0) {
            SDL_Init(SDL_INIT_EVENTS);
        }
        count_ += 1;
    }

    void quit() {
        count_ -= 1;
        if (count_ == 0) {
            SDL_Quit();
        }
    }

private:
    int count_ = 0;
};
}

// ----------------------------------------------------------------------------
namespace open3d {
namespace gui {

struct Window::Impl
{
    SDL_Window *window = nullptr;
    Renderer *renderer;
};

Window::Window(const std::string& title, int width, int height)
    : impl_(new Window::Impl()) {
    SDLLibrary::instance().init();

    const int x = SDL_WINDOWPOS_CENTERED;
    const int y = SDL_WINDOWPOS_CENTERED;
    uint32_t flags = SDL_WINDOW_SHOWN |  // so SDL's context gets created
                     SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    impl_->window = SDL_CreateWindow(title.c_str(), x, y, width, height, flags);

    // On single-threaded platforms, Filament's OpenGL context must be current,
    // not SDL's context, so create the renderer after the window.
    impl_->renderer = new Renderer();
}

Window::~Window() {
    delete impl_->renderer;
    SDL_DestroyWindow(impl_->window);

    SDLLibrary::instance().quit();
}

Renderer& Window::renderer() {
    return *impl_->renderer;
}

void Window::show(bool vis /*= true*/) {
    if (vis) {
        SDL_ShowWindow(impl_->window);
    } else {
        SDL_HideWindow(impl_->window);
    }
}

} // gui
} // opend3d
