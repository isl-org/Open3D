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

#include "Renderer.h"

#include <filament/Engine.h>

using namespace filament;

namespace open3d {
namespace gui {

struct Renderer::Impl
{
    filament::Engine::Backend driver;
    filament::Engine *engine;
};

Renderer::Renderer()
    : impl_(new Renderer::Impl())
{
    impl_->driver = filament::Engine::Backend::OPENGL;

    // On single-threaded platforms, Filament's OpenGL context must be current,
    // not SDL's context.
    impl_->engine = Engine::create(impl_->driver);

//    void* nativeWindow = ::getNativeWindow(mWindow);
//    void* nativeSwapChain = nativeWindow;
/*
#if defined(__APPLE__)
    void* metalLayer = nullptr;
    if (config.backend == filament::Engine::Backend::METAL) {
        metalLayer = setUpMetalLayer(nativeWindow);
        // The swap chain on Metal is a CAMetalLayer.
        nativeSwapChain = metalLayer;
    }
#if defined(FILAMENT_DRIVER_SUPPORTS_VULKAN)
    if (config.backend == filament::Engine::Backend::VULKAN) {
        // We request a Metal layer for rendering via MoltenVK.
        setUpMetalLayer(nativeWindow);
    }
#endif // FILAMENT_DRIVER_SUPPORTS_VULKAN
#endif // __APPLE__
*/
}

Renderer::~Renderer()
{
    Engine::destroy(&impl_->engine);
}

} // gui
} // open3d
