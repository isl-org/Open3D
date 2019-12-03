// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "FilamentView.h"

#include "FilamentCamera.h"

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/View.h>

namespace open3d
{
namespace visualization
{

FilamentView::FilamentView(filament::Engine& aEngine, filament::Scene& aScene)
    : engine(aEngine)
    , scene(aScene)
{
    view = engine.createView();
    view->setScene(&scene);

    camera = std::make_unique<FilamentCamera>(engine);
    view->setCamera(camera->GetNativeCamera());

    view->setClearColor(filament::LinearColorA{ 0.0f, 0.0f, 0.0f, 1.0f });
    camera->SetProjection(0.01, 50, 90.0);
}

FilamentView::~FilamentView()
{
    view->setCamera(nullptr);
    view->setScene(nullptr);

    camera.reset();
    engine.destroy(view);
}

void FilamentView::SetClearColor(const filament::LinearColorA& color)
{
    view->setClearColor(color);
}

void FilamentView::SetViewport(const filament::Viewport& viewport)
{
    view->setViewport(viewport);
    camera->ChangeAspectRatio(float(viewport.width) / float(viewport.height));
}

}
}