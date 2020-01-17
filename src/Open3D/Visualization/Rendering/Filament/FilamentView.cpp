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
#include "FilamentScene.h"

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/View.h>

namespace open3d {
namespace visualization {

namespace {
const filament::LinearColorA kDepthClearColor = {0,0,0,0};
}

FilamentView::FilamentView(filament::Engine& aEngine, FilamentScene& aScene)
    : engine_(aEngine), scene_(aScene) {
    view_ = engine_.createView();
    view_->setScene(scene_.GetNativeScene());
    view_->setSampleCount(8);
    view_->setAntiAliasing(filament::View::AntiAliasing::FXAA);
    view_->setPostProcessingEnabled(true);

    camera_ = std::make_unique<FilamentCamera>(engine_);
    view_->setCamera(camera_->GetNativeCamera());

    camera_->SetProjection(90, 4.f / 3.f, 0.01, 1000,
                           Camera::FovType::Horizontal);
}

FilamentView::~FilamentView() {
    view_->setCamera(nullptr);
    view_->setScene(nullptr);

    camera_.reset();
    engine_.destroy(view_);
}

void FilamentView::SetMode(Mode mode) {
    switch (mode) {
        case Mode::Color:
            view_->setClearColor({clearColor_.x(), clearColor_.y(), clearColor_.z(), 1.f});
            break;
        case Mode::Depth:
            view_->setClearColor(kDepthClearColor);
            break;
        case Mode::Normal: break;
    }

    mode_ = mode;
}

void FilamentView::SetDiscardBuffers(const TargetBuffers& buffers)
{
    using namespace std;

    auto rawBuffers = static_cast<uint8_t>(buffers);
    uint8_t rawFilamentBuffers = 0;
    if (rawBuffers | (uint8_t)TargetBuffers::Color) {
        rawFilamentBuffers |= (uint8_t)filament::View::TargetBufferFlags::COLOR;
    }
    if (rawBuffers | (uint8_t)TargetBuffers::Depth) {
        rawFilamentBuffers |= (uint8_t)filament::View::TargetBufferFlags::DEPTH;
    }
    if (rawBuffers | (uint8_t)TargetBuffers::Stencil) {
        rawFilamentBuffers |= (uint8_t)filament::View::TargetBufferFlags::STENCIL;
    }

    view_->setRenderTarget(nullptr, static_cast<filament::View::TargetBufferFlags>(rawFilamentBuffers));
}

void FilamentView::SetViewport(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) {
    view_->setViewport({x, y, w, h});
}

void FilamentView::SetClearColor(const Eigen::Vector3f& color) {
    clearColor_ = color;

    // We apply changes immediately only in color mode
    // In other cases color will be setted on mode switch
    if (mode_ == Mode::Color) {
        view_->setClearColor({color.x(), color.y(), color.z(), 1.f});
    }
}

Camera* FilamentView::GetCamera() const { return camera_.get(); }

void FilamentView::PreRender() {

}

void FilamentView::PostRender() {

}

}  // namespace visualization
}  // namespace open3d