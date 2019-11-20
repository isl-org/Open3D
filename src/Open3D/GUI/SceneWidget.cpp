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

#include "SceneWidget.h"

namespace open3d {
namespace gui {

struct SceneWidget::Impl {
    RendererView view;

    Impl(Renderer& r, Renderer::ViewId viewId)
        : view(r, viewId)
    {}
};

SceneWidget::SceneWidget(Renderer& r)
    : impl_(new SceneWidget::Impl(r, r.CreateView()))
{
}

SceneWidget::~SceneWidget() {
}

void SceneWidget::SetFrame(const Rect& f) {
    Super::SetFrame(f);
    impl_->view.SetViewport(f);
}

void SceneWidget::SetBackgroundColor(const Color& color) {
    impl_->view.SetClearColor(color);
}

RendererCamera& SceneWidget::GetCamera() {
    return impl_->view.GetCamera();
}

void SceneWidget::AddLight(Renderer::LightId lightId) {
    impl_->view.GetScene().AddLight(lightId);
}

void SceneWidget::RemoveLight(Renderer::LightId lightId) {
    impl_->view.GetScene().RemoveLight(lightId);
}

void SceneWidget::AddMesh(Renderer::MeshId meshId,
                          float x /*=0*/, float y /*=0*/, float z /*=0*/) {
    Transform t;
    t.Translate(x, y, z);
    impl_->view.GetScene().AddMesh(meshId, t);
}

void SceneWidget::RemoveMesh(Renderer::MeshId meshId) {
    impl_->view.GetScene().RemoveMesh(meshId);
}

Widget::DrawResult SceneWidget::Draw(const DrawContext& context) {
    impl_->view.Draw();
    return Widget::DrawResult::NONE;
}

} // gui
} // open3d
