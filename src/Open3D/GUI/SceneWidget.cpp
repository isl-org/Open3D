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

#include "Events.h"

#include "Open3D/Visualization/Rendering/CameraManipulator.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

namespace open3d {
namespace gui {

struct SceneWidget::Impl {
    visualization::Scene& scene;
    visualization::ViewHandle viewId;
    std::unique_ptr<visualization::CameraManipulator> cameraManipulator;
    bool frameChanged = false;

    explicit Impl(visualization::Scene& aScene) : scene(aScene) {}
};

SceneWidget::SceneWidget(visualization::Scene& scene)
    : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0,0,1,1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->cameraManipulator = std::make_unique<visualization::CameraManipulator>(*view->GetCamera(), 1, 1);
}

SceneWidget::~SceneWidget() {
    impl_->scene.RemoveView(impl_->viewId);
}

void SceneWidget::SetFrame(const Rect& f) {
    Super::SetFrame(f);

    impl_->frameChanged = true;
}

void SceneWidget::SetBackgroundColor(const Color& color) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetClearColor({color.GetRed(), color.GetGreen(), color.GetBlue()});
}

void SceneWidget::SetDiscardBuffers(const visualization::View::TargetBuffers& buffers) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetDiscardBuffers(buffers);
}

visualization::Scene* SceneWidget::GetScene() const {
    return &impl_->scene;
}

visualization::CameraManipulator* SceneWidget::GetCameraManipulator() const {
    return impl_->cameraManipulator.get();
}

Widget::DrawResult SceneWidget::Draw(const DrawContext& context) {
    if (impl_->frameChanged) {
        impl_->frameChanged = false;

        auto f = GetFrame();

        // GUI have null of Y axis at top, but renderer have it at bottom
        // so we need to convert coordinates
        int y = context.screenHeight - (f.height + f.y);

        auto view = impl_->scene.GetView(impl_->viewId);
        view->SetViewport(f.x, y, f.width, f.height);

        impl_->cameraManipulator->SetViewport(f.width, f.height);
    }

    return Widget::DrawResult::NONE;
}

void SceneWidget::Mouse(const MouseEvent& e) {
    switch (e.type) {
        case MouseEvent::BUTTON_DOWN:
            break;
        case MouseEvent::DRAG:
            break;
        case MouseEvent::BUTTON_UP:
            break;
        default:
            break;
    }
}

} // gui
} // open3d
