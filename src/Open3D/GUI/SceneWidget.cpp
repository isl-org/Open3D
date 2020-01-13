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

#include "Color.h"
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

SceneWidget::SceneWidget(visualization::Scene& scene) : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0, 0, 1, 1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->cameraManipulator =
            std::make_unique<visualization::CameraManipulator>(
                    *view->GetCamera(), 1, 1);
}

SceneWidget::~SceneWidget() { impl_->scene.RemoveView(impl_->viewId); }

void SceneWidget::SetFrame(const Rect& f) {
    Super::SetFrame(f);

    impl_->frameChanged = true;
}

void SceneWidget::SetBackgroundColor(const Color& color) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetClearColor({color.GetRed(), color.GetGreen(), color.GetBlue()});
}

void SceneWidget::SetDiscardBuffers(
        const visualization::View::TargetBuffers& buffers) {
    auto view = impl_->scene.GetView(impl_->viewId);
    view->SetDiscardBuffers(buffers);
}

visualization::Scene* SceneWidget::GetScene() const { return &impl_->scene; }

visualization::CameraManipulator* SceneWidget::GetCameraManipulator() const {
    return impl_->cameraManipulator.get();
}

void SceneWidget::SetSelectedGeometry(const visualization::GeometryHandle& geometry, const bool switchCamera) {
    if (switchCamera) {
        auto boundingSphere = impl_->scene.GetEntityBoundingSphere(geometry);
        SetCameraPOI(boundingSphere.first);

        // TODO: Rotate camera to entity?
    }
}

void SceneWidget::SetCameraPOI(const Eigen::Vector3f& location) {
    cameraControlsState_.poi = location;

    auto cameraman = GetCameraManipulator();
    cameraControlsState_.orbitHeight = (cameraman->GetPosition() - location).norm();
}

Widget::DrawResult SceneWidget::Draw(const DrawContext& context, const float frameDelta) {
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

    if (cameraControlsState_.orbiting) {
        // Maybe poi isn't initialized
        if (cameraControlsState_.orbitHeight <= 0.1f) {
            SetCameraPOI({0.f, 0.f, 0.f});
        }

        auto cameraman = GetCameraManipulator();

        float orbit = cameraControlsState_.orbitHeight +
                      cameraControlsState_.frameWheelDelta;
        if (orbit < 0.1f) {
            orbit = 0.1f;
        }
        cameraControlsState_.orbitHeight = orbit;

        cameraman->Orbit(cameraControlsState_.poi,
                         cameraControlsState_.orbitHeight,
                         -cameraControlsState_.frameDx * frameDelta,
                         -cameraControlsState_.frameDy * frameDelta,
                         cameraControlsState_.rotationSpeed);
    }

    cameraControlsState_.Reset();

    return Widget::DrawResult::NONE;
}

void SceneWidget::Mouse(const MouseEvent& e) {
    switch (e.type) {
        case MouseEvent::BUTTON_DOWN:
            if (e.button.button == MouseButton::LEFT) {
                cameraControlsState_.orbiting = true;
                cameraControlsState_.lastMouseX = e.x;
                cameraControlsState_.lastMouseY = e.y;
            }
            break;
        case MouseEvent::DRAG:
            if (cameraControlsState_.orbiting) {
                float fX = e.x;
                float fY = e.y;

                cameraControlsState_.frameDx +=
                        fX - cameraControlsState_.lastMouseX;
                cameraControlsState_.frameDy +=
                        fY - cameraControlsState_.lastMouseY;

                cameraControlsState_.lastMouseX = fX;
                cameraControlsState_.lastMouseY = fY;
            }
            break;
        case MouseEvent::WHEEL:
            if (cameraControlsState_.orbiting) {
                cameraControlsState_.frameWheelDelta += e.wheel.dy;
            }
            break;
        case MouseEvent::BUTTON_UP:
            if (e.button.button == MouseButton::LEFT) {
                cameraControlsState_.orbiting = false;
            }
            break;
        default:
            break;
    }
}

void SceneWidget::Key(const KeyEvent& e) {}

void SceneWidget::CameraControlsState::Reset() {
    frameDx = 0.f;
    frameDy = 0.f;
    frameWheelDelta = 0.f;
}

} // gui
} // open3d
