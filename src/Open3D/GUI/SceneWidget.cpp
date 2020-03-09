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

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

#include "Open3D/Visualization/Rendering/CameraInteractor.h"
#include "Open3D/Visualization/Rendering/LightDirectionInteractor.h"

#include <Eigen/Geometry>

#define ENABLE_PAN 1

namespace open3d {
namespace gui {

static const double NEAR_PLANE = 0.1;
static const double MIN_FAR_PLANE = 20.0;

// ----------------------------------------------------------------------------
class MouseInteractor {
public:
    MouseInteractor(visualization::Scene* scene, visualization::Camera* camera)
        : cameraControls_(std::make_unique<visualization::CameraInteractor>(
                  camera, MIN_FAR_PLANE)),
          lightDir_(std::make_unique<visualization::LightDirectionInteractor>(
                  scene, camera)) {}

    void SetViewSize(const Size& size) {
        cameraControls_->SetViewSize(size.width, size.height);
        lightDir_->SetViewSize(size.width, size.height);
    }

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        cameraControls_->SetBoundingBox(bounds);
        lightDir_->SetBoundingBox(bounds);
    }

    void SetDirectionalLight(
            visualization::LightHandle dirLight,
            std::function<void(const Eigen::Vector3f&)> onChanged) {
        lightDir_->SetDirectionalLight(dirLight);
        onLightDirChanged_ = onChanged;
    }

    void GoToCameraPreset(SceneWidget::CameraPreset preset) {
        switch (preset) {
            case SceneWidget::CameraPreset::PLUS_X:
                cameraControls_->GoToPreset(
                        visualization::CameraInteractor::CameraPreset::PLUS_X);
                break;
            case SceneWidget::CameraPreset::PLUS_Y:
                cameraControls_->GoToPreset(
                        visualization::CameraInteractor::CameraPreset::PLUS_Y);
                break;
            case SceneWidget::CameraPreset::PLUS_Z:
                cameraControls_->GoToPreset(
                        visualization::CameraInteractor::CameraPreset::PLUS_Z);
                break;
        }
    }

    void Mouse(const MouseEvent& e) {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                if (e.button.button == MouseButton::LEFT) {
                    if (e.modifiers & int(KeyModifier::SHIFT)) {
                        state_ = State::DOLLY;
#if ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::CTRL)) {
                        state_ = State::PAN;
#endif  // ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::META)) {
                        state_ = State::ROTATE_Z;
                    } else if (e.modifiers & int(KeyModifier::ALT)) {
                        state_ = State::ROTATE_LIGHT;
                    } else {
                        state_ = State::ROTATE_XY;
                    }
#if ENABLE_PAN
                } else if (e.button.button == MouseButton::RIGHT) {
                    state_ = State::PAN;
#endif  // ENABLE_PAN
                } else if (e.button.button == MouseButton::MIDDLE) {
                    state_ = State::ROTATE_LIGHT;
                }
                if (state_ == State::ROTATE_LIGHT) {
                    lightDir_->StartMouseDrag();
                } else if (state_ != State::NONE) {
                    cameraControls_->StartMouseDrag();
                }
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                switch (state_) {
                    case State::NONE:
                        break;
                    case State::PAN:
                        cameraControls_->Pan(dx, dy);
                        break;
                    case State::DOLLY:
                        cameraControls_->Dolly(dy,
                                               visualization::MatrixInteractor::
                                                       DragType::MOUSE);
                        break;
                    case State::ZOOM:
                        cameraControls_->Zoom(dy,
                                              visualization::MatrixInteractor::
                                                      DragType::MOUSE);
                        break;
                    case State::ROTATE_XY:
                        cameraControls_->Rotate(dx, dy);
                        break;
                    case State::ROTATE_Z:
                        cameraControls_->RotateZ(dx, dy);
                        break;
                    case State::ROTATE_LIGHT:
                        lightDir_->Rotate(dx, dy);
                        if (onLightDirChanged_) {
                            onLightDirChanged_(
                                    lightDir_->GetCurrentDirection());
                        }
                        break;
                }
                break;
            }
            case MouseEvent::WHEEL: {
                if (e.modifiers & int(KeyModifier::SHIFT)) {
                    cameraControls_->Zoom(
                            e.wheel.dy,
                            e.wheel.isTrackpad
                                    ? visualization::MatrixInteractor::
                                              DragType::TWO_FINGER
                                    : visualization::MatrixInteractor::
                                              DragType::WHEEL);
                } else {
                    cameraControls_->Dolly(
                            e.wheel.dy,
                            e.wheel.isTrackpad
                                    ? visualization::MatrixInteractor::
                                              DragType::TWO_FINGER
                                    : visualization::MatrixInteractor::
                                              DragType::WHEEL);
                }
                break;
            }
            case MouseEvent::BUTTON_UP:
                cameraControls_->EndMouseDrag();
                lightDir_->EndMouseDrag();
                state_ = State::NONE;
                break;
            default:
                break;
        }
    }

private:
    std::unique_ptr<visualization::CameraInteractor> cameraControls_;
    std::unique_ptr<visualization::LightDirectionInteractor> lightDir_;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged_;

    int mouseDownX_ = 0;
    int mouseDownY_ = 0;

    enum class State {
        NONE,
        PAN,
        DOLLY,
        ZOOM,
        ROTATE_XY,
        ROTATE_Z,
        ROTATE_LIGHT
    };
    State state_ = State::NONE;
};

// ----------------------------------------------------------------------------
struct SceneWidget::Impl {
    visualization::Scene& scene;
    visualization::ViewHandle viewId;
    std::shared_ptr<MouseInteractor> controls;
    bool frameChanged = false;
    visualization::LightHandle dirLight;
    std::function<void(const Eigen::Vector3f&)> onLightDirChanged;

    explicit Impl(visualization::Scene& aScene) : scene(aScene) {}
};

SceneWidget::SceneWidget(visualization::Scene& scene) : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0, 0, 1, 1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->controls =
            std::make_shared<MouseInteractor>(&scene, view->GetCamera());
}

SceneWidget::~SceneWidget() { impl_->scene.RemoveView(impl_->viewId); }

void SceneWidget::SetFrame(const Rect& f) {
    Super::SetFrame(f);

    impl_->controls->SetViewSize(Size(f.width, f.height));

    // We need to update the viewport and camera, but we can't do it here
    // because we need to know the window height to convert the frame
    // to OpenGL coordinates. We will actually do the updating in Draw().
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

void SceneWidget::SetupCamera(
        float verticalFoV,
        const geometry::AxisAlignedBoundingBox& geometryBounds,
        const Eigen::Vector3f& centerOfRotation) {
    impl_->controls->SetBoundingBox(geometryBounds);

    auto f = GetFrame();
    float aspect = 1.0f;
    if (f.height > 0) {
        aspect = float(f.width) / float(f.height);
    }
    auto far = std::max(MIN_FAR_PLANE, 2.0 * geometryBounds.GetExtent().norm());
    GetCamera()->SetProjection(verticalFoV, aspect, NEAR_PLANE, far,
                               visualization::Camera::FovType::Vertical);

    GoToCameraPreset(CameraPreset::PLUS_Z);  // default OpenGL view
}

void SceneWidget::SelectDirectionalLight(
        visualization::LightHandle dirLight,
        std::function<void(const Eigen::Vector3f&)> onDirChanged) {
    impl_->dirLight = dirLight;
    impl_->onLightDirChanged = onDirChanged;
    impl_->controls->SetDirectionalLight(
            dirLight, [this, dirLight](const Eigen::Vector3f& dir) {
                impl_->scene.SetLightDirection(dirLight, dir);
                if (impl_->onLightDirChanged) {
                    impl_->onLightDirChanged(dir);
                }
            });
}

void SceneWidget::GoToCameraPreset(CameraPreset preset) {
    impl_->controls->GoToCameraPreset(preset);
}

visualization::View* SceneWidget::GetView() const {
    return impl_->scene.GetView(impl_->viewId);
}

visualization::Scene* SceneWidget::GetScene() const { return &impl_->scene; }

visualization::Camera* SceneWidget::GetCamera() const {
    auto view = impl_->scene.GetView(impl_->viewId);
    return view->GetCamera();
}

Widget::DrawResult SceneWidget::Draw(const DrawContext& context) {
    // If the widget has changed size we need to update the viewport and the
    // camera. We can't do it in SetFrame() because we need to know the height
    // of the window to convert to OpenGL coordinates for the viewport.
    if (impl_->frameChanged) {
        impl_->frameChanged = false;

        auto f = GetFrame();
        impl_->controls->SetViewSize(Size(f.width, f.height));
        // GUI has origin of Y axis at top, but renderer has it at bottom
        // so we need to convert coordinates.
        int y = context.screenHeight - (f.height + f.y);

        auto view = impl_->scene.GetView(impl_->viewId);
        view->SetViewport(f.x, y, f.width, f.height);

        auto* camera = GetCamera();
        float aspect = 1.0f;
        if (f.height > 0) {
            aspect = float(f.width) / float(f.height);
        }
        GetCamera()->SetProjection(camera->GetFieldOfView(), aspect,
                                   camera->GetNear(), camera->GetFar(),
                                   camera->GetFieldOfViewType());
    }

    // The actual drawing is done later, at the end of drawing in
    // Window::OnDraw(), in FilamentRenderer::Draw(). We can always
    // return NONE because any changes this frame will automatically
    // be rendered (unlike the ImGUI parts).
    return Widget::DrawResult::NONE;
}

void SceneWidget::Mouse(const MouseEvent& e) { impl_->controls->Mouse(e); }

}  // namespace gui
}  // namespace open3d
