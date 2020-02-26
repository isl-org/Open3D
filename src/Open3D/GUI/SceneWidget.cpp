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

#include <Eigen/Geometry>

#define ENABLE_PAN 1

namespace open3d {
namespace gui {

static const double MIN_FAR_PLANE = 100.0;

// ----------------------------------------------------------------------------
class CameraControls {
public:
    CameraControls(visualization::Camera* c) : camera_(c) {}

    void Pan(int dx, int dy) {
        // Calculate the depth to the pixel we clicked on, so that we
        // can compensate for perspective and have the mouse stays on
        // that location. Unfortunately, we don't really have access to
        // the depth buffer with Filament, so we'll fake it by finding
        // the depth of the center of rotation.
        auto pos = camera_->GetPosition();
        auto forward = camera_->GetForwardVector();
        float near = camera_->GetNear();
        float dist = forward.dot(centerOfRotationAtMouseDown_ - pos);
        dist = std::max(near, dist);

        // How far is one pixel?
        auto modelMatrix = matrixAtMouseDown_;  // copy
        float halfFoV = camera_->GetFieldOfView() / 2.0;
        float halfFoVRadians = halfFoV * M_PI / 180.0;
        float unitsAtDist = 2.0f * std::tan(halfFoVRadians) * (near + dist);
        float unitsPerPx = unitsAtDist / float(viewSize_.height);

        // Move camera and center of rotation. Adjust values from the
        // original positions at mousedown to avoid hysteresis problems.
        auto localMove = Eigen::Vector3f(-dx * unitsPerPx, dy * unitsPerPx, 0);
        auto worldMove = modelMatrix.rotation() * localMove;
        centerOfRotation_ = centerOfRotationAtMouseDown_ + worldMove;
        modelMatrix.translate(localMove);
        camera_->SetModelMatrix(modelMatrix);
    }

    void Rotate(int dx, int dy) {
        auto matrix = matrixAtMouseDown_;  // copy
        Eigen::AngleAxisf rotMatrix(0, Eigen::Vector3f(1, 0, 0));

        // We want to rotate as if we were rotating an imaginary trackball
        // centered at the point of rotation. To do this we need an axis
        // of rotation and an angle about the axis. To find the axis, we
        // imagine that the viewing plane has been translated into the screen
        // so that it intersects the center of rotation. The axis we want
        // to rotate around is perpendicular to the vector defined by (dx, dy)
        // (assuming +x is right and +y is up). (Imagine the situation if the
        // mouse movement is (100, 0) or (0, 100).) Now it is easy to find
        // the perpendicular in 2D. Conveniently, (axis.x, axis.y, 0) is the
        // correct axis in camera-local coordinates. We can multiply by the
        // camera's rotation matrix to get the correct world vector.
        dy = -dy;  // up is negative, but the calculations are easiest to
                   // imagine up is positive.
        Eigen::Vector3f axis(-dy, dx, 0);  // rotate by 90 deg in 2D
        float theta =
                0.5 * M_PI * axis.norm() / (0.5f * float(viewSize_.height));
        axis = axis.normalized();

        axis = matrix.rotation() * axis;  // convert axis to world coords
        rotMatrix = rotMatrix * Eigen::AngleAxisf(-theta, axis);

        auto pos = matrix * Eigen::Vector3f(0, 0, 0);
        auto dist = (centerOfRotation_ - pos).norm();
        visualization::Camera::Transform m;
        m.fromPositionOrientationScale(centerOfRotation_,
                                       rotMatrix * matrix.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        m.translate(Eigen::Vector3f(0, 0, dist));

        camera_->SetModelMatrix(m);
    }

    void RotateZ(int dx, int dy) {
        // RotateZ rotates around the axis normal to the screen. Since we
        // will be rotating using camera coordinates, we want to rotate
        // about (0, 0, 1).
        Eigen::Vector3f axis(0, 0, 1);
        // Moving half the height should rotate 360 deg (= 2 * PI).
        // This makes it easy to rotate enough without rotating too much.
        auto rad = 4.0 * M_PI * dy / viewSize_.height;

        auto matrix = matrixAtMouseDown_;  // copy
        matrix.rotate(Eigen::AngleAxisf(rad, axis));
        camera_->SetModelMatrix(matrix);
    }

    enum class DragType { MOUSE, WHEEL, TWO_FINGER };

    void Zoom(int dy, DragType dragType) {
        float dFOV = 0.0f;  // initialize to make GCC happy
        switch (dragType) {
            case DragType::MOUSE:
                dFOV = float(-dy) * 0.1;  // deg
                break;
            case DragType::TWO_FINGER:
                dFOV = float(dy) * 0.2f;  // deg
                break;
            case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
                dFOV = float(dy) * 2.0f;  // deg
                break;
        }
        float newFOV = 0.0;
        if (dragType == DragType::MOUSE) {
            newFOV = fovAtMouseDown_ + dFOV;
        } else {
            newFOV = camera_->GetFieldOfView() + dFOV;
        }
        newFOV = std::max(2.0f, newFOV);
        newFOV = std::min(175.0f, newFOV);
        
        float aspect = 1.0f;
        if (viewSize_.height > 0) {
            aspect = float(viewSize_.width) / float(viewSize_.height);
        }
        camera_->SetProjection(newFOV, aspect,
                               camera_->GetNear(), camera_->GetFar(),
                               camera_->GetFieldOfViewType());
    }

    void Dolly(int dy, DragType dragType) {
        float dist = 0.0f;  // initialize to make GCC happy
        switch (dragType) {
            case DragType::MOUSE:
                // Zoom out is "push away" or up, is a negative value for
                // mousing
                dist = float(dy) * 0.0025f * modelSize_;
                break;
            case DragType::TWO_FINGER:
                // Zoom out is "push away" or up, is a positive value for
                // two-finger scrolling, so we need to invert dy.
                dist = float(-dy) * 0.005f * modelSize_;
                break;
            case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
                dist = float(-dy) * 0.1f * modelSize_;
                break;
        }

        // Dolly is just moving the camera forward. Filament uses right as +x,
        // up as +y, and forward as -z (standard OpenGL coordinates). So to
        // move forward all we need to do is translate the camera matrix by
        // dist * (0, 0, -1). Note that translating by camera_->GetForwardVector
        // would be incorrect, since GetForwardVector() returns the forward
        // vector in world space, but the translation happens in camera space.)
        // Since we want trackpad down (negative) to go forward ("pulling" the
        // model toward the viewer) we need to negate dy.
        auto forward = Eigen::Vector3f(0, 0, -dist);  // dist * (0, 0, -1)
        visualization::Camera::Transform matrix;
        if (dragType == DragType::MOUSE) {
            matrix = matrixAtMouseDown_;  // copy
            matrix.translate(forward);
        } else {
            matrix = camera_->GetModelMatrix().translate(forward);
        }
        camera_->SetModelMatrix(matrix);

        // Update the far plane so that we don't get clipped by it as we dolly
        // out or lose precision as we dolly in.
        auto pos = matrix.translation().cast<double>();
        auto far1 = (geometryBounds_.GetMinBound() - pos).norm();
        auto far2 = (geometryBounds_.GetMaxBound() - pos).norm();
        auto modelSize = 2.0 * geometryBounds_.GetExtent().norm();
        auto far = std::max(MIN_FAR_PLANE, std::max(far1, far2) + modelSize);
        float aspect = 1.0f;
        if (viewSize_.height > 0) {
            aspect = float(viewSize_.width) / float(viewSize_.height);
        }
        camera_->SetProjection(camera_->GetFieldOfView(), aspect,
                               camera_->GetNear(), far,
                               camera_->GetFieldOfViewType());
    }

    void GoToPreset(SceneWidget::CameraPreset preset) {
        auto boundsMax = geometryBounds_.GetMaxBound();
        auto maxDim =
                std::max(boundsMax.x(), std::max(boundsMax.y(), boundsMax.z()));
        maxDim = 1.5f * maxDim;
        auto center = centerOfRotation_;
        Eigen::Vector3f eye, up;
        switch (preset) {
            case SceneWidget::CameraPreset::PLUS_X: {
                eye = Eigen::Vector3f(maxDim, center.y(), center.z());
                up = Eigen::Vector3f(0, 1, 0);
                break;
            }
            case SceneWidget::CameraPreset::PLUS_Y: {
                eye = Eigen::Vector3f(center.x(), maxDim, center.z());
                up = Eigen::Vector3f(1, 0, 0);
                break;
            }
            case SceneWidget::CameraPreset::PLUS_Z: {
                eye = Eigen::Vector3f(center.x(), center.y(), maxDim);
                up = Eigen::Vector3f(0, 1, 0);
                break;
            }
        }
        camera_->LookAt(center, eye, up);
    }

    void SetViewSize(const Size& size) { viewSize_ = size; }

    void SetBoundingBox(const geometry::AxisAlignedBoundingBox& bounds) {
        geometryBounds_ = bounds;
        modelSize_ = (bounds.GetMaxBound() - bounds.GetMinBound()).norm();
        centerOfRotation_ = bounds.GetCenter().cast<float>();
    }

    void Mouse(const MouseEvent& e) {
        switch (e.type) {
            case MouseEvent::BUTTON_DOWN:
                mouseDownX_ = e.x;
                mouseDownY_ = e.y;
                matrixAtMouseDown_ = camera_->GetModelMatrix();
                centerOfRotationAtMouseDown_ = centerOfRotation_;
                fovAtMouseDown_ = camera_->GetFieldOfView();
                if (e.button.button == MouseButton::LEFT) {
                    if (e.modifiers & int(KeyModifier::SHIFT)) {
                        state_ = State::DOLLY;
#if ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::CTRL)) {
                        state_ = State::PAN;
#endif  // ENABLE_PAN
                    } else if (e.modifiers & int(KeyModifier::META)) {
                        state_ = State::ROTATE_Z;
                    } else {
                        state_ = State::ROTATE_XY;
                    }
#if ENABLE_PAN
                } else if (e.button.button == MouseButton::RIGHT) {
                    state_ = State::PAN;
#endif  // ENABLE_PAN
                }
                break;
            case MouseEvent::DRAG: {
                int dx = e.x - mouseDownX_;
                int dy = e.y - mouseDownY_;
                switch (state_) {
                    case State::NONE:
                        break;
                    case State::PAN:
                        Pan(dx, dy);
                        break;
                    case State::DOLLY:
                        Dolly(dy, DragType::MOUSE);
                        break;
                    case State::ZOOM:
                        Zoom(dy, DragType::MOUSE);
                        break;
                    case State::ROTATE_XY:
                        Rotate(dx, dy);
                        break;
                    case State::ROTATE_Z:
                        RotateZ(dx, dy);
                        break;
                }
                break;
            }
            case MouseEvent::WHEEL: {
                Zoom(e.wheel.dy, e.wheel.isTrackpad ? DragType::TWO_FINGER
                                                    : DragType::WHEEL);
                break;
            }
            case MouseEvent::BUTTON_UP:
                state_ = State::NONE;
                break;
            default:
                break;
        }
    }

    void Key(const KeyEvent& e) {}

private:
    visualization::Camera* camera_;
    Size viewSize_;
    double modelSize_ = 100;
    geometry::AxisAlignedBoundingBox geometryBounds_;
    Eigen::Vector3f centerOfRotation_;

    visualization::Camera::Transform matrixAtMouseDown_;
    Eigen::Vector3f centerOfRotationAtMouseDown_;
    double fovAtMouseDown_;
    int mouseDownX_;
    int mouseDownY_;

    enum class State { NONE, PAN, DOLLY, ZOOM, ROTATE_XY, ROTATE_Z };
    State state_ = State::NONE;
};
// ----------------------------------------------------------------------------
struct SceneWidget::Impl {
    visualization::Scene& scene;
    visualization::ViewHandle viewId;
    //    std::unique_ptr<visualization::CameraManipulator> cameraManipulator;
    std::shared_ptr<CameraControls> controls;
    bool frameChanged = false;

    explicit Impl(visualization::Scene& aScene) : scene(aScene) {}
};

SceneWidget::SceneWidget(visualization::Scene& scene) : impl_(new Impl(scene)) {
    impl_->viewId = scene.AddView(0, 0, 1, 1);

    auto view = impl_->scene.GetView(impl_->viewId);
    impl_->controls = std::make_shared<CameraControls>(view->GetCamera());
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
    auto near = 0.1f;
    auto far = std::max(MIN_FAR_PLANE, 2.0 * geometryBounds.GetExtent().norm());
    GetCamera()->SetProjection(verticalFoV, aspect, near, far,
                               visualization::Camera::FovType::Vertical);

    GoToCameraPreset(CameraPreset::PLUS_Z);  // default OpenGL view
}

void SceneWidget::GoToCameraPreset(CameraPreset preset) {
    impl_->controls->GoToPreset(preset);
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

void SceneWidget::Key(const KeyEvent& e) { impl_->controls->Key(e); }

}  // namespace gui
}  // namespace open3d
