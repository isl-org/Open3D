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

#include "open3d/visualization/rendering/RotationInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

RotationInteractorLogic::RotationInteractorLogic(Camera* camera,
                                                 double min_far_plane)
    : min_far_plane_(min_far_plane), camera_(camera) {}

RotationInteractorLogic::~RotationInteractorLogic() {}

void RotationInteractorLogic::SetCenterOfRotation(
        const Eigen::Vector3f& center) {
    center_of_rotation_ = center;
}

void RotationInteractorLogic::Pan(int dx, int dy) {
    Eigen::Vector3f world_move = CalcPanVectorWorld(dx, dy);
    center_of_rotation_ = center_of_rotation_at_mouse_down_ + world_move;

    auto matrix = matrix_at_mouse_down_;  // copy
    // matrix.translate(cameraLocalMove) would work if
    // matrix == camara matrix. Since it isn't necessarily true,
    // we need to translate the position of the matrix in the world
    // coordinate system.
    Eigen::Vector3f new_trans = matrix.translation() + world_move;
    matrix.fromPositionOrientationScale(new_trans, matrix.rotation(),
                                        Eigen::Vector3f(1, 1, 1));
    SetMatrix(matrix);
}

Eigen::Vector3f RotationInteractorLogic::CalcPanVectorWorld(int dx, int dy) {
    // Calculate the depth to the pixel we clicked on, so that we
    // can compensate for perspective and have the mouse stays on
    // that location. Unfortunately, we don't really have access to
    // the depth buffer with Filament, so we'll fake it by finding
    // the depth of the center of rotation.
    auto pos = camera_->GetPosition();
    auto forward = camera_->GetForwardVector();
    float near = camera_->GetNear();
    float dist = forward.dot(center_of_rotation_at_mouse_down_ - pos);
    dist = std::max(near, dist);

    // How far is one pixel?
    float half_fov = camera_->GetFieldOfView() / 2.0;
    float hal_fov_radians = half_fov * M_PI / 180.0;
    float units_at_dist = 2.0f * std::tan(hal_fov_radians) * (near + dist);
    float units_per_px = units_at_dist / float(view_height_);

    // Move camera and center of rotation. Adjust values from the
    // original positions at mousedown to avoid hysteresis problems.
    // Note that the interactor's matrix may not be the same as the
    // camera's matrix.
    Eigen::Vector3f camera_local_move(-dx * units_per_px, dy * units_per_px, 0);
    Eigen::Vector3f world_move =
            camera_->GetModelMatrix().rotation() * camera_local_move;

    return world_move;
}

void RotationInteractorLogic::StartMouseDrag() {
    Super::SetMouseDownInfo(GetMatrix(), center_of_rotation_);
}

void RotationInteractorLogic::UpdateMouseDragUI() {}

void RotationInteractorLogic::EndMouseDrag() {}

void RotationInteractorLogic::UpdateCameraFarPlane() {
    // Remember that the camera matrix is not necessarily the
    // interactor's matrix.
    // Also, the far plane needs to be able to show the
    // axis if it is visible, so we need the far plane to include
    // the origin. (See also SceneWidget::SetupCamera())
    auto pos = camera_->GetModelMatrix().translation().cast<double>();
    auto far1 = model_bounds_.GetMinBound().norm();
    auto far2 = model_bounds_.GetMaxBound().norm();
    auto far3 = pos.norm();
    auto model_size = 2.0 * model_bounds_.GetExtent().norm();
    auto far = std::max(min_far_plane_,
                        std::max(std::max(far1, far2), far3) + model_size);
    float aspect = 1.0f;
    if (view_height_ > 0) {
        aspect = float(view_width_) / float(view_height_);
    }
    camera_->SetProjection(camera_->GetFieldOfView(), aspect,
                           camera_->GetNear(), far,
                           camera_->GetFieldOfViewType());
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
