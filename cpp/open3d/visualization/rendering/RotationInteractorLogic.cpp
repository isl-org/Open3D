// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    float near = float(camera_->GetNear());
    float dist = forward.dot(center_of_rotation_at_mouse_down_ - pos);
    dist = std::max(near, dist);

    // How far is one pixel?
    float half_fov = float(camera_->GetFieldOfView() / 2.0);
    float hal_fov_radians = half_fov * float(M_PI / 180.0);
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
    // the origin.
    auto far = Camera::CalcFarPlane(*camera_, model_bounds_);
    auto proj = camera_->GetProjection();
    if (proj.is_intrinsic) {
        Eigen::Matrix3d intrinsic;
        intrinsic << proj.proj.intrinsics.fx, 0.0, proj.proj.intrinsics.cx, 0.0,
                proj.proj.intrinsics.fy, proj.proj.intrinsics.cy, 0.0, 0.0, 1.0;
        camera_->SetProjection(intrinsic, proj.proj.intrinsics.near_plane, far,
                               proj.proj.intrinsics.width,
                               proj.proj.intrinsics.height);
    } else if (proj.is_ortho) {
        camera_->SetProjection(proj.proj.ortho.projection, proj.proj.ortho.left,
                               proj.proj.ortho.right, proj.proj.ortho.bottom,
                               proj.proj.ortho.top, proj.proj.ortho.near_plane,
                               far);
    } else {
        camera_->SetProjection(proj.proj.perspective.fov,
                               proj.proj.perspective.aspect,
                               proj.proj.perspective.near_plane, far,
                               proj.proj.perspective.fov_type);
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
