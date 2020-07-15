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

#include "open3d/visualization/rendering/CameraInteractorLogic.h"

namespace open3d {
namespace visualization {
namespace rendering {

CameraInteractorLogic::CameraInteractorLogic(Camera* c, double min_far_plane)
    : RotationInteractorLogic(c, min_far_plane), fov_at_mouse_down_(60.0) {}

void CameraInteractorLogic::SetBoundingBox(
        const geometry::AxisAlignedBoundingBox& bounds) {
    Super::SetBoundingBox(bounds);
    // Initialize parent's matrix_ (in case we do a mouse wheel, which
    // doesn't involve a mouse down) and the center of rotation.
    SetMouseDownInfo(camera_->GetModelMatrix(),
                     bounds.GetCenter().cast<float>());
}

void CameraInteractorLogic::Rotate(int dx, int dy) {
    Super::Rotate(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractorLogic::RotateZ(int dx, int dy) {
    Super::RotateZ(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractorLogic::RotateFly(int dx, int dy) {
    // Fly/first-person shooter rotation is always about the current camera
    // matrix, and the camera's position, so we need to update Super's
    // matrix information.
    Super::SetMouseDownInfo(camera_->GetModelMatrix(), camera_->GetPosition());
    Super::Rotate(-dx, -dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractorLogic::Dolly(int dy, DragType type) {
    // Parent's matrix_ may not have been set yet
    if (type != DragType::MOUSE) {
        SetMouseDownInfo(camera_->GetModelMatrix(), center_of_rotation_);
    }
    Super::Dolly(dy, type);
}

void CameraInteractorLogic::Dolly(float z_dist, Camera::Transform matrix_in) {
    Super::Dolly(z_dist, matrix_in);
    auto matrix = GetMatrix();
    camera_->SetModelMatrix(matrix);

    UpdateCameraFarPlane();
}

void CameraInteractorLogic::Pan(int dx, int dy) {
    Super::Pan(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractorLogic::RotateLocal(float angle_rad,
                                        const Eigen::Vector3f& axis) {
    auto model_matrix = camera_->GetModelMatrix();  // copy
    model_matrix.rotate(Eigen::AngleAxis<float>(angle_rad, axis));
    camera_->SetModelMatrix(model_matrix);
}

void CameraInteractorLogic::MoveLocal(const Eigen::Vector3f& v) {
    auto model_matrix = camera_->GetModelMatrix();  // copy
    model_matrix.translate(v);
    camera_->SetModelMatrix(model_matrix);
}

void CameraInteractorLogic::Zoom(int dy, DragType drag_type) {
    float d_fov = 0.0f;  // initialize to make GCC happy
    switch (drag_type) {
        case DragType::MOUSE:
            d_fov = float(-dy) * 0.1;  // deg
            break;
        case DragType::TWO_FINGER:
            d_fov = float(dy) * 0.2f;  // deg
            break;
        case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
            d_fov = float(dy) * 2.0f;  // deg
            break;
    }
    float old_fov = 0.0;
    if (drag_type == DragType::MOUSE) {
        old_fov = fov_at_mouse_down_;
    } else {
        old_fov = camera_->GetFieldOfView();
    }
    float new_fov = old_fov + d_fov;
    new_fov = std::max(5.0f, new_fov);
    new_fov = std::min(90.0f, new_fov);

    float to_radians = M_PI / 180.0;
    float near = camera_->GetNear();
    Eigen::Vector3f camera_pos, cor;
    if (drag_type == DragType::MOUSE) {
        camera_pos = matrix_at_mouse_down_.translation();
        cor = center_of_rotation_at_mouse_down_;
    } else {
        camera_pos = camera_->GetPosition();
        cor = center_of_rotation_;
    }
    Eigen::Vector3f to_cor = cor - camera_pos;
    float old_dist_from_plane_to_cor = to_cor.norm() - near;
    float new_dist_from_plane_to_cor =
            (near + old_dist_from_plane_to_cor) *
                    std::tan(old_fov / 2.0 * to_radians) /
                    std::tan(new_fov / 2.0 * to_radians) -
            near;
    if (drag_type == DragType::MOUSE) {
        Dolly(-(new_dist_from_plane_to_cor - old_dist_from_plane_to_cor),
              matrix_at_mouse_down_);
    } else {
        Dolly(-(new_dist_from_plane_to_cor - old_dist_from_plane_to_cor),
              camera_->GetModelMatrix());
    }

    float aspect = 1.0f;
    if (view_height_ > 0) {
        aspect = float(view_width_) / float(view_height_);
    }
    camera_->SetProjection(new_fov, aspect, camera_->GetNear(),
                           camera_->GetFar(), camera_->GetFieldOfViewType());
}

void CameraInteractorLogic::StartMouseDrag() {
    Super::SetMouseDownInfo(camera_->GetModelMatrix(), center_of_rotation_);
    fov_at_mouse_down_ = camera_->GetFieldOfView();
}

void CameraInteractorLogic::ResetMouseDrag() { StartMouseDrag(); }

void CameraInteractorLogic::UpdateMouseDragUI() {}

void CameraInteractorLogic::EndMouseDrag() {}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
