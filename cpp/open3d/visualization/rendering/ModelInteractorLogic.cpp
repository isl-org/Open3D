// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/visualization/rendering/ModelInteractorLogic.h"

#include <Eigen/src/Core/Transpose.h>

#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {
namespace rendering {

ModelInteractorLogic::ModelInteractorLogic(Open3DScene* scene,
                                           Camera* camera,
                                           double min_far_plane)
    : RotationInteractorLogic(camera, min_far_plane), scene_(scene) {}

ModelInteractorLogic::~ModelInteractorLogic() {}

void ModelInteractorLogic::SetBoundingBox(
        const geometry::AxisAlignedBoundingBox& bounds) {
    Super::SetBoundingBox(bounds);
    // Initialize parent's matrix_ (in case we do a mouse wheel, which
    // doesn't involve a mouse down) and the center of rotation.
    SetMouseDownInfo(Camera::Transform::Identity(),
                     bounds.GetCenter().cast<float>());
}

void ModelInteractorLogic::SetModel(
        GeometryHandle axes, const std::vector<GeometryHandle>& objects) {}

void ModelInteractorLogic::Rotate(int dx, int dy) {
    Eigen::Vector3f x_axis = -camera_->GetLeftVector();
    Eigen::Vector3f y_axis = camera_->GetUpVector();

    Eigen::Vector3f axis = -dy * x_axis + dx * y_axis;
    axis = axis.normalized();
    float theta = float(CalcRotateRadians(dx, dy));
    Eigen::AngleAxisf rot_matrix(theta, axis);

    // Rotations about a point using a world axis do not produce
    // a matrix that can be applied to any matrix; each individual
    // matrix must be rotated around the point.
    for (auto o : transforms_at_mouse_down_) {
        Camera::Transform t = o.second;  // copy
        t.translate(center_of_rotation_);
        t.fromPositionOrientationScale(t.translation(),
                                       rot_matrix * t.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        t.translate(-center_of_rotation_);
        scene_->GetScene()->SetGeometryTransform(o.first, t);
    }
    UpdateBoundingBox(Camera::Transform(rot_matrix));
}

void ModelInteractorLogic::RotateZ(int dx, int dy) {
    auto rad = CalcRotateZRadians(dx, dy);
    Eigen::AngleAxisf rot_matrix(rad, camera_->GetForwardVector());

    for (auto o : transforms_at_mouse_down_) {
        Camera::Transform t = o.second;  // copy
        t.translate(center_of_rotation_);
        t.fromPositionOrientationScale(t.translation(),
                                       rot_matrix * t.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        t.translate(-center_of_rotation_);
        scene_->GetScene()->SetGeometryTransform(o.first, t);
    }
    UpdateBoundingBox(Camera::Transform(rot_matrix));
}

void ModelInteractorLogic::Dolly(float dy, DragType drag_type) {
    float z_dist = CalcDollyDist(dy, drag_type, matrix_at_mouse_down_);
    Eigen::Vector3f world_move = -z_dist * camera_->GetForwardVector();

    for (auto o : transforms_at_mouse_down_) {
        Camera::Transform t;
        if (drag_type == DragType::MOUSE) {
            t = o.second;  // copy
        } else {
            t = scene_->GetScene()->GetGeometryTransform(o.first);
        }
        Eigen::Vector3f new_trans = t.translation() + world_move;
        t.fromPositionOrientationScale(new_trans, t.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        scene_->GetScene()->SetGeometryTransform(o.first, t);
    }

    Camera::Transform t = Camera::Transform::Identity();
    t.translate(world_move);
    UpdateBoundingBox(t);

    UpdateCameraFarPlane();
}

void ModelInteractorLogic::Pan(int dx, int dy) {
    Eigen::Vector3f world_move = CalcPanVectorWorld(-dx, -dy);
    center_of_rotation_ = center_of_rotation_at_mouse_down_ + world_move;

    for (auto o : transforms_at_mouse_down_) {
        Camera::Transform t = o.second;  // copy
        Eigen::Vector3f new_trans = t.translation() + world_move;
        t.fromPositionOrientationScale(new_trans, t.rotation(),
                                       Eigen::Vector3f(1, 1, 1));
        scene_->GetScene()->SetGeometryTransform(o.first, t);
    }
    Camera::Transform t = Camera::Transform::Identity();
    t.translate(world_move);
    UpdateBoundingBox(t);
}

void ModelInteractorLogic::UpdateBoundingBox(const Camera::Transform& t) {
    using Transform4 = Eigen::Transform<double, 3, Eigen::Affine>;
    Transform4 change = t.cast<double>();
    Eigen::Vector3d new_min = change * model_bounds_.GetMinBound();
    Eigen::Vector3d new_max = change * model_bounds_.GetMaxBound();
    // Call super's not our SetBoundingBox(): we also update the
    // center of rotation, because normally this call is not done during
    // mouse movement, but rather, once to initialize the interactor.
    Super::SetBoundingBox(geometry::AxisAlignedBoundingBox(new_min, new_max));
}

const std::string kAxisObjectName("__axis__");

void ModelInteractorLogic::StartMouseDrag() {
    SetMouseDownInfo(Camera::Transform::Identity(), center_of_rotation_);

    transforms_at_mouse_down_.clear();
    auto models = scene_->GetGeometries();
    auto* scene = scene_->GetScene();

    for (const auto& m : models) {
        transforms_at_mouse_down_[m] = scene->GetGeometryTransform(m);
    }

    // Show axes while user is dragging
    is_axes_visible_ = scene_->GetScene()->GeometryIsVisible(kAxisObjectName);
    if (!is_axes_visible_)
        scene_->GetScene()->ShowGeometry(kAxisObjectName, true);

    // Fix far plane if the center of the model is offset from origin
    Super::UpdateCameraFarPlane();
}

void ModelInteractorLogic::UpdateMouseDragUI() {}

void ModelInteractorLogic::EndMouseDrag() {
    if (!is_axes_visible_)
        scene_->GetScene()->ShowGeometry(kAxisObjectName, false);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
