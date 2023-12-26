// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/IBLRotationInteractorLogic.h"

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {
namespace rendering {

IBLRotationInteractorLogic::IBLRotationInteractorLogic(Scene* scene,
                                                       Camera* camera)
    : scene_(scene), camera_(camera) {}

void IBLRotationInteractorLogic::Rotate(int dx, int dy) {
    Eigen::Vector3f up = camera_->GetUpVector();
    Eigen::Vector3f right = -camera_->GetLeftVector();
    RotateWorld(-dx, -dy, up, right);
    scene_->SetIndirectLightRotation(GetCurrentRotation());
    UpdateMouseDragUI();
}

void IBLRotationInteractorLogic::RotateZ(int dx, int dy) {
    Eigen::Vector3f forward = camera_->GetForwardVector();
    RotateWorld(0, dy, {0, 0, 0}, forward);
    scene_->SetIndirectLightRotation(GetCurrentRotation());
    UpdateMouseDragUI();
}

void IBLRotationInteractorLogic::StartMouseDrag() {
    ibl_rotation_at_mouse_down_ = scene_->GetIndirectLightRotation();
    auto identity = Camera::Transform::Identity();
    Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

    skybox_currently_visible_ = scene_->GetSkyboxVisible();
    if (!skybox_currently_visible_) {
        scene_->ShowSkybox(true);
    }

    ClearUI();

    UpdateMouseDragUI();
}

void IBLRotationInteractorLogic::UpdateMouseDragUI() {}

void IBLRotationInteractorLogic::EndMouseDrag() {
    ClearUI();
    if (!skybox_currently_visible_) {
        scene_->ShowSkybox(false);
    }
}

void IBLRotationInteractorLogic::ClearUI() {}

Camera::Transform IBLRotationInteractorLogic::GetCurrentRotation() const {
    return GetMatrix() * ibl_rotation_at_mouse_down_;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
