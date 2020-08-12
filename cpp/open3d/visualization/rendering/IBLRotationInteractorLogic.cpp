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

void IBLRotationInteractorLogic::ShowSkybox(bool is_on) {
    skybox_is_normally_on_ = is_on;
}

void IBLRotationInteractorLogic::StartMouseDrag() {
    ibl_rotation_at_mouse_down_ = scene_->GetIndirectLightRotation();
    auto identity = Camera::Transform::Identity();
    Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

    if (!skybox_is_normally_on_) {
        scene_->ShowSkybox(true);
    }

    ClearUI();

    UpdateMouseDragUI();
}

void IBLRotationInteractorLogic::UpdateMouseDragUI() {}

void IBLRotationInteractorLogic::EndMouseDrag() {
    ClearUI();
    if (!skybox_is_normally_on_) {
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
