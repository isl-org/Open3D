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

#include "IBLRotationInteractor.h"

#include "Camera.h"
#include "Scene.h"

#include "Open3D/Geometry/TriangleMesh.h"

namespace open3d {
namespace visualization {

IBLRotationInteractor::IBLRotationInteractor(Scene* scene, Camera* camera)
    : scene_(scene), camera_(camera) {}

void IBLRotationInteractor::Rotate(int dx, int dy) {
    Eigen::Vector3f up = camera_->GetUpVector();
    Eigen::Vector3f right = -camera_->GetLeftVector();
    RotateWorld(-dx, -dy, up, right);
    scene_->SetIndirectLightRotation(GetCurrentRotation());
    UpdateMouseDragUI();
}

void IBLRotationInteractor::StartMouseDrag() {
    iblRotationAtMouseDown_ = scene_->GetIndirectLightRotation();
    auto identity = Camera::Transform::Identity();
    Super::SetMouseDownInfo(identity, {0.0f, 0.0f, 0.0f});

    ClearUI();

    UpdateMouseDragUI();
}

void IBLRotationInteractor::UpdateMouseDragUI() {
    Camera::Transform current = GetCurrentRotation();
    for (auto& o : uiObjs_) {
        scene_->SetEntityTransform(o.handle, current);
    }
}

void IBLRotationInteractor::EndMouseDrag() { ClearUI(); }

void IBLRotationInteractor::ClearUI() {
    for (auto& o : uiObjs_) {
        scene_->RemoveGeometry(o.handle);
    }
    uiObjs_.clear();
}

Camera::Transform IBLRotationInteractor::GetCurrentRotation() const {
    return GetMatrix() * iblRotationAtMouseDown_;
}

}  // namespace visualization
}  // namespace open3d
