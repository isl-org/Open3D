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

#include "CameraInteractor.h"

namespace open3d {
namespace visualization {

CameraInteractor::CameraInteractor(Camera* c, double minFarPlane)
    : minFarPlane_(minFarPlane), camera_(c), fovAtMouseDown_(60.0) {}

void CameraInteractor::SetBoundingBox(
        const geometry::AxisAlignedBoundingBox& bounds) {
    Super::SetBoundingBox(bounds);
    // Initialize parent's matrix_ (in case we do a mouse wheel, which
    // doesn't involve a mouse down) and the center of rotation.
    SetMouseDownInfo(camera_->GetModelMatrix(),
                     bounds.GetCenter().cast<float>());
}

void CameraInteractor::SetCenterOfRotation(const Eigen::Vector3f& center) {
    centerOfRotation_ = center;
}

void CameraInteractor::Rotate(int dx, int dy) {
    Super::Rotate(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractor::RotateZ(int dx, int dy) {
    Super::RotateZ(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractor::Dolly(int dy, DragType type) {
    // Parent's matrix_ may not have been set yet
    if (type != DragType::MOUSE) {
        SetMouseDownInfo(camera_->GetModelMatrix(), centerOfRotation_);
    }
    Super::Dolly(dy, type);
}

void CameraInteractor::Dolly(float zDist, Camera::Transform matrixIn) {
    Super::Dolly(zDist, matrixIn);
    auto matrix = GetMatrix();
    camera_->SetModelMatrix(matrix);

    // Update the far plane so that we don't get clipped by it as we dolly
    // out or lose precision as we dolly in.
    auto pos = matrix.translation().cast<double>();
    auto far1 = (modelBounds_.GetMinBound() - pos).norm();
    auto far2 = (modelBounds_.GetMaxBound() - pos).norm();
    auto modelSize = 2.0 * modelBounds_.GetExtent().norm();
    auto far = std::max(minFarPlane_, std::max(far1, far2) + modelSize);
    float aspect = 1.0f;
    if (viewHeight_ > 0) {
        aspect = float(viewWidth_) / float(viewHeight_);
    }
    camera_->SetProjection(camera_->GetFieldOfView(), aspect,
                           camera_->GetNear(), far,
                           camera_->GetFieldOfViewType());
}

void CameraInteractor::Pan(int dx, int dy) {
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
    float unitsPerPx = unitsAtDist / float(viewHeight_);

    // Move camera and center of rotation. Adjust values from the
    // original positions at mousedown to avoid hysteresis problems.
    auto localMove = Eigen::Vector3f(-dx * unitsPerPx, dy * unitsPerPx, 0);
    auto worldMove = modelMatrix.rotation() * localMove;
    centerOfRotation_ = centerOfRotationAtMouseDown_ + worldMove;
    modelMatrix.translate(localMove);
    camera_->SetModelMatrix(modelMatrix);
}

void CameraInteractor::RotateLocal(float angleRad,
                                   const Eigen::Vector3f& axis) {
    auto modelMatrix = camera_->GetModelMatrix();  // copy
    modelMatrix.rotate(Eigen::AngleAxis<float>(angleRad, axis));
    camera_->SetModelMatrix(modelMatrix);
}

void CameraInteractor::MoveLocal(const Eigen::Vector3f& v) {
    auto modelMatrix = camera_->GetModelMatrix();  // copy
    modelMatrix.translate(v);
    camera_->SetModelMatrix(modelMatrix);
}

void CameraInteractor::Zoom(int dy, DragType dragType) {
    float dFOV = 0.0f;  // initialize to make GCC happy
    switch (dragType) {
        case DragType::MOUSE:
            dFOV = float(-dy) * 0.1;  // deg
            break;
        case DragType::TWO_FINGER:
            dFOV = float(dy) * 0.2f;  // deg
            break;
        case DragType::WHEEL:         // actual mouse wheel, same as two-fingers
            dFOV = float(dy) * 2.0f;  // deg
            break;
    }
    float oldFOV = 0.0;
    if (dragType == DragType::MOUSE) {
        oldFOV = fovAtMouseDown_;
    } else {
        oldFOV = camera_->GetFieldOfView();
    }
    float newFOV = oldFOV + dFOV;
    newFOV = std::max(5.0f, newFOV);
    newFOV = std::min(90.0f, newFOV);

    float toRadians = M_PI / 180.0;
    float near = camera_->GetNear();
    Eigen::Vector3f cameraPos, COR;
    if (dragType == DragType::MOUSE) {
        cameraPos = matrixAtMouseDown_.translation();
        COR = centerOfRotationAtMouseDown_;
    } else {
        cameraPos = camera_->GetPosition();
        COR = centerOfRotation_;
    }
    Eigen::Vector3f toCOR = COR - cameraPos;
    float oldDistFromPlaneToCOR = toCOR.norm() - near;
    float newDistFromPlaneToCOR = (near + oldDistFromPlaneToCOR) *
                                          std::tan(oldFOV / 2.0 * toRadians) /
                                          std::tan(newFOV / 2.0 * toRadians) -
                                  near;
    if (dragType == DragType::MOUSE) {
        Dolly(-(newDistFromPlaneToCOR - oldDistFromPlaneToCOR),
              matrixAtMouseDown_);
    } else {
        Dolly(-(newDistFromPlaneToCOR - oldDistFromPlaneToCOR),
              camera_->GetModelMatrix());
    }

    float aspect = 1.0f;
    if (viewHeight_ > 0) {
        aspect = float(viewWidth_) / float(viewHeight_);
    }
    camera_->SetProjection(newFOV, aspect, camera_->GetNear(),
                           camera_->GetFar(), camera_->GetFieldOfViewType());
}

void CameraInteractor::StartMouseDrag() {
    matrixAtMouseDown_ = camera_->GetModelMatrix();
    centerOfRotationAtMouseDown_ = centerOfRotation_;
    fovAtMouseDown_ = camera_->GetFieldOfView();

    Super::SetMouseDownInfo(matrixAtMouseDown_, centerOfRotation_);
}

void CameraInteractor::UpdateMouseDragUI() {}

void CameraInteractor::EndMouseDrag() {}

}  // namespace visualization
}  // namespace open3d
