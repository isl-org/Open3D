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

#include "CameraInteractorLogic.h"

namespace open3d {
namespace visualization {

CameraInteractorLogic::CameraInteractorLogic(Camera* c, double minFarPlane)
    : RotationInteractorLogic(c, minFarPlane), fovAtMouseDown_(60.0) {}

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
        SetMouseDownInfo(camera_->GetModelMatrix(), centerOfRotation_);
    }
    Super::Dolly(dy, type);
}

void CameraInteractorLogic::Dolly(float zDist, Camera::Transform matrixIn) {
    Super::Dolly(zDist, matrixIn);
    auto matrix = GetMatrix();
    camera_->SetModelMatrix(matrix);

    UpdateCameraFarPlane();
}

void CameraInteractorLogic::Pan(int dx, int dy) {
    Super::Pan(dx, dy);
    camera_->SetModelMatrix(GetMatrix());
}

void CameraInteractorLogic::RotateLocal(float angleRad,
                                        const Eigen::Vector3f& axis) {
    auto modelMatrix = camera_->GetModelMatrix();  // copy
    modelMatrix.rotate(Eigen::AngleAxis<float>(angleRad, axis));
    camera_->SetModelMatrix(modelMatrix);
}

void CameraInteractorLogic::MoveLocal(const Eigen::Vector3f& v) {
    auto modelMatrix = camera_->GetModelMatrix();  // copy
    modelMatrix.translate(v);
    camera_->SetModelMatrix(modelMatrix);
}

void CameraInteractorLogic::Zoom(int dy, DragType dragType) {
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

void CameraInteractorLogic::StartMouseDrag() {
    Super::SetMouseDownInfo(camera_->GetModelMatrix(), centerOfRotation_);
    fovAtMouseDown_ = camera_->GetFieldOfView();
}

void CameraInteractorLogic::ResetMouseDrag() { StartMouseDrag(); }

void CameraInteractorLogic::UpdateMouseDragUI() {}

void CameraInteractorLogic::EndMouseDrag() {}

}  // namespace visualization
}  // namespace open3d
