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

#include "RotationInteractorLogic.h"

namespace open3d {
namespace visualization {

RotationInteractorLogic::RotationInteractorLogic(visualization::Camera* camera,
                                                 double minFarPlane)
    : minFarPlane_(minFarPlane), camera_(camera) {}

RotationInteractorLogic::~RotationInteractorLogic() {}

void RotationInteractorLogic::SetCenterOfRotation(
        const Eigen::Vector3f& center) {
    centerOfRotation_ = center;
}

void RotationInteractorLogic::Pan(int dx, int dy) {
    Eigen::Vector3f worldMove = CalcPanVectorWorld(dx, dy);
    centerOfRotation_ = centerOfRotationAtMouseDown_ + worldMove;

    auto matrix = matrixAtMouseDown_;  // copy
    // matrix.translate(cameraLocalMove) would work if
    // matrix == camara matrix. Since it isn't necessarily true,
    // we need to translate the position of the matrix in the world
    // coordinate system.
    Eigen::Vector3f newTrans = matrix.translation() + worldMove;
    matrix.fromPositionOrientationScale(newTrans, matrix.rotation(),
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
    float dist = forward.dot(centerOfRotationAtMouseDown_ - pos);
    dist = std::max(near, dist);

    // How far is one pixel?
    float halfFoV = camera_->GetFieldOfView() / 2.0;
    float halfFoVRadians = halfFoV * M_PI / 180.0;
    float unitsAtDist = 2.0f * std::tan(halfFoVRadians) * (near + dist);
    float unitsPerPx = unitsAtDist / float(viewHeight_);

    // Move camera and center of rotation. Adjust values from the
    // original positions at mousedown to avoid hysteresis problems.
    // Note that the interactor's matrix may not be the same as the
    // camera's matrix.
    Eigen::Vector3f cameraLocalMove(-dx * unitsPerPx, dy * unitsPerPx, 0);
    Eigen::Vector3f worldMove =
            camera_->GetModelMatrix().rotation() * cameraLocalMove;

    return worldMove;
}

void RotationInteractorLogic::StartMouseDrag() {
    Super::SetMouseDownInfo(GetMatrix(), centerOfRotation_);
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
    auto far1 = modelBounds_.GetMinBound().norm();
    auto far2 = modelBounds_.GetMaxBound().norm();
    auto far3 = pos.norm();
    auto modelSize = 2.0 * modelBounds_.GetExtent().norm();
    auto far = std::max(minFarPlane_,
                        std::max(std::max(far1, far2), far3) + modelSize);
    float aspect = 1.0f;
    if (viewHeight_ > 0) {
        aspect = float(viewWidth_) / float(viewHeight_);
    }
    camera_->SetProjection(camera_->GetFieldOfView(), aspect,
                           camera_->GetNear(), far,
                           camera_->GetFieldOfViewType());
}

}  // namespace visualization
}  // namespace open3d
