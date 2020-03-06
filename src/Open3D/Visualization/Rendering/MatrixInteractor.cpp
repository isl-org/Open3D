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

#include "MatrixInteractor.h"

namespace open3d {
namespace visualization {

MatrixInteractor::~MatrixInteractor() {}

void MatrixInteractor::SetViewSize(int width, int height) {
    viewWidth_ = width;
    viewHeight_ = height;
}

void MatrixInteractor::SetBoundingBox(
        const geometry::AxisAlignedBoundingBox& bounds) {
    modelSize_ = (bounds.GetMaxBound() - bounds.GetMinBound()).norm();
    modelBounds_ = bounds;
}

void MatrixInteractor::SetMouseDownInfo(
        const Camera::Transform& matrix,
        const Eigen::Vector3f& centerOfRotation) {
    matrix_ = matrix;
    centerOfRotation_ = centerOfRotation;

    matrixAtMouseDown_ = matrix;
    centerOfRotationAtMouseDown_ = centerOfRotation;
}

const Camera::Transform& MatrixInteractor::GetMatrix() const { return matrix_; }

void MatrixInteractor::Rotate(int dx, int dy) {
    auto matrix = matrixAtMouseDown_;  // copy
    Eigen::AngleAxisf rotMatrix(0, Eigen::Vector3f(1, 0, 0));

    // We want to rotate as if we were rotating an imaginary trackball
    // centered at the point of rotation. To do this we need an axis
    // of rotation and an angle about the axis. To find the axis, we
    // imagine that the viewing plane has been translated into the screen
    // so that it intersects the center of rotation. The axis we want
    // to rotate around is perpendicular to the vector defined by (dx, dy)
    // (assuming +x is right and +y is up). (Imagine the situation if the
    // mouse movement is (100, 0) or (0, 100).) Now it is easy to find
    // the perpendicular in 2D. Conveniently, (axis.x, axis.y, 0) is the
    // correct axis in camera-local coordinates. We can multiply by the
    // camera's rotation matrix to get the correct world vector.
    dy = -dy;  // up is negative, but the calculations are easiest to
               // imagine up is positive.
    Eigen::Vector3f axis(-dy, dx, 0);  // rotate by 90 deg in 2D
    float theta = 0.5 * M_PI * axis.norm() / (0.5f * float(viewHeight_));
    axis = axis.normalized();

    axis = matrix.rotation() * axis;  // convert axis to world coords
    rotMatrix = rotMatrix * Eigen::AngleAxisf(-theta, axis);

    auto pos = matrix * Eigen::Vector3f(0, 0, 0);
    auto dist = (centerOfRotation_ - pos).norm();
    visualization::Camera::Transform m;
    m.fromPositionOrientationScale(centerOfRotation_,
                                   rotMatrix * matrix.rotation(),
                                   Eigen::Vector3f(1, 1, 1));
    m.translate(Eigen::Vector3f(0, 0, dist));

    matrix_ = m;
}

void MatrixInteractor::RotateWorld(int dx,
                                   int dy,
                                   const Eigen::Vector3f& xAxis,
                                   const Eigen::Vector3f& yAxis) {
    auto matrix = matrixAtMouseDown_;  // copy

    dy = -dy;  // up is negative, but the calculations are easiest to
               // imagine up is positive.
    Eigen::Vector3f axis = dx * xAxis + dy * yAxis;
    float theta = 0.5 * M_PI * axis.norm() / (0.5f * float(viewHeight_));
    axis = axis.normalized();

    axis = matrix.rotation() * axis;  // convert axis to world coords
    auto rotMatrix = visualization::Camera::Transform::Identity() *
                     Eigen::AngleAxisf(-theta, axis);

    auto pos = matrix * Eigen::Vector3f(0, 0, 0);
    auto dist = (centerOfRotation_ - pos).norm();
    visualization::Camera::Transform m;
    m.fromPositionOrientationScale(centerOfRotation_,
                                   rotMatrix * matrix.rotation(),
                                   Eigen::Vector3f(1, 1, 1));
    m.translate(Eigen::Vector3f(0, 0, dist));

    matrix_ = m;
}

void MatrixInteractor::RotateZ(int dx, int dy) {
    // RotateZ rotates around the axis normal to the screen. Since we
    // will be rotating using camera coordinates, we want to rotate
    // about (0, 0, 1).
    Eigen::Vector3f axis(0, 0, 1);
    // Moving half the height should rotate 360 deg (= 2 * PI).
    // This makes it easy to rotate enough without rotating too much.
    auto rad = 4.0 * M_PI * dy / viewHeight_;

    auto matrix = matrixAtMouseDown_;  // copy
    matrix.rotate(Eigen::AngleAxisf(rad, axis));
    matrix_ = matrix;
}

enum class DragType { MOUSE, WHEEL, TWO_FINGER };

void MatrixInteractor::Dolly(int dy, DragType dragType) {
    float dist = 0.0f;  // initialize to make GCC happy
    switch (dragType) {
        case DragType::MOUSE:
            // Zoom out is "push away" or up, is a negative value for
            // mousing
            dist = float(dy) * 0.0025f * modelSize_;
            break;
        case DragType::TWO_FINGER:
            // Zoom out is "push away" or up, is a positive value for
            // two-finger scrolling, so we need to invert dy.
            dist = float(-dy) * 0.005f * modelSize_;
            break;
        case DragType::WHEEL:  // actual mouse wheel, same as two-fingers
            dist = float(-dy) * 0.1f * modelSize_;
            break;
    }

    if (dragType == DragType::MOUSE) {
        Dolly(dist, matrixAtMouseDown_);  // copies the matrix
    } else {
        Dolly(dist, matrix_);
    }
}

// Note: we pass `matrix` by value because we want to copy it,
//       as translate() will be modifying it.
void MatrixInteractor::Dolly(float zDist, Camera::Transform matrix) {
    // Dolly is just moving the camera forward. Filament uses right as +x,
    // up as +y, and forward as -z (standard OpenGL coordinates). So to
    // move forward all we need to do is translate the camera matrix by
    // dist * (0, 0, -1). Note that translating by camera_->GetForwardVector
    // would be incorrect, since GetForwardVector() returns the forward
    // vector in world space, but the translation happens in camera space.)
    // Since we want trackpad down (negative) to go forward ("pulling" the
    // model toward the viewer) we need to negate dy.
    auto forward = Eigen::Vector3f(0, 0, -zDist);  // zDist * (0, 0, -1)
    matrix.translate(forward);
    matrix_ = matrix;
}

}  // namespace visualization
}  // namespace open3d
