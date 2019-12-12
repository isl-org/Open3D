// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "CameraManipulator.h"

#include "Camera.h"

namespace open3d {
namespace visualization {

CameraManipulator::CameraManipulator(Camera& aCamera,
                                     const float w,
                                     const float h)
    : camera_(aCamera),
      viewportW_(w),
      viewportH_(h),
      fov_(90.f),
      near_(0.01f),
      far_(100.f) {
    UpdateCameraProjection();
}

void CameraManipulator::SetViewport(const float w, const float h) {
    viewportW_ = w;
    viewportH_ = h;

    UpdateCameraProjection();
}

void CameraManipulator::SetFov(const float aFov) {
    fov_ = aFov;

    UpdateCameraProjection();
}

void CameraManipulator::SetNearPlane(const float aNear) {
    near_ = aNear;

    UpdateCameraProjection();
}

void CameraManipulator::SetFarPlane(const float aFar) {
    far_ = aFar;

    UpdateCameraProjection();
}

Eigen::Vector3f CameraManipulator::GetPosition() {
    return camera_.GetPosition();
}

Eigen::Vector3f CameraManipulator::GetForwardVector() {
    return camera_.GetForwardVector();
}

Eigen::Vector3f CameraManipulator::GetLeftVector() {
    return camera_.GetLeftVector();
}

Eigen::Vector3f CameraManipulator::GetUpVector() {
    return camera_.GetUpVector();
}

void CameraManipulator::LookAt(const Eigen::Vector3f& center,
                               const Eigen::Vector3f& eye,
                               const Eigen::Vector3f& up /*= {0, 1.f, 0.f}*/) {
    camera_.LookAt(center, eye, up);
}

void CameraManipulator::UpdateCameraProjection() {
    const float aspect = viewportW_ / viewportH_;
    camera_.SetProjection(fov_, aspect, near_, far_, Camera::FovType::Vertical);
}

}
}
