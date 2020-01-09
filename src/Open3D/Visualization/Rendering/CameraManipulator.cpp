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

#include "Open3D/Utility/Eigen.h"

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

void CameraManipulator::SetPosition(const Eigen::Vector3f& pos) {
    auto transform = camera_.GetModelMatrix();
    transform.matrix().col(3) = Eigen::Vector4f(pos.x(), pos.y(), pos.z(), 1.f);
    camera_.SetModelMatrix(transform);
}

Eigen::Vector3f CameraManipulator::GetPosition() {
    return camera_.GetPosition();
}

void CameraManipulator::SetForwardVector(const Eigen::Vector3f& forward) {
    auto newForward = -forward;
    auto newLeft = -(newForward.cross(Eigen::Vector3f{0, 1.f, 0.f}));
    auto newUp = newLeft.cross(-newForward);

    auto transform = camera_.GetModelMatrix();
    transform.matrix().col(0) =
            Eigen::Vector4f(newLeft.x(), newLeft.y(), newLeft.z(), 0.f)
                    .normalized();
    transform.matrix().col(1) =
            Eigen::Vector4f(newUp.x(), newUp.y(), newUp.z(), 0.f).normalized();
    transform.matrix().col(2) =
            Eigen::Vector4f(newForward.x(), newForward.y(), newForward.z(), 0.f)
                    .normalized();

    camera_.SetModelMatrix(transform);
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

void CameraManipulator::SetCameraTransform(const Camera::Transform& transform) {
    camera_.SetModelMatrix(transform);
}

void CameraManipulator::LookAt(const Eigen::Vector3f& center,
                               const Eigen::Vector3f& eye,
                               const Eigen::Vector3f& up /*= {0, 1.f, 0.f}*/) {
    camera_.LookAt(center, eye, up);
}

void CameraManipulator::Orbit(const Eigen::Vector3f& center,
                              const float radius,
                              const float deltaPhi,
                              const float deltaTheta,
                              const float rotationSpeed) {
    auto transform = camera_.GetModelMatrix();
    auto pos = transform.translation();

    Eigen::Vector2f delta = {deltaPhi*rotationSpeed, deltaTheta*rotationSpeed};

    Eigen::Vector3f direction = center - pos;
    direction.normalize();

    Eigen::Vector3f leftVector;

    auto facing = direction.dot(Eigen::Vector3f::UnitY());
    if ((1.f - fabsf(facing)) < 0.001f) {
        auto up = camera_.GetUpVector();
        direction = std::copysignf(1.f, facing) * Eigen::Vector3f::UnitY();

        auto rotY = Eigen::AngleAxis<float>(delta.x(), Eigen::Vector3f::UnitY());
        up = rotY * up;

        leftVector = direction.cross(up);
        leftVector.normalize();

        direction = up.cross(leftVector);
        direction.normalize();

        if ((facing > 0.f && deltaTheta < 0.f) ||
            (facing < 0.f && deltaTheta > 0.f)) {
            auto rotR = Eigen::AngleAxis<float>(delta.y(), leftVector);
            direction = rotR * direction;
        }

        pos = center - direction * radius;
        camera_.LookAt(center, pos, up);
    } else {
        auto rotY = Eigen::AngleAxis<float>(delta.x(), Eigen::Vector3f::UnitY());
        direction = rotY * direction;

        leftVector = direction.cross(Eigen::Vector3f::UnitY());
        leftVector.normalize();

        auto rotR = Eigen::AngleAxis<float>(delta.y(), leftVector);
        direction = rotR * direction;

        pos = center - direction * radius;
        camera_.LookAt(center, pos, leftVector.cross(direction).normalized());
    }
}

void CameraManipulator::UpdateCameraProjection() {
    const float aspect = viewportW_ / viewportH_;
    camera_.SetProjection(fov_, aspect, near_, far_, Camera::FovType::Vertical);
}

}
}
