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

#pragma once

#include "Camera.h"
#include <Eigen/Geometry>

namespace open3d {
namespace visualization {

class CameraManipulator {
public:
    CameraManipulator(Camera& camera, float viewportW, float viewportH);

    void SetViewport(float w, float h);
    void SetFov(float fov);
    void SetNearPlane(float near);
    void SetFarPlane(float far);

    float GetFov() const { return fov_; }
    float GetNearPlane() const { return near_; }
    float GetFarPlane() const { return far_; }

    void SetPosition(const Eigen::Vector3f& pos);
    Eigen::Vector3f GetPosition();

    void SetForwardVector(const Eigen::Vector3f& pos);
    Eigen::Vector3f GetForwardVector();

    Eigen::Vector3f GetLeftVector();
    Eigen::Vector3f GetUpVector();

    void SetCameraTransform(const Camera::Transform& transform);

    void LookAt(const Eigen::Vector3f& center,
                const Eigen::Vector3f& eye,
                const Eigen::Vector3f& up = {0, 1.f, 0.f});

    void Orbit(const Eigen::Vector3f& center,
               float radius,
               float deltaPhi,
               float deltaTheta,
               float rotationSpeed = M_PI / 2.f);

private:
    void UpdateCameraProjection();

    Camera& camera_;

    float viewportW_;
    float viewportH_;
    float fov_;
    float near_;
    float far_;
};

}  // namespace visualization
}  // namespace open3d
