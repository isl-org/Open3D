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

#include "FilamentCamera.h"

#include <filament/Camera.h>
#include <filament/Engine.h>

#include <iostream>

namespace open3d {
namespace visualization {

FilamentCamera::FilamentCamera(filament::Engine& aEngine) : engine_(aEngine) {
    camera_ = engine_.createCamera();
}

FilamentCamera::~FilamentCamera() { engine_.destroy(camera_); }

void FilamentCamera::SetProjection(
        double fov, double aspect, double near, double far, FovType fovType) {
    if (aspect > 0.0) {
        filament::Camera::Fov dir = (fovType == FovType::Horizontal)
                                            ? filament::Camera::Fov::HORIZONTAL
                                            : filament::Camera::Fov::VERTICAL;

        camera_->setProjection(fov, aspect, near, far, dir);
    }
}

void FilamentCamera::SetProjection(Projection projection,
                                   double left,
                                   double right,
                                   double bottom,
                                   double top,
                                   double near,
                                   double far) {
    filament::Camera::Projection proj =
            (projection == Projection::Ortho)
                    ? filament::Camera::Projection::ORTHO
                    : filament::Camera::Projection::PERSPECTIVE;

    camera_->setProjection(proj, left, right, bottom, top, near, far);
}

void FilamentCamera::SetModelMatrix(const Transform& view) {
    using namespace filament::math;

    auto eMatrix = view.matrix();
    mat4f fTransform(mat4f::row_major_init{
            eMatrix(0, 0), eMatrix(0, 1), eMatrix(0, 2), eMatrix(0, 3),
            eMatrix(1, 0), eMatrix(1, 1), eMatrix(1, 2), eMatrix(1, 3),
            eMatrix(2, 0), eMatrix(2, 1), eMatrix(2, 2), eMatrix(2, 3),
            eMatrix(3, 0), eMatrix(3, 1), eMatrix(3, 2), eMatrix(3, 3)});

    camera_->setModelMatrix(fTransform);
}

void FilamentCamera::SetModelMatrix(const Eigen::Vector3f& forward,
                                    const Eigen::Vector3f& left,
                                    const Eigen::Vector3f& up) {
    using namespace filament;

    math::mat4f fTransform = camera_->getModelMatrix();
    fTransform[0].xyz = math::float3(left.x(),left.y(),left.z());
    fTransform[1].xyz = math::float3(up.x(),up.y(),up.z());
    fTransform[2].xyz = math::float3(forward.x(),forward.y(),forward.z());

    camera_->setModelMatrix(fTransform);
}

void FilamentCamera::LookAt(const Eigen::Vector3f& center,
                            const Eigen::Vector3f& eye,
                            const Eigen::Vector3f& up) {
    camera_->lookAt({eye.x(), eye.y(), eye.z()},
                    {center.x(), center.y(), center.z()},
                    {up.x(), up.y(), up.z()});
}

Eigen::Vector3f FilamentCamera::GetPosition() {
    auto camPos = camera_->getPosition();
    return {camPos.x, camPos.y, camPos.z};
}

Eigen::Vector3f FilamentCamera::GetForwardVector() {
    auto forward = camera_->getForwardVector();
    return {forward.x, forward.y, forward.z};
}

Eigen::Vector3f FilamentCamera::GetLeftVector() {
    auto left = camera_->getLeftVector();
    return {left.x, left.y, left.z};
}

Eigen::Vector3f FilamentCamera::GetUpVector() {
    auto up = camera_->getUpVector();
    return {up.x, up.y, up.z};
}

FilamentCamera::Transform FilamentCamera::GetModelMatrix() {
    auto fTransform = camera_->getModelMatrix();

    Transform::MatrixType matrix;

    matrix << fTransform(0,0), fTransform(0,1), fTransform(0,2), fTransform(0,3),
            fTransform(1,0), fTransform(1,1), fTransform(1,2), fTransform(1,3),
            fTransform(2,0), fTransform(2,1), fTransform(2,2), fTransform(2,3),
            fTransform(3,0), fTransform(3,1), fTransform(3,2), fTransform(3,3);

    return Transform(matrix);
}

}
}