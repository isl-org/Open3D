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

#include "open3d/visualization/rendering/filament/FilamentCamera.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068)
#endif  // _MSC_VER

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <math/mat4.h>  // necessary for mat4f

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

namespace open3d {
namespace visualization {
namespace rendering {

FilamentCamera::FilamentCamera(filament::Engine& engine) : engine_(engine) {
    camera_ = engine_.createCamera();
}

FilamentCamera::~FilamentCamera() { engine_.destroy(camera_); }

void FilamentCamera::SetProjection(
        double fov, double aspect, double near, double far, FovType fov_type) {
    if (aspect > 0.0) {
        filament::Camera::Fov dir = (fov_type == FovType::Horizontal)
                                            ? filament::Camera::Fov::HORIZONTAL
                                            : filament::Camera::Fov::VERTICAL;

        camera_->setProjection(fov, aspect, near, far, dir);
        fov_ = fov;
        fov_type_ = fov_type;
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
    // technically orthographic projection is lim(fov->0) as dist->inf,
    // but it also serves as an obviously wrong value if you call
    // GetFieldOfView() after setting an orthographic projection
    fov_ = 0.0;
}

double FilamentCamera::GetNear() const { return camera_->getNear(); }

double FilamentCamera::GetFar() const { return camera_->getCullingFar(); }

double FilamentCamera::GetFieldOfView() const { return fov_; }

Camera::FovType FilamentCamera::GetFieldOfViewType() const { return fov_type_; }

void FilamentCamera::SetModelMatrix(const Transform& view) {
    using namespace filament::math;

    auto e_matrix = view.matrix();
    mat4f ftransform(mat4f::row_major_init{
            e_matrix(0, 0), e_matrix(0, 1), e_matrix(0, 2), e_matrix(0, 3),
            e_matrix(1, 0), e_matrix(1, 1), e_matrix(1, 2), e_matrix(1, 3),
            e_matrix(2, 0), e_matrix(2, 1), e_matrix(2, 2), e_matrix(2, 3),
            e_matrix(3, 0), e_matrix(3, 1), e_matrix(3, 2), e_matrix(3, 3)});

    camera_->setModelMatrix(ftransform);
}

void FilamentCamera::SetModelMatrix(const Eigen::Vector3f& forward,
                                    const Eigen::Vector3f& left,
                                    const Eigen::Vector3f& up) {
    using namespace filament;

    math::mat4f ftransform = camera_->getModelMatrix();
    ftransform[0].xyz = math::float3(left.x(), left.y(), left.z());
    ftransform[1].xyz = math::float3(up.x(), up.y(), up.z());
    ftransform[2].xyz = math::float3(forward.x(), forward.y(), forward.z());

    camera_->setModelMatrix(ftransform);
}

void FilamentCamera::LookAt(const Eigen::Vector3f& center,
                            const Eigen::Vector3f& eye,
                            const Eigen::Vector3f& up) {
    camera_->lookAt({eye.x(), eye.y(), eye.z()},
                    {center.x(), center.y(), center.z()},
                    {up.x(), up.y(), up.z()});
}

Eigen::Vector3f FilamentCamera::GetPosition() const {
    auto cam_pos = camera_->getPosition();
    return {cam_pos.x, cam_pos.y, cam_pos.z};
}

Eigen::Vector3f FilamentCamera::GetForwardVector() const {
    auto forward = camera_->getForwardVector();
    return {forward.x, forward.y, forward.z};
}

Eigen::Vector3f FilamentCamera::GetLeftVector() const {
    auto left = camera_->getLeftVector();
    return {left.x, left.y, left.z};
}

Eigen::Vector3f FilamentCamera::GetUpVector() const {
    auto up = camera_->getUpVector();
    return {up.x, up.y, up.z};
}

FilamentCamera::Transform FilamentCamera::GetModelMatrix() const {
    auto ftransform = camera_->getModelMatrix();

    Transform::MatrixType matrix;

    matrix << ftransform(0, 0), ftransform(0, 1), ftransform(0, 2),
            ftransform(0, 3), ftransform(1, 0), ftransform(1, 1),
            ftransform(1, 2), ftransform(1, 3), ftransform(2, 0),
            ftransform(2, 1), ftransform(2, 2), ftransform(2, 3),
            ftransform(3, 0), ftransform(3, 1), ftransform(3, 2),
            ftransform(3, 3);

    return Transform(matrix);
}

FilamentCamera::Transform FilamentCamera::GetViewMatrix() const {
    auto ftransform = camera_->getViewMatrix();  // returns mat4 (not mat4f)

    Transform::MatrixType matrix;

    matrix << float(ftransform(0, 0)), float(ftransform(0, 1)),
            float(ftransform(0, 2)), float(ftransform(0, 3)),
            float(ftransform(1, 0)), float(ftransform(1, 1)),
            float(ftransform(1, 2)), float(ftransform(1, 3)),
            float(ftransform(2, 0)), float(ftransform(2, 1)),
            float(ftransform(2, 2)), float(ftransform(2, 3)),
            float(ftransform(3, 0)), float(ftransform(3, 1)),
            float(ftransform(3, 2)), float(ftransform(3, 3));

    return Transform(matrix);
}

FilamentCamera::Transform FilamentCamera::GetProjectionMatrix() const {
    auto ftransform = camera_->getProjectionMatrix();  // mat4 (not mat4f)

    Transform::MatrixType matrix;

    matrix << float(ftransform(0, 0)), float(ftransform(0, 1)),
            float(ftransform(0, 2)), float(ftransform(0, 3)),
            float(ftransform(1, 0)), float(ftransform(1, 1)),
            float(ftransform(1, 2)), float(ftransform(1, 3)),
            float(ftransform(2, 0)), float(ftransform(2, 1)),
            float(ftransform(2, 2)), float(ftransform(2, 3)),
            float(ftransform(3, 0)), float(ftransform(3, 1)),
            float(ftransform(3, 2)), float(ftransform(3, 3));

    return Transform(matrix);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
