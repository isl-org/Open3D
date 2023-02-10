// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/Transform.h"

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/TensorCheck.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace transform {

void TransformPoints(const core::Tensor& transformation, core::Tensor& points) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorShape(transformation, {4, 4});

    core::Tensor points_contiguous = points.Contiguous();
    core::Tensor transformation_contiguous =
            transformation.To(points.GetDevice(), points.GetDtype())
                    .Contiguous();

    if (points.IsCPU()) {
        TransformPointsCPU(transformation_contiguous, points_contiguous);
    } else if (points.IsCUDA()) {
        CUDA_CALL(TransformPointsCUDA, transformation_contiguous,
                  points_contiguous);
    } else {
        utility::LogError("Unimplemented device");
    }

    points = points_contiguous;
}

void TransformNormals(const core::Tensor& transformation,
                      core::Tensor& normals) {
    core::AssertTensorShape(normals, {utility::nullopt, 3});
    core::AssertTensorShape(transformation, {4, 4});

    core::Tensor normals_contiguous = normals.Contiguous();
    core::Tensor transformation_contiguous =
            transformation.To(normals.GetDevice(), normals.GetDtype())
                    .Contiguous();

    if (normals.IsCPU()) {
        TransformNormalsCPU(transformation_contiguous, normals_contiguous);
    } else if (normals.IsCUDA()) {
        CUDA_CALL(TransformNormalsCUDA, transformation_contiguous,
                  normals_contiguous);
    } else {
        utility::LogError("Unimplemented device");
    }

    normals = normals_contiguous;
}

void RotatePoints(const core::Tensor& R,
                  core::Tensor& points,
                  const core::Tensor& center) {
    core::AssertTensorShape(points, {utility::nullopt, 3});
    core::AssertTensorShape(R, {3, 3});
    core::AssertTensorShape(center, {3});

    core::Tensor points_contiguous = points.Contiguous();
    core::Tensor R_contiguous =
            R.To(points.GetDevice(), points.GetDtype()).Contiguous();
    core::Tensor center_contiguous =
            center.To(points.GetDevice(), points.GetDtype()).Contiguous();

    if (points.IsCPU()) {
        RotatePointsCPU(R_contiguous, points_contiguous, center_contiguous);
    } else if (points.IsCUDA()) {
        CUDA_CALL(RotatePointsCUDA, R_contiguous, points_contiguous,
                  center_contiguous);
    } else {
        utility::LogError("Unimplemented device");
    }

    points = points_contiguous;
}

void RotateNormals(const core::Tensor& R, core::Tensor& normals) {
    core::AssertTensorShape(normals, {utility::nullopt, 3});
    core::AssertTensorShape(R, {3, 3});

    core::Tensor normals_contiguous = normals.Contiguous();
    core::Tensor R_contiguous =
            R.To(normals.GetDevice(), normals.GetDtype()).Contiguous();

    if (normals.IsCPU()) {
        RotateNormalsCPU(R_contiguous, normals_contiguous);
    } else if (normals.IsCUDA()) {
        CUDA_CALL(RotateNormalsCUDA, R_contiguous, normals_contiguous);
    } else {
        utility::LogError("Unimplemented device");
    }

    normals = normals_contiguous;
}

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
