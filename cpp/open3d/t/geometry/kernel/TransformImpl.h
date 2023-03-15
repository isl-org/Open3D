// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace transform {

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void TransformPointsKernel(
        const scalar_t* transformation_ptr, scalar_t* points_ptr) {
    scalar_t x[4] = {transformation_ptr[0] * points_ptr[0] +
                             transformation_ptr[1] * points_ptr[1] +
                             transformation_ptr[2] * points_ptr[2] +
                             transformation_ptr[3],
                     transformation_ptr[4] * points_ptr[0] +
                             transformation_ptr[5] * points_ptr[1] +
                             transformation_ptr[6] * points_ptr[2] +
                             transformation_ptr[7],
                     transformation_ptr[8] * points_ptr[0] +
                             transformation_ptr[9] * points_ptr[1] +
                             transformation_ptr[10] * points_ptr[2] +
                             transformation_ptr[11],
                     transformation_ptr[12] * points_ptr[0] +
                             transformation_ptr[13] * points_ptr[1] +
                             transformation_ptr[14] * points_ptr[2] +
                             transformation_ptr[15]};

    points_ptr[0] = x[0] / x[3];
    points_ptr[1] = x[1] / x[3];
    points_ptr[2] = x[2] / x[3];
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void TransformNormalsKernel(
        const scalar_t* transformation_ptr, scalar_t* normals_ptr) {
    scalar_t x[3] = {transformation_ptr[0] * normals_ptr[0] +
                             transformation_ptr[1] * normals_ptr[1] +
                             transformation_ptr[2] * normals_ptr[2],
                     transformation_ptr[4] * normals_ptr[0] +
                             transformation_ptr[5] * normals_ptr[1] +
                             transformation_ptr[6] * normals_ptr[2],
                     transformation_ptr[8] * normals_ptr[0] +
                             transformation_ptr[9] * normals_ptr[1] +
                             transformation_ptr[10] * normals_ptr[2]};

    normals_ptr[0] = x[0];
    normals_ptr[1] = x[1];
    normals_ptr[2] = x[2];
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void RotatePointsKernel(
        const scalar_t* R_ptr, scalar_t* points_ptr, const scalar_t* center) {
    scalar_t x[3] = {points_ptr[0] - center[0], points_ptr[1] - center[1],
                     points_ptr[2] - center[2]};

    points_ptr[0] =
            R_ptr[0] * x[0] + R_ptr[1] * x[1] + R_ptr[2] * x[2] + center[0];
    points_ptr[1] =
            R_ptr[3] * x[0] + R_ptr[4] * x[1] + R_ptr[5] * x[2] + center[1];
    points_ptr[2] =
            R_ptr[6] * x[0] + R_ptr[7] * x[1] + R_ptr[8] * x[2] + center[2];
}

template <typename scalar_t>
OPEN3D_HOST_DEVICE OPEN3D_FORCE_INLINE void RotateNormalsKernel(
        const scalar_t* R_ptr, scalar_t* normals_ptr) {
    scalar_t x[3] = {R_ptr[0] * normals_ptr[0] + R_ptr[1] * normals_ptr[1] +
                             R_ptr[2] * normals_ptr[2],
                     R_ptr[3] * normals_ptr[0] + R_ptr[4] * normals_ptr[1] +
                             R_ptr[5] * normals_ptr[2],
                     R_ptr[6] * normals_ptr[0] + R_ptr[7] * normals_ptr[1] +
                             R_ptr[8] * normals_ptr[2]};

    normals_ptr[0] = x[0];
    normals_ptr[1] = x[1];
    normals_ptr[2] = x[2];
}

#ifdef __CUDACC__
void TransformPointsCUDA
#else
void TransformPointsCPU
#endif
        (const core::Tensor& transformation, core::Tensor& points) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();

        core::ParallelFor(transformation.GetDevice(), points.GetLength(),
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              TransformPointsKernel(
                                      transformation_ptr,
                                      points_ptr + 3 * workload_idx);
                          });
    });
}

#ifdef __CUDACC__
void TransformNormalsCUDA
#else
void TransformNormalsCPU
#endif
        (const core::Tensor& transformation, core::Tensor& normals) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();

        core::ParallelFor(transformation.GetDevice(), normals.GetLength(),
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              TransformNormalsKernel(
                                      transformation_ptr,
                                      normals_ptr + 3 * workload_idx);
                          });
    });
}

#ifdef __CUDACC__
void RotatePointsCUDA
#else
void RotatePointsCPU
#endif
        (const core::Tensor& R,
         core::Tensor& points,
         const core::Tensor& center) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();
        const scalar_t* center_ptr = center.GetDataPtr<scalar_t>();

        core::ParallelFor(R.GetDevice(), points.GetLength(),
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              RotatePointsKernel(R_ptr,
                                                 points_ptr + 3 * workload_idx,
                                                 center_ptr);
                          });
    });
}

#ifdef __CUDACC__
void RotateNormalsCUDA
#else
void RotateNormalsCPU
#endif
        (const core::Tensor& R, core::Tensor& normals) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();

        core::ParallelFor(R.GetDevice(), normals.GetLength(),
                          [=] OPEN3D_DEVICE(int64_t workload_idx) {
                              RotateNormalsKernel(
                                      R_ptr, normals_ptr + 3 * workload_idx);
                          });
    });
}

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
