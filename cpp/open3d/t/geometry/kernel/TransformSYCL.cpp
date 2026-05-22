// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Skip the CPU/CUDA main function definitions; only use the helper templates.
#define OPEN3D_SKIP_TRANSFORM_MAIN
#include "open3d/t/geometry/kernel/TransformImpl.h"
#undef OPEN3D_SKIP_TRANSFORM_MAIN

#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace transform {

void TransformPointsSYCL(const core::Tensor& transformation,
                         core::Tensor& points) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(
                    points.GetDevice());
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();
        const int64_t n = points.GetLength();
        queue.parallel_for(sycl::range<1>{(size_t)n},
                           [=](sycl::id<1> id) {
                               TransformPointsKernel(
                                       transformation_ptr,
                                       points_ptr + 3 * id[0]);
                           })
                .wait_and_throw();
    });
}

void TransformNormalsSYCL(const core::Tensor& transformation,
                          core::Tensor& normals) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(
                    normals.GetDevice());
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();
        const int64_t n = normals.GetLength();
        queue.parallel_for(sycl::range<1>{(size_t)n},
                           [=](sycl::id<1> id) {
                               TransformNormalsKernel(
                                       transformation_ptr,
                                       normals_ptr + 3 * id[0]);
                           })
                .wait_and_throw();
    });
}

void RotatePointsSYCL(const core::Tensor& R,
                      core::Tensor& points,
                      const core::Tensor& center) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(
                    points.GetDevice());
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();
        const scalar_t* center_ptr = center.GetDataPtr<scalar_t>();
        const int64_t n = points.GetLength();
        queue.parallel_for(sycl::range<1>{(size_t)n},
                           [=](sycl::id<1> id) {
                               RotatePointsKernel(R_ptr,
                                                  points_ptr + 3 * id[0],
                                                  center_ptr);
                           })
                .wait_and_throw();
    });
}

void RotateNormalsSYCL(const core::Tensor& R, core::Tensor& normals) {
    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(
                    normals.GetDevice());
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();
        const int64_t n = normals.GetLength();
        queue.parallel_for(
                     sycl::range<1>{(size_t)n},
                     [=](sycl::id<1> id) {
                         RotateNormalsKernel(R_ptr, normals_ptr + 3 * id[0]);
                     })
                .wait_and_throw();
    });
}

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
