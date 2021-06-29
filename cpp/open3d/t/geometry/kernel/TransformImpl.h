// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace transform {

void TransformPointsCPU(const core::Tensor& transformation,
                        core::Tensor& points);

void TransformNormalsCPU(const core::Tensor& transformation,
                         core::Tensor& normals);

void RotatePointsCPU(const core::Tensor& R,
                     core::Tensor& points,
                     const core::Tensor& center);

void RotateNormalsCPU(const core::Tensor& R, core::Tensor& normals);

#ifdef BUILD_CUDA_MODULE
void TransformPointsCUDA(const core::Tensor& transformation,
                         core::Tensor& points);

void TransformNormalsCUDA(const core::Tensor& transformation,
                          core::Tensor& normals);

void RotatePointsCUDA(const core::Tensor& R,
                      core::Tensor& points,
                      const core::Tensor& center);

void RotateNormalsCUDA(const core::Tensor& R, core::Tensor& normals);
#endif

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

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
