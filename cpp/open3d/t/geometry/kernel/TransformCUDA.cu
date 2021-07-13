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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/Transform.h"
#include "open3d/t/geometry/kernel/TransformImpl.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace transform {

void TransformPointsCUDA(const core::Tensor& transformation,
                         core::Tensor& points) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();

        core::kernel::cuda_launcher::ParallelFor(
                points.GetLength(), [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    TransformPointsKernel(transformation_ptr,
                                          points_ptr + 3 * workload_idx);
                });
    });

    return;
}

void TransformNormalsCUDA(const core::Tensor& transformation,
                          core::Tensor& normals) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* transformation_ptr =
                transformation.GetDataPtr<scalar_t>();

        core::kernel::cuda_launcher::ParallelFor(
                normals.GetLength(), [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    TransformNormalsKernel(transformation_ptr,
                                           normals_ptr + 3 * workload_idx);
                });
    });

    return;
}

void RotatePointsCUDA(const core::Tensor& R,
                      core::Tensor& points,
                      const core::Tensor& center) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(points.GetDtype(), [&]() {
        scalar_t* points_ptr = points.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();
        const scalar_t* center_ptr = center.GetDataPtr<scalar_t>();

        core::kernel::cuda_launcher::ParallelFor(
                points.GetLength(), [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    RotatePointsKernel(R_ptr, points_ptr + 3 * workload_idx,
                                       center_ptr);
                });
    });

    return;
}

void RotateNormalsCUDA(const core::Tensor& R, core::Tensor& normals) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(normals.GetDtype(), [&]() {
        scalar_t* normals_ptr = normals.GetDataPtr<scalar_t>();
        const scalar_t* R_ptr = R.GetDataPtr<scalar_t>();

        core::kernel::cuda_launcher::ParallelFor(
                normals.GetLength(), [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    RotateNormalsKernel(R_ptr, normals_ptr + 3 * workload_idx);
                });
    });

    return;
}

}  // namespace transform
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
