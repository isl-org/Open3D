// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include <cuda.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

const int kThread1DUnit = 256;

template <typename scalar_t, class funct_t>
__global__ void ComputePosePointToPlaneCUDAKernel(
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondences_second,
        const int n,
        scalar_t *global_sum,
        funct_t op) {
    __shared__ scalar_t local_sum0[kThread1DUnit];
    __shared__ scalar_t local_sum1[kThread1DUnit];
    __shared__ scalar_t local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    scalar_t J[6] = {0}, reduction[29] = {0};
    scalar_t r = 0;

    bool valid = GetJacobianPointToPlane<scalar_t>(
            workload_idx, source_points_ptr, target_points_ptr,
            target_normals_ptr, correspondences_second, J, r);

    scalar_t w = op(r);

    printf(" residual: %lf, weight: %lf", (double)r, (double)w);

    if (valid) {
        // Dump J, r into JtJ and Jtr
        int offset = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j <= i; ++j) {
                reduction[offset++] = J[i] * J[j];
            }
        }
        for (int i = 0; i < 6; ++i) {
            reduction[offset++] = J[i] * r;
        }
        reduction[offset++] = r * r;
        reduction[offset++] = valid;
    }

    ReduceSum6x6LinearSystem<scalar_t, kThread1DUnit>(tid, valid, reduction,
                                                   local_sum0, local_sum1,
                                                   local_sum2, global_sum);
}

void ComputePosePointToPlaneCUDA(const core::Tensor &source_points,
                                 const core::Tensor &target_points,
                                 const core::Tensor &target_normals,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &pose,
                                 float &residual,
                                 int &inlier_count,
                                 const core::Dtype &dtype,
                                 const core::Device &device) {
    int n = source_points.GetLength();


    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {

        auto robust_kernel = [=] OPEN3D_HOST_DEVICE(scalar_t r) -> scalar_t {
            return 2 * r;
        };

        core::Tensor global_sum = core::Tensor::Zeros({29}, dtype, device);
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
        const dim3 threads(kThread1DUnit);

        ComputePosePointToPlaneCUDAKernel<<<blocks, threads>>>(
                source_points.GetDataPtr<scalar_t>(),
                target_points.GetDataPtr<scalar_t>(),
                target_normals.GetDataPtr<scalar_t>(),
                correspondence_indices.GetDataPtr<int64_t>(), n, global_sum_ptr,
                robust_kernel);

        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());

        DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
    });
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
