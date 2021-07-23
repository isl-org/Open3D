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

#include <cuda.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/ComputeTransformImpl.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

const int kThread1DUnit = 256;

template <typename scalar_t, typename func_t>
__global__ void ComputePosePointToPlaneKernelCUDA(
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        const int n,
        scalar_t *global_sum,
        func_t GetWeightFromRobustKernel) {
    __shared__ scalar_t local_sum0[kThread1DUnit];
    __shared__ scalar_t local_sum1[kThread1DUnit];
    __shared__ scalar_t local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    scalar_t J_ij[6] = {0}, reduction[29] = {0};
    scalar_t r = 0;

    bool valid = GetJacobianPointToPlane<scalar_t>(
            workload_idx, source_points_ptr, target_points_ptr,
            target_normals_ptr, correspondence_indices, J_ij, r);

    scalar_t w = GetWeightFromRobustKernel(r);

    if (valid) {
        // Dump J, r into JtJ and Jtr
        int i = 0;
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k <= j; ++k) {
                reduction[i] += J_ij[j] * w * J_ij[k];
                ++i;
            }
            reduction[21 + j] += J_ij[j] * w * r;
        }
        reduction[27] += r;
        reduction[28] += 1;
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
                                 const core::Device &device,
                                 const registration::RobustKernel &kernel) {
    int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({29}, dtype, device);
    const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
    const dim3 threads(kThread1DUnit);

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    ComputePosePointToPlaneKernelCUDA<<<
                            blocks, threads, 0, core::cuda::GetStream()>>>(
                            source_points.GetDataPtr<scalar_t>(),
                            target_points.GetDataPtr<scalar_t>(),
                            target_normals.GetDataPtr<scalar_t>(),
                            correspondence_indices.GetDataPtr<int64_t>(), n,
                            global_sum_ptr, GetWeightFromRobustKernel);
                });
    });

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
