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

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

const int kThread1DUnit = 256;

__global__ void ComputePosePointToPlaneCUDAKernel(
        const float *source_points_ptr,
        const float *target_points_ptr,
        const float *target_normals_ptr,
        const int64_t *correspondences_first,
        const int64_t *correspondences_second,
        const int n,
        float *global_sum) {
    __shared__ float local_sum0[kThread1DUnit];
    __shared__ float local_sum1[kThread1DUnit];
    __shared__ float local_sum2[kThread1DUnit];

    const int tid = threadIdx.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    const int workload_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (workload_idx >= n) return;

    float J[6] = {0}, reduction[21 + 6 + 2];
    float r = 0;

    bool valid = GetJacobianPointToPlane(workload_idx, source_points_ptr,
                                         target_points_ptr, target_normals_ptr,
                                         correspondences_first,
                                         correspondences_second, J, r);

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

    ReduceSum6x6LinearSystem<float, kThread1DUnit>(tid, valid, reduction,
                                                   local_sum0, local_sum1,
                                                   local_sum2, global_sum);
}

void ComputePosePointToPlaneCUDA(const float *source_points_ptr,
                                 const float *target_points_ptr,
                                 const float *target_normals_ptr,
                                 const int64_t *correspondences_first,
                                 const int64_t *correspondences_second,
                                 const int n,
                                 core::Tensor &pose,
                                 const core::Dtype &dtype,
                                 const core::Device &device) {
    core::Tensor global_sum =
            core::Tensor::Zeros({29}, core::Dtype::Float32, device);
    float *global_sum_ptr = global_sum.GetDataPtr<float>();

    const dim3 blocks((n + kThread1DUnit - 1) / kThread1DUnit);
    const dim3 threads(kThread1DUnit);

    ComputePosePointToPlaneCUDAKernel<<<blocks, threads>>>(
            source_points_ptr, target_points_ptr, target_normals_ptr,
            correspondences_first, correspondences_second, n, global_sum_ptr);

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());

    // TODO (@rishabh), residual will be used for adding robust kernel support.
    float residual;
    int inlier_count;
    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
