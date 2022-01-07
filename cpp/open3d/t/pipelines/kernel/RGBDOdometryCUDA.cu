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
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryJacobianImpl.h"
#include "open3d/t/pipelines/kernel/Reduction6x6Impl.cuh"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

__global__ void ComputeOdometryResultPointToPlaneCUDAKernel(
        NDArrayIndexer source_vertex_indexer,
        NDArrayIndexer target_vertex_indexer,
        NDArrayIndexer target_normal_indexer,
        TransformIndexer ti,
        float* global_sum,
        int rows,
        int cols,
        const float depth_outlier_trunc,
        const float depth_huber_delta) {
    const int kBlockSize = 256;
    __shared__ float local_sum0[kBlockSize];
    __shared__ float local_sum1[kBlockSize];
    __shared__ float local_sum2[kBlockSize];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (y >= rows || x >= cols) return;

    float J[6] = {0}, reduction[21 + 6 + 2];
    float r = 0;
    bool valid = GetJacobianPointToPlane(
            x, y, depth_outlier_trunc, source_vertex_indexer,
            target_vertex_indexer, target_normal_indexer, ti, J, r);

    float d_huber = HuberDeriv(r, depth_huber_delta);
    float r_huber = HuberLoss(r, depth_huber_delta);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J[i] * J[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J[i] * d_huber;
    }
    reduction[offset++] = r_huber;
    reduction[offset++] = valid;

    // Sum reduction: JtJ(21) and Jtr(6)
    for (size_t i = 0; i < 27; i += 3) {
        local_sum0[tid] = valid ? reduction[i + 0] : 0;
        local_sum1[tid] = valid ? reduction[i + 1] : 0;
        local_sum2[tid] = valid ? reduction[i + 2] : 0;
        __syncthreads();

        BlockReduceSum<float, kBlockSize>(tid, local_sum0, local_sum1,
                                          local_sum2);

        if (tid == 0) {
            atomicAdd(&global_sum[i + 0], local_sum0[0]);
            atomicAdd(&global_sum[i + 1], local_sum1[0]);
            atomicAdd(&global_sum[i + 2], local_sum2[0]);
        }
        __syncthreads();
    }

    // Sum reduction: residual(1) and inlier(1)
    {
        local_sum0[tid] = valid ? reduction[27] : 0;
        local_sum1[tid] = valid ? reduction[28] : 0;
        __syncthreads();

        BlockReduceSum<float, kBlockSize>(tid, local_sum0, local_sum1);
        if (tid == 0) {
            atomicAdd(&global_sum[27], local_sum0[0]);
            atomicAdd(&global_sum[28], local_sum1[0]);
        }
        __syncthreads();
    }
}

void ComputeOdometryResultPointToPlaneCUDA(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& target_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta) {
    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);
    NDArrayIndexer target_normal_indexer(target_normal_map, 2);

    core::Device device = source_vertex_map.GetDevice();

    core::Tensor trans = init_source_to_target;
    TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum = core::Tensor::Zeros({29}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputeOdometryResultPointToPlaneCUDAKernel<<<blocks, threads, 0,
                                                  core::cuda::GetStream()>>>(
            source_vertex_indexer, target_vertex_indexer, target_normal_indexer,
            ti, global_sum_ptr, rows, cols, depth_outlier_trunc,
            depth_huber_delta);
    core::cuda::Synchronize();
    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

__global__ void ComputeOdometryResultIntensityCUDAKernel(
        NDArrayIndexer source_depth_indexer,
        NDArrayIndexer target_depth_indexer,
        NDArrayIndexer source_intensity_indexer,
        NDArrayIndexer target_intensity_indexer,
        NDArrayIndexer target_intensity_dx_indexer,
        NDArrayIndexer target_intensity_dy_indexer,
        NDArrayIndexer source_vertex_indexer,
        TransformIndexer ti,
        float* global_sum,
        int rows,
        int cols,
        const float depth_outlier_trunc,
        const float intensity_huber_delta) {
    const int kBlockSize = 256;
    __shared__ float local_sum0[kBlockSize];
    __shared__ float local_sum1[kBlockSize];
    __shared__ float local_sum2[kBlockSize];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (y >= rows || x >= cols) return;

    float J[6] = {0}, reduction[21 + 6 + 2];
    float r = 0;
    bool valid = GetJacobianIntensity(
            x, y, depth_outlier_trunc, source_depth_indexer,
            target_depth_indexer, source_intensity_indexer,
            target_intensity_indexer, target_intensity_dx_indexer,
            target_intensity_dy_indexer, source_vertex_indexer, ti, J, r);

    float d_huber = HuberDeriv(r, intensity_huber_delta);
    float r_huber = HuberLoss(r, intensity_huber_delta);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J[i] * J[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J[i] * HuberDeriv(r, intensity_huber_delta);
    }
    reduction[offset++] = HuberLoss(r, intensity_huber_delta);
    reduction[offset++] = valid;

    ReduceSum6x6LinearSystem<float, kBlockSize>(tid, valid, reduction,
                                                local_sum0, local_sum1,
                                                local_sum2, global_sum);
}

void ComputeOdometryResultIntensityCUDA(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float intensity_huber_delta) {
    NDArrayIndexer source_depth_indexer(source_depth, 2);
    NDArrayIndexer target_depth_indexer(target_depth, 2);

    NDArrayIndexer source_intensity_indexer(source_intensity, 2);
    NDArrayIndexer target_intensity_indexer(target_intensity, 2);

    NDArrayIndexer target_intensity_dx_indexer(target_intensity_dx, 2);
    NDArrayIndexer target_intensity_dy_indexer(target_intensity_dy, 2);

    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);

    core::Device device = source_vertex_map.GetDevice();
    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum = core::Tensor::Zeros({29}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputeOdometryResultIntensityCUDAKernel<<<blocks, threads, 0,
                                               core::cuda::GetStream()>>>(
            source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, global_sum_ptr, rows, cols,
            depth_outlier_trunc, intensity_huber_delta);
    core::cuda::Synchronize();
    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

__global__ void ComputeOdometryResultHybridCUDAKernel(
        NDArrayIndexer source_depth_indexer,
        NDArrayIndexer target_depth_indexer,
        NDArrayIndexer source_intensity_indexer,
        NDArrayIndexer target_intensity_indexer,
        NDArrayIndexer target_depth_dx_indexer,
        NDArrayIndexer target_depth_dy_indexer,
        NDArrayIndexer target_intensity_dx_indexer,
        NDArrayIndexer target_intensity_dy_indexer,
        NDArrayIndexer source_vertex_indexer,
        TransformIndexer ti,
        float* global_sum,
        int rows,
        int cols,
        const float depth_outlier_trunc,
        const float depth_huber_delta,
        const float intensity_huber_delta) {
    const int kBlockSize = 256;
    __shared__ float local_sum0[kBlockSize];
    __shared__ float local_sum1[kBlockSize];
    __shared__ float local_sum2[kBlockSize];

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (y >= rows || x >= cols) return;

    float J_I[6] = {0}, J_D[6] = {0}, reduction[21 + 6 + 2];
    float r_I = 0, r_D = 0;
    bool valid = GetJacobianHybrid(
            x, y, depth_outlier_trunc, source_depth_indexer,
            target_depth_indexer, source_intensity_indexer,
            target_intensity_indexer, target_depth_dx_indexer,
            target_depth_dy_indexer, target_intensity_dx_indexer,
            target_intensity_dy_indexer, source_vertex_indexer, ti, J_I, J_D,
            r_I, r_D);

    float d_huber_D = HuberDeriv(r_D, depth_huber_delta);
    float d_huber_I = HuberDeriv(r_I, intensity_huber_delta);

    float r_huber_D = HuberLoss(r_D, depth_huber_delta);
    float r_huber_I = HuberLoss(r_I, intensity_huber_delta);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J_I[i] * J_I[j] + J_D[i] * J_D[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J_I[i] * d_huber_I + J_D[i] * d_huber_D;
    }
    reduction[offset++] = r_huber_D + r_huber_I;
    reduction[offset++] = valid;

    ReduceSum6x6LinearSystem<float, kBlockSize>(tid, valid, reduction,
                                                local_sum0, local_sum1,
                                                local_sum2, global_sum);
}

void ComputeOdometryResultHybridCUDA(const core::Tensor& source_depth,
                                     const core::Tensor& target_depth,
                                     const core::Tensor& source_intensity,
                                     const core::Tensor& target_intensity,
                                     const core::Tensor& target_depth_dx,
                                     const core::Tensor& target_depth_dy,
                                     const core::Tensor& target_intensity_dx,
                                     const core::Tensor& target_intensity_dy,
                                     const core::Tensor& source_vertex_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     core::Tensor& delta,
                                     float& inlier_residual,
                                     int& inlier_count,
                                     const float depth_outlier_trunc,
                                     const float depth_huber_delta,
                                     const float intensity_huber_delta) {
    NDArrayIndexer source_depth_indexer(source_depth, 2);
    NDArrayIndexer target_depth_indexer(target_depth, 2);

    NDArrayIndexer source_intensity_indexer(source_intensity, 2);
    NDArrayIndexer target_intensity_indexer(target_intensity, 2);

    NDArrayIndexer target_depth_dx_indexer(target_depth_dx, 2);
    NDArrayIndexer target_depth_dy_indexer(target_depth_dy, 2);
    NDArrayIndexer target_intensity_dx_indexer(target_intensity_dx, 2);
    NDArrayIndexer target_intensity_dy_indexer(target_intensity_dy, 2);

    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);

    core::Device device = source_vertex_map.GetDevice();
    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum = core::Tensor::Zeros({29}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputeOdometryResultHybridCUDAKernel<<<blocks, threads, 0,
                                            core::cuda::GetStream()>>>(
            source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_depth_dx_indexer, target_depth_dy_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, global_sum_ptr, rows, cols,
            depth_outlier_trunc, depth_huber_delta, intensity_huber_delta);
    core::cuda::Synchronize();
    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
