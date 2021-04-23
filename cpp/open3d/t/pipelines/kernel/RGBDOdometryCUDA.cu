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
#include "open3d/core/CoreUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
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

#define sign(x) (x < 0 ? -1 : (x > 0 ? 1 : 0))

__global__ void ComputePosePointToPlaneCUDAKernel(
        NDArrayIndexer source_vertex_indexer,
        NDArrayIndexer target_vertex_indexer,
        NDArrayIndexer target_normal_indexer,
        TransformIndexer ti,
        float* global_sum,
        int rows,
        int cols,
        float depth_diff) {
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
            x, y, depth_diff, source_vertex_indexer, target_vertex_indexer,
            target_normal_indexer, ti, J, r);

    // Dump J, r into JtJ and Jtr
    const float h = 0.05;
    float huber_r = abs(r) < h ? 0.5 * r * r : h * abs(r) - 0.5 * h * h;
    float deriv_r = abs(r) < h ? r : h * sign(r);

    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J[i] * J[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J[i] * deriv_r;
    }

    reduction[offset++] = huber_r;
    reduction[offset++] = valid;

    ReduceSum6x6LinearSystem<float, kBlockSize>(tid, valid, reduction,
                                                local_sum0, local_sum1,
                                                local_sum2, global_sum);
}

void ComputePosePointToPlaneCUDA(const core::Tensor& source_vertex_map,
                                 const core::Tensor& target_vertex_map,
                                 const core::Tensor& target_normal_map,
                                 const core::Tensor& intrinsics,
                                 const core::Tensor& init_source_to_target,
                                 core::Tensor& delta,
                                 core::Tensor& residual,
                                 float depth_diff) {
    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);
    NDArrayIndexer target_normal_indexer(target_normal_map, 2);

    core::Device device = source_vertex_map.GetDevice();

    core::Tensor trans = init_source_to_target;
    TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum =
            core::Tensor::Zeros({29}, core::Dtype::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputePosePointToPlaneCUDAKernel<<<blocks, threads>>>(
            source_vertex_indexer, target_vertex_indexer, target_normal_indexer,
            ti, global_sum_ptr, rows, cols, depth_diff);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    DecodeAndSolve6x6(global_sum, delta, residual);
}

__global__ void ComputePoseIntensityCUDAKernel(
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
        float depth_diff) {
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
            x, y, depth_diff, source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, J, r);

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

    ReduceSum6x6LinearSystem<float, kBlockSize>(tid, valid, reduction,
                                                local_sum0, local_sum1,
                                                local_sum2, global_sum);
}

void ComputePoseIntensityCUDA(const core::Tensor& source_depth,
                              const core::Tensor& target_depth,
                              const core::Tensor& source_intensity,
                              const core::Tensor& target_intensity,
                              const core::Tensor& target_intensity_dx,
                              const core::Tensor& target_intensity_dy,
                              const core::Tensor& source_vertex_map,
                              const core::Tensor& intrinsics,
                              const core::Tensor& init_source_to_target,
                              core::Tensor& delta,
                              core::Tensor& residual,
                              float depth_diff) {
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

    core::Tensor global_sum =
            core::Tensor::Zeros({29}, core::Dtype::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputePoseIntensityCUDAKernel<<<blocks, threads>>>(
            source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, global_sum_ptr, rows, cols, depth_diff);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    DecodeAndSolve6x6(global_sum, delta, residual);
}

__global__ void ComputePoseHybridCUDAKernel(
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
        float depth_diff) {
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
            x, y, depth_diff, source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_depth_dx_indexer, target_depth_dy_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, J_I, J_D, r_I, r_D);

    // Dump J, r into JtJ and Jtr
    int offset = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            reduction[offset++] = J_I[i] * J_I[j] + J_D[i] * J_D[j];
        }
    }
    for (int i = 0; i < 6; ++i) {
        reduction[offset++] = J_I[i] * r_I + J_D[i] * r_D;
    }
    reduction[offset++] = r_I * r_D + r_D * r_D;
    reduction[offset++] = valid;

    ReduceSum6x6LinearSystem<float, kBlockSize>(tid, valid, reduction,
                                                local_sum0, local_sum1,
                                                local_sum2, global_sum);
}

void ComputePoseHybridCUDA(const core::Tensor& source_depth,
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
                           core::Tensor& residual,
                           float depth_diff) {
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

    core::Tensor global_sum =
            core::Tensor::Zeros({29}, core::Dtype::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const int kThreadSize = 16;
    const dim3 blocks((cols + kThreadSize - 1) / kThreadSize,
                      (rows + kThreadSize - 1) / kThreadSize);
    const dim3 threads(kThreadSize, kThreadSize);
    ComputePoseHybridCUDAKernel<<<blocks, threads>>>(
            source_depth_indexer, target_depth_indexer,
            source_intensity_indexer, target_intensity_indexer,
            target_depth_dx_indexer, target_depth_dy_indexer,
            target_intensity_dx_indexer, target_intensity_dy_indexer,
            source_vertex_indexer, ti, global_sum_ptr, rows, cols, depth_diff);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    DecodeAndSolve6x6(global_sum, delta, residual);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
