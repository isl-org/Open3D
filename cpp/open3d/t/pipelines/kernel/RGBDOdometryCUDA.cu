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

#include <cub/cub.cuh>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryJacobianImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ReduceAndSolve6x6(float* A_reduction,
                       core::Tensor& delta,
                       core::Tensor& residual,
                       int64_t n,
                       const core::Device& device) {
    core::Tensor output_29 =
            core::Tensor::Empty({29}, core::Dtype::Float32, device);
    float* output_29_data = output_29.GetDataPtr<float>();

    // Reduction of {29, N} to {29}.
    for (int i = 0; i < 29; i++) {
        // Determine temporary device storage requirements.
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               A_reduction + i * n, output_29_data + i, n);
        // Allocate temporary storage.
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run sum-reduction.
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                               A_reduction + i * n, output_29_data + i, n);
        cudaFree(d_temp_storage);
    }

    DecodeAndSolve6x6(output_29, delta, residual);
}

template <typename T>
__device__ inline void WarpReduceSum(volatile T* local_sum, const int tid) {
    local_sum[tid] += local_sum[tid + 32];
    local_sum[tid] += local_sum[tid + 16];
    local_sum[tid] += local_sum[tid + 8];
    local_sum[tid] += local_sum[tid + 4];
    local_sum[tid] += local_sum[tid + 2];
    local_sum[tid] += local_sum[tid + 1];
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid, volatile T* local_sum) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum[tid] += local_sum[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum[tid] += local_sum[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum[tid] += local_sum[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        WarpReduceSum<T>(local_sum, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile T* local_sum0,
                                      volatile T* local_sum1) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
    }
}

template <typename T, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile T* local_sum0,
                                      volatile T* local_sum1,
                                      volatile T* local_sum2) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
            local_sum2[tid] += local_sum2[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
        WarpReduceSum<float>(local_sum2, tid);
    }
}

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

    // A_29xN is a {29, N} shaped tensor, which is later reduced to {29} where
    // [0, 20] elements are used to construct {6,6} shaped symmetric AtA
    // matrix, [21, 26] elements are used to construct {6} AtB matrix, element
    // [27] stores residual and element [28] stores count.
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
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later reduced to
    // {29} where [0, 20] elements are used to construct {6,6} shaped
    // symmetric AtA matrix, [21, 26] elements are used to construct {6} AtB
    // matrix, element [27] stores residual and element [28] stores count.
    core::Tensor A_29xN =
            core::Tensor::Empty({29, n}, core::Dtype::Float32, device);
    float* A_reduction = A_29xN.GetDataPtr<float>();

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float J_I[6];
                float r_I;

                bool valid = GetJacobianIntensity(
                        workload_idx, cols, depth_diff, source_depth_indexer,
                        target_depth_indexer, source_intensity_indexer,
                        target_intensity_indexer, target_intensity_dx_indexer,
                        target_intensity_dy_indexer, source_vertex_indexer, ti,
                        J_I, r_I);

                if (valid) {
                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            A_reduction[n * i + workload_idx] = J_I[j] * J_I[k];
                            i++;
                        }
                        A_reduction[n * (21 + j) + workload_idx] = J_I[j] * r_I;
                    }
                    A_reduction[n * 27 + workload_idx] = r_I * r_I;
                    A_reduction[n * 28 + workload_idx] = 1;
                } else {
                    for (int i = 0; i < 29; i++) {
                        A_reduction[n * i + workload_idx] = 0;
                    }
                }
            });

    ReduceAndSolve6x6(A_reduction, delta, residual, n, device);
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
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later reduced to
    // {29} where [0, 20] elements are used to construct {6,6} shaped
    // symmetric AtA matrix, [21, 26] elements are used to construct {6} AtB
    // matrix, element [27] stores residual and element [28] stores count.
    core::Tensor A_29xN =
            core::Tensor::Empty({29, n}, core::Dtype::Float32, device);
    float* A_reduction = A_29xN.GetDataPtr<float>();

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float J_I[6], J_D[6];
                float r_I, r_D;

                bool valid = GetJacobianHybrid(
                        workload_idx, cols, depth_diff, source_depth_indexer,
                        target_depth_indexer, source_intensity_indexer,
                        target_intensity_indexer, target_depth_dx_indexer,
                        target_depth_dy_indexer, target_intensity_dx_indexer,
                        target_intensity_dy_indexer, source_vertex_indexer, ti,
                        J_I, J_D, r_I, r_D);

                if (valid) {
                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            A_reduction[n * i + workload_idx] =
                                    J_I[j] * J_I[k] + J_D[j] * J_D[k];
                            i++;
                        }
                        A_reduction[n * (21 + j) + workload_idx] =
                                J_I[j] * r_I + J_D[j] * r_D;
                    }
                    A_reduction[n * 27 + workload_idx] = r_I * r_I + r_D * r_D;
                    A_reduction[n * 28 + workload_idx] = 1;
                } else {
                    for (int i = 0; i < 29; i++) {
                        A_reduction[n * i + workload_idx] = 0;
                    }
                }
            });
    ReduceAndSolve6x6(A_reduction, delta, residual, n, device);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
