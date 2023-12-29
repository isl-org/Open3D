// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cuda.h>

#include <cub/cub.cuh>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryJacobianImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

const int kBlockSize = 256;
const int kJtJDim = 21;
const int kJtrDim = 6;
const int kReduceDim =
        kJtJDim + kJtrDim + 1 + 1;  // 21 (JtJ) + 6 (Jtr) + 1 (inlier) + 1 (r)
typedef utility::MiniVec<float, kReduceDim> ReduceVec;
typedef cub::BlockReduce<ReduceVec, kBlockSize> BlockReduce;

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
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int workload = threadIdx.x + blockIdx.x * blockDim.x;
    int y = workload / cols;
    int x = workload % cols;
    const int tid = threadIdx.x;

    ReduceVec local_sum(0.0f);
    if (workload < rows * cols) {
        float J[6] = {0};
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
                local_sum[offset++] = valid ? J[i] * J[j] : 0;
            }
        }
        for (int i = 0; i < 6; ++i) {
            local_sum[offset++] = valid ? J[i] * d_huber : 0;
        }
        local_sum[offset++] = valid ? r_huber : 0;
        local_sum[offset++] = valid;
    }

    auto result = BlockReduce(temp_storage).Sum(local_sum);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kReduceDim; ++i) {
            atomicAdd(&global_sum[i], result[i]);
        }
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
    core::CUDAScopedDevice scoped_device(source_vertex_map.GetDevice());

    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);
    NDArrayIndexer target_normal_indexer(target_normal_map, 2);

    core::Device device = source_vertex_map.GetDevice();

    core::Tensor trans = init_source_to_target;
    TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum =
            core::Tensor::Zeros({kReduceDim}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const dim3 blocks((rows * cols + kBlockSize - 1) / kBlockSize);
    const dim3 threads(kBlockSize);
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
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int workload = threadIdx.x + blockIdx.x * blockDim.x;
    int y = workload / cols;
    int x = workload % cols;
    const int tid = threadIdx.x;

    ReduceVec local_sum(0.0f);
    if (workload < rows * cols) {
        float J[6] = {0};
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
                local_sum[offset++] = J[i] * J[j];
            }
        }
        for (int i = 0; i < 6; ++i) {
            local_sum[offset++] = J[i] * HuberDeriv(r, intensity_huber_delta);
        }
        local_sum[offset++] = HuberLoss(r, intensity_huber_delta);
        local_sum[offset++] = valid;
    }

    auto result = BlockReduce(temp_storage).Sum(local_sum);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kReduceDim; ++i) {
            atomicAdd(&global_sum[i], result[i]);
        }
    }
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
    core::CUDAScopedDevice scoped_device(source_depth.GetDevice());

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
            core::Tensor::Zeros({kReduceDim}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const dim3 blocks((cols * rows + kBlockSize - 1) / kBlockSize);
    const dim3 threads(kBlockSize);
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
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int workload = threadIdx.x + blockIdx.x * blockDim.x;
    int y = workload / cols;
    int x = workload % cols;
    const int tid = threadIdx.x;

    ReduceVec local_sum(0.0f);
    if (workload < rows * cols) {
        float J_I[6] = {0}, J_D[6] = {0};
        float r_I = 0, r_D = 0;
        bool valid = GetJacobianHybrid(
                x, y, depth_outlier_trunc, source_depth_indexer,
                target_depth_indexer, source_intensity_indexer,
                target_intensity_indexer, target_depth_dx_indexer,
                target_depth_dy_indexer, target_intensity_dx_indexer,
                target_intensity_dy_indexer, source_vertex_indexer, ti, J_I,
                J_D, r_I, r_D);

        float d_huber_D = HuberDeriv(r_D, depth_huber_delta);
        float d_huber_I = HuberDeriv(r_I, intensity_huber_delta);

        float r_huber_D = HuberLoss(r_D, depth_huber_delta);
        float r_huber_I = HuberLoss(r_I, intensity_huber_delta);

        // Dump J, r into JtJ and Jtr
        int offset = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j <= i; ++j) {
                local_sum[offset++] = J_I[i] * J_I[j] + J_D[i] * J_D[j];
            }
        }
        for (int i = 0; i < 6; ++i) {
            local_sum[offset++] = J_I[i] * d_huber_I + J_D[i] * d_huber_D;
        }
        local_sum[offset++] = r_huber_D + r_huber_I;
        local_sum[offset++] = valid;
    }

    auto result = BlockReduce(temp_storage).Sum(local_sum);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kReduceDim; ++i) {
            atomicAdd(&global_sum[i], result[i]);
        }
    }
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
    core::CUDAScopedDevice scoped_device(source_depth.GetDevice());

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
            core::Tensor::Zeros({kReduceDim}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const dim3 blocks((cols * rows + kBlockSize - 1) / kBlockSize);
    const dim3 threads(kBlockSize);
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

__global__ void ComputeOdometryInformationMatrixCUDAKernel(
        NDArrayIndexer source_vertex_indexer,
        NDArrayIndexer target_vertex_indexer,
        TransformIndexer ti,
        float* global_sum,
        int rows,
        int cols,
        const float square_dist_thr) {
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int workload = threadIdx.x + blockIdx.x * blockDim.x;
    int y = workload / cols;
    int x = workload % cols;
    const int tid = threadIdx.x;

    ReduceVec local_sum(0.0f);
    if (workload < rows * cols) {
        float J_x[6], J_y[6], J_z[6];
        float rx = 0, ry = 0, rz = 0;
        bool valid = GetJacobianPointToPoint(
                x, y, square_dist_thr, source_vertex_indexer,
                target_vertex_indexer, ti, J_x, J_y, J_z, rx, ry, rz);

        // Dump J, r into JtJ and Jtr
        int offset = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j <= i; ++j) {
                local_sum[offset] = valid ? J_x[i] * J_x[j] : 0;
                local_sum[offset] += valid ? J_y[i] * J_y[j] : 0;
                local_sum[offset] += valid ? J_z[i] * J_z[j] : 0;
                offset++;
            }
        }
    }

    auto result = BlockReduce(temp_storage).Sum(local_sum);
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < kJtJDim; ++i) {
            atomicAdd(&global_sum[i], result[i]);
        }
    }
}

void ComputeOdometryInformationMatrixCUDA(const core::Tensor& source_vertex_map,
                                          const core::Tensor& target_vertex_map,
                                          const core::Tensor& intrinsic,
                                          const core::Tensor& source_to_target,
                                          const float square_dist_thr,
                                          core::Tensor& information) {
    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);

    core::Device device = source_vertex_map.GetDevice();
    core::Tensor trans = source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsic, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);

    core::Tensor global_sum =
            core::Tensor::Zeros({kJtJDim}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    const dim3 blocks((cols * rows + kBlockSize - 1) / kBlockSize);
    const dim3 threads(kBlockSize);
    ComputeOdometryInformationMatrixCUDAKernel<<<blocks, threads, 0,
                                                 core::cuda::GetStream()>>>(
            source_vertex_indexer, target_vertex_indexer, ti, global_sum_ptr,
            rows, cols, square_dist_thr);
    core::cuda::Synchronize();

    // 21 => 6x6
    const core::Device host(core::Device("CPU:0"));
    information = core::Tensor::Empty({6, 6}, core::Float64, host);
    global_sum = global_sum.To(host, core::Float64);

    double* info_ptr = information.GetDataPtr<double>();
    double* reduction_ptr = global_sum.GetDataPtr<double>();
    for (int j = 0; j < 6; j++) {
        const int64_t reduction_idx = ((j * (j + 1)) / 2);
        for (int k = 0; k <= j; k++) {
            info_ptr[j * 6 + k] = reduction_ptr[reduction_idx + k];
            info_ptr[k * 6 + j] = reduction_ptr[reduction_idx + k];
        }
    }
}
}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
