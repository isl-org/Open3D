// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
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

using t::geometry::kernel::NDArrayIndexer;
using t::geometry::kernel::TransformIndexer;

static constexpr int kReduceDimOdometry = 29;
static constexpr int kJtJDimOdometry = 21;

void ComputeOdometryResultPointToPlaneSYCL(
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
    const int64_t n = rows * cols;

    core::Tensor global_sum =
            core::Tensor::Zeros({kReduceDimOdometry}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     const int workload_idx = id[0];
                     const int y = workload_idx / cols;
                     const int x = workload_idx % cols;

                     float J[6] = {0};
                     float r = 0;
                     const bool valid = GetJacobianPointToPlane(
                             x, y, depth_outlier_trunc, source_vertex_indexer,
                             target_vertex_indexer, target_normal_indexer, ti,
                             J, r);

                     if (valid) {
                         const float d_huber =
                                 HuberDeriv(r, depth_huber_delta);
                         const float r_huber =
                                 HuberLoss(r, depth_huber_delta);

                         int offset = 0;
                         for (int i = 0; i < 6; ++i) {
                             for (int j = 0; j <= i; ++j) {
                                 sycl::atomic_ref<
                                         float, sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device>(
                                         global_sum_ptr[offset++]) +=
                                         J[i] * J[j];
                             }
                         }
                         for (int i = 0; i < 6; ++i) {
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     global_sum_ptr[21 + i]) +=
                                     J[i] * d_huber;
                         }
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[27]) += r_huber;
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[28]) += 1.0f;
                     }
                 })
            .wait_and_throw();

    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

void ComputeOdometryResultIntensitySYCL(
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
    TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    core::Tensor global_sum =
            core::Tensor::Zeros({kReduceDimOdometry}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     const int workload_idx = id[0];
                     const int y = workload_idx / cols;
                     const int x = workload_idx % cols;

                     float J_I[6] = {0};
                     float r_I = 0;
                     const bool valid = GetJacobianIntensity(
                             x, y, depth_outlier_trunc, source_depth_indexer,
                             target_depth_indexer, source_intensity_indexer,
                             target_intensity_indexer,
                             target_intensity_dx_indexer,
                             target_intensity_dy_indexer, source_vertex_indexer,
                             ti, J_I, r_I);

                     if (valid) {
                         const float d_huber =
                                 HuberDeriv(r_I, intensity_huber_delta);
                         const float r_huber =
                                 HuberLoss(r_I, intensity_huber_delta);

                         int offset = 0;
                         for (int i = 0; i < 6; ++i) {
                             for (int j = 0; j <= i; ++j) {
                                 sycl::atomic_ref<
                                         float, sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device>(
                                         global_sum_ptr[offset++]) +=
                                         J_I[i] * J_I[j];
                             }
                         }
                         for (int i = 0; i < 6; ++i) {
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     global_sum_ptr[21 + i]) +=
                                     J_I[i] * d_huber;
                         }
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[27]) += r_huber;
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[28]) += 1.0f;
                     }
                 })
            .wait_and_throw();

    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

void ComputeOdometryResultHybridSYCL(
        const core::Tensor& source_depth,
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
    TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    core::Tensor global_sum =
            core::Tensor::Zeros({kReduceDimOdometry}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     const int workload_idx = id[0];
                     const int y = workload_idx / cols;
                     const int x = workload_idx % cols;

                     float J_I[6] = {0}, J_D[6] = {0};
                     float r_I = 0, r_D = 0;
                     const bool valid = GetJacobianHybrid(
                             x, y, depth_outlier_trunc, source_depth_indexer,
                             target_depth_indexer, source_intensity_indexer,
                             target_intensity_indexer, target_depth_dx_indexer,
                             target_depth_dy_indexer,
                             target_intensity_dx_indexer,
                             target_intensity_dy_indexer, source_vertex_indexer,
                             ti, J_I, J_D, r_I, r_D);

                     if (valid) {
                         int offset = 0;
                         for (int i = 0; i < 6; ++i) {
                             for (int j = 0; j <= i; ++j) {
                                 sycl::atomic_ref<
                                         float, sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device>(
                                         global_sum_ptr[offset++]) +=
                                         J_I[i] * J_I[j] + J_D[i] * J_D[j];
                             }
                         }
                         for (int i = 0; i < 6; ++i) {
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     global_sum_ptr[21 + i]) +=
                                     J_I[i] *
                                             HuberDeriv(r_I,
                                                        intensity_huber_delta) +
                                     J_D[i] * HuberDeriv(r_D, depth_huber_delta);
                         }
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[27]) +=
                                 HuberLoss(r_I, intensity_huber_delta) +
                                 HuberLoss(r_D, depth_huber_delta);
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 global_sum_ptr[28]) += 1.0f;
                     }
                 })
            .wait_and_throw();

    DecodeAndSolve6x6(global_sum, delta, inlier_residual, inlier_count);
}

void ComputeOdometryInformationMatrixSYCL(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& intrinsic,
        const core::Tensor& source_to_target,
        const float square_dist_thr,
        core::Tensor& information) {
    NDArrayIndexer source_vertex_indexer(source_vertex_map, 2);
    NDArrayIndexer target_vertex_indexer(target_vertex_map, 2);

    core::Device device = source_vertex_map.GetDevice();
    core::Tensor trans = source_to_target;
    TransformIndexer ti(intrinsic, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    core::Tensor global_sum =
            core::Tensor::Zeros({kJtJDimOdometry}, core::Float32, device);
    float* global_sum_ptr = global_sum.GetDataPtr<float>();

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     const int workload_idx = id[0];
                     const int y = workload_idx / cols;
                     const int x = workload_idx % cols;

                     float J_x[6] = {0}, J_y[6] = {0}, J_z[6] = {0};
                     float rx = 0, ry = 0, rz = 0;
                     const bool valid = GetJacobianPointToPoint(
                             x, y, square_dist_thr, source_vertex_indexer,
                             target_vertex_indexer, ti, J_x, J_y, J_z, rx, ry,
                             rz);

                     if (valid) {
                         int offset = 0;
                         for (int i = 0; i < 6; ++i) {
                             for (int j = 0; j <= i; ++j) {
                                 sycl::atomic_ref<
                                         float, sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device>(
                                         global_sum_ptr[offset++]) +=
                                         J_x[i] * J_x[j] + J_y[i] * J_y[j] +
                                         J_z[i] * J_z[j];
                             }
                         }
                     }
                 })
            .wait_and_throw();

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
