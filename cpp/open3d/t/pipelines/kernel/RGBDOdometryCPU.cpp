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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryJacobianImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeOdometryResultPointToPlaneCPU(
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _MSC_VER
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    int y = workload_idx / cols;
                    int x = workload_idx % cols;

                    float J_ij[6];
                    float r;

                    bool valid = GetJacobianPointToPlane(
                            x, y, depth_outlier_trunc, source_vertex_indexer,
                            target_vertex_indexer, target_normal_indexer, ti,
                            J_ij, r);

                    if (valid) {
                        float d_huber = HuberDeriv(r, depth_huber_delta);
                        float r_huber = HuberLoss(r, depth_huber_delta);
                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] += J_ij[j] * J_ij[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_ij[j] * d_huber;
                        }
                        A_reduction[27] += r_huber;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _MSC_VER
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<float> a, std::vector<float> b) {
                std::vector<float> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif
    core::Tensor A_reduction_tensor(A_1x29, {29}, core::Float32, device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, inlier_residual, inlier_count);
}

void ComputeOdometryResultIntensityCPU(
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _MSC_VER
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    int y = workload_idx / cols;
                    int x = workload_idx % cols;

                    float J_I[6];
                    float r_I;

                    bool valid = GetJacobianIntensity(
                            x, y, depth_outlier_trunc, source_depth_indexer,
                            target_depth_indexer, source_intensity_indexer,
                            target_intensity_indexer,
                            target_intensity_dx_indexer,
                            target_intensity_dy_indexer, source_vertex_indexer,
                            ti, J_I, r_I);

                    if (valid) {
                        float d_huber = HuberDeriv(r_I, intensity_huber_delta);
                        float r_huber = HuberLoss(r_I, intensity_huber_delta);

                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] += J_I[j] * J_I[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_I[j] * d_huber;
                        }
                        A_reduction[27] += r_huber;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _MSC_VER
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<float> a, std::vector<float> b) {
                std::vector<float> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif
    core::Tensor A_reduction_tensor(A_1x29, {29}, core::Float32, device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, inlier_residual, inlier_count);
}

void ComputeOdometryResultHybridCPU(const core::Tensor& source_depth,
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _MSC_VER
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static) num_threads(utility::EstimateMaxThreads())
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    int y = workload_idx / cols;
                    int x = workload_idx % cols;

                    float J_I[6], J_D[6];
                    float r_I, r_D;

                    bool valid = GetJacobianHybrid(
                            x, y, depth_outlier_trunc, source_depth_indexer,
                            target_depth_indexer, source_intensity_indexer,
                            target_intensity_indexer, target_depth_dx_indexer,
                            target_depth_dy_indexer,
                            target_intensity_dx_indexer,
                            target_intensity_dy_indexer, source_vertex_indexer,
                            ti, J_I, J_D, r_I, r_D);

                    if (valid) {
                        float d_huber_I =
                                HuberDeriv(r_I, intensity_huber_delta);
                        float d_huber_D = HuberDeriv(r_D, depth_huber_delta);

                        float r_huber_I = HuberLoss(r_I, intensity_huber_delta);
                        float r_huber_D = HuberLoss(r_D, depth_huber_delta);

                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] +=
                                        J_I[j] * J_I[k] + J_D[j] * J_D[k];
                                i++;
                            }
                            A_reduction[21 + j] +=
                                    J_I[j] * d_huber_I + J_D[j] * d_huber_D;
                        }
                        A_reduction[27] += r_huber_I + r_huber_D;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _MSC_VER
                return A_reduction;
            },
            // TBB: Defining reduction operation.
            [&](std::vector<float> a, std::vector<float> b) {
                std::vector<float> result(29);
                for (int j = 0; j < 29; j++) {
                    result[j] = a[j] + b[j];
                }
                return result;
            });
#endif
    core::Tensor A_reduction_tensor(A_1x29, {29}, core::Float32, device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, inlier_residual, inlier_count);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
