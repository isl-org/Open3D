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

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
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

// TODO (Wei): add shared shape checkers for geometry and pipeline kernel calls
void PreprocessDepthCPU(const core::Tensor& depth,
                        core::Tensor& depth_processed,
                        float depth_scale,
                        float depth_max) {
    NDArrayIndexer depth_in_indexer(depth, 2);

    depth_processed = core::Tensor::Empty(
            depth.GetShape(), core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer depth_out_indexer(depth_processed, 2);

    // Output
    int64_t rows = depth_in_indexer.GetShape(0);
    int64_t cols = depth_in_indexer.GetShape(1);

    int64_t n = rows * cols;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&] {
        core::kernel::CPULauncher::LaunchGeneralKernel(
                n, [&](int64_t workload_idx) {
                    int64_t y = workload_idx / cols;
                    int64_t x = workload_idx % cols;

                    float d = *depth_in_indexer.GetDataPtrFromCoord<scalar_t>(
                                      x, y) /
                              depth_scale;
                    float* d_out_ptr =
                            depth_out_indexer.GetDataPtrFromCoord<float>(x, y);

                    bool valid = (d > 0 && d < depth_max);
                    *d_out_ptr = valid ? d : NAN;
                });
    });
}

void PyrDownDepthCPU(const core::Tensor& depth,
                     core::Tensor& depth_down,
                     float depth_diff) {
    t::geometry::kernel::NDArrayIndexer depth_indexer(depth, 2);
    int rows = depth_indexer.GetShape(0);
    int cols = depth_indexer.GetShape(1);

    int rows_down = rows / 2;
    int cols_down = cols / 2;
    depth_down = core::Tensor::Empty({rows_down, cols_down},
                                     core::Dtype::Float32, depth.GetDevice());

    t::geometry::kernel::NDArrayIndexer depth_down_indexer(depth_down, 2);

    int n = rows_down * cols_down;

    // Gaussian filter window size
    const int D = 5;
    // Gaussian filter weights
    const float weights[3] = {0.375f, 0.25f, 0.0625f};

    // Reference:
    // https://github.com/mp3guy/ICPCUDA/blob/master/Cuda/pyrdown.cu#L41
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                int y = workload_idx / cols_down;
                int x = workload_idx % cols_down;

                float center =
                        *depth_indexer.GetDataPtrFromCoord<float>(2 * x, 2 * y);
                if (std::isnan(center)) {
                    *depth_down_indexer.GetDataPtrFromCoord<float>(x, y) = NAN;
                    return;
                }

                int x_min = std::max(0, 2 * x - D / 2) - 2 * x;
                int y_min = std::max(0, 2 * y - D / 2) - 2 * y;

                int x_max = std::min(cols, 2 * x - D / 2 + D) - 2 * x;
                int y_max = std::min(rows, 2 * y - D / 2 + D) - 2 * y;

                float sum = 0;
                float sum_weight = 0;
                for (int yi = y_min; yi < y_max; ++yi) {
                    for (int xi = x_min; xi < x_max; ++xi) {
                        float val = *depth_indexer.GetDataPtrFromCoord<float>(
                                2 * x + xi, 2 * y + yi);
                        if (!std::isnan(val) &&
                            abs(val - center) < depth_diff) {
                            sum += val * weights[abs(xi)] * weights[abs(yi)];
                            sum_weight += weights[abs(xi)] * weights[abs(yi)];
                        }
                    }
                }

                *depth_down_indexer.GetDataPtrFromCoord<float>(x, y) =
                        sum / sum_weight;
            });
}

void CreateVertexMapCPU(const core::Tensor& depth_map,
                        const core::Tensor& intrinsics,
                        core::Tensor& vertex_map) {
    NDArrayIndexer depth_indexer(depth_map, 2);
    t::geometry::kernel::TransformIndexer ti(
            intrinsics,
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0")));

    // Output
    int64_t rows = depth_indexer.GetShape(0);
    int64_t cols = depth_indexer.GetShape(1);

    vertex_map = core::Tensor::Empty({rows, cols, 3}, core::Dtype::Float32,
                                     depth_map.GetDevice());
    NDArrayIndexer vertex_indexer(vertex_map, 2);

    int64_t n = rows * cols;

    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = *depth_indexer.GetDataPtrFromCoord<float>(x, y);

                float* vertex = vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                if (!std::isnan(d)) {
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, vertex + 0, vertex + 1, vertex + 2);
                } else {
                    vertex[0] = NAN;
                }
            });
}

void CreateNormalMapCPU(const core::Tensor& vertex_map,
                        core::Tensor& normal_map) {
    NDArrayIndexer vertex_indexer(vertex_map, 2);

    // Output
    int64_t rows = vertex_indexer.GetShape(0);
    int64_t cols = vertex_indexer.GetShape(1);

    normal_map =
            core::Tensor::Empty(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());
    NDArrayIndexer normal_indexer(normal_map, 2);

    int64_t n = rows * cols;

    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float* normal = normal_indexer.GetDataPtrFromCoord<float>(x, y);

                if (y < rows - 1 && x < cols - 1) {
                    float* v00 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                    float* v10 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x + 1, y);
                    float* v01 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y + 1);

                    if (std::isnan(v00[0]) || std::isnan(v10[0]) ||
                        std::isnan(v01[0])) {
                        normal[0] = NAN;
                        return;
                    }

                    float dx0 = v01[0] - v00[0];
                    float dy0 = v01[1] - v00[1];
                    float dz0 = v01[2] - v00[2];

                    float dx1 = v10[0] - v00[0];
                    float dy1 = v10[1] - v00[1];
                    float dz1 = v10[2] - v00[2];

                    normal[0] = dy0 * dz1 - dz0 * dy1;
                    normal[1] = dz0 * dx1 - dx0 * dz1;
                    normal[2] = dx0 * dy1 - dy0 * dx1;

                    float normal_norm =
                            sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                 normal[2] * normal[2]);
                    normal[0] /= normal_norm;
                    normal[1] /= normal_norm;
                    normal[2] /= normal_norm;
                } else {
                    normal[0] = NAN;
                }
            });
}

void ComputePoseIntensityCPU(const core::Tensor& source_depth,
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _WIN32
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static)
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    float J_I[6];
                    float r_I;

                    bool valid = GetJacobianIntensity(
                            workload_idx, cols, depth_diff,
                            source_depth_indexer, target_depth_indexer,
                            source_intensity_indexer, target_intensity_indexer,
                            target_intensity_dx_indexer,
                            target_intensity_dy_indexer, source_vertex_indexer,
                            ti, J_I, r_I);

                    if (valid) {
                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] += J_I[j] * J_I[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_I[j] * r_I;
                        }
                        A_reduction[27] += r_I * r_I;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _WIN32
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
    core::Tensor A_reduction_tensor(A_1x29, {1, 29}, core::Dtype::Float32,
                                    device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, residual);
}

void ComputePosePointToPlaneCPU(const core::Tensor& source_vertex_map,
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _WIN32
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static)
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    float J_ij[6];
                    float r;

                    bool valid = GetJacobianPointToPlane(
                            workload_idx, cols, depth_diff,
                            source_vertex_indexer, target_vertex_indexer,
                            target_normal_indexer, ti, J_ij, r);

                    if (valid) {
                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] += J_ij[j] * J_ij[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_ij[j] * r;
                        }
                        A_reduction[27] += r * r;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _WIN32
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
    core::Tensor A_reduction_tensor(A_1x29, {1, 29}, core::Dtype::Float32,
                                    device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, residual);
}

void ComputePoseHybridCPU(const core::Tensor& source_depth,
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

    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();

    int64_t n = rows * cols;

    std::vector<float> A_1x29(29, 0.0);

#ifdef _WIN32
    std::vector<float> zeros_29(29, 0.0);
    A_1x29 = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, n), zeros_29,
            [&](tbb::blocked_range<int> r, std::vector<float> A_reduction) {
                for (int workload_idx = r.begin(); workload_idx < r.end();
                     workload_idx++) {
#else
    float* A_reduction = A_1x29.data();
#pragma omp parallel for reduction(+ : A_reduction[:29]) schedule(static)
    for (int workload_idx = 0; workload_idx < n; workload_idx++) {
#endif
                    float J_I[6], J_D[6];
                    float r_I, r_D;

                    bool valid = GetJacobianHybrid(
                            workload_idx, cols, depth_diff,
                            source_depth_indexer, target_depth_indexer,
                            source_intensity_indexer, target_intensity_indexer,
                            target_depth_dx_indexer, target_depth_dy_indexer,
                            target_intensity_dx_indexer,
                            target_intensity_dy_indexer, source_vertex_indexer,
                            ti, J_I, J_D, r_I, r_D);

                    if (valid) {
                        for (int i = 0, j = 0; j < 6; j++) {
                            for (int k = 0; k <= j; k++) {
                                A_reduction[i] +=
                                        J_I[j] * J_I[k] + J_D[j] * J_D[k];
                                i++;
                            }
                            A_reduction[21 + j] += J_I[j] * r_I + J_D[j] * r_D;
                        }
                        A_reduction[27] += r_I * r_I + r_D * r_D;
                        A_reduction[28] += 1;
                    }
                }
#ifdef _WIN32
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
    core::Tensor A_reduction_tensor(A_1x29, {1, 29}, core::Dtype::Float32,
                                    device);
    DecodeAndSolve6x6(A_reduction_tensor, delta, residual);
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
