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

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void CreateVertexMapCPU(const core::Tensor& depth_map,
                        const core::Tensor& intrinsics,
                        core::Tensor& vertex_map,
                        float depth_scale,
                        float depth_max) {
    t::geometry::kernel::NDArrayIndexer depth_indexer(depth_map, 2);
    t::geometry::kernel::TransformIndexer ti(intrinsics);

    // Output
    int64_t rows = depth_indexer.GetShape(0);
    int64_t cols = depth_indexer.GetShape(1);

    vertex_map = core::Tensor::Zeros({rows, cols, 3}, core::Dtype::Float32,
                                     depth_map.GetDevice());
    t::geometry::kernel::NDArrayIndexer vertex_indexer(vertex_map, 2);

    int64_t n = rows * cols;

    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = *depth_indexer.GetDataPtrFromCoord<float>(x, y) /
                          depth_scale;

                float* vertex = vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                if (d > 0 && d < depth_max) {
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, vertex + 0, vertex + 1, vertex + 2);
                } else {
                    vertex[0] = INFINITY;
                }
            });
}

void CreateNormalMapCPU(const core::Tensor& vertex_map,
                        core::Tensor& normal_map) {
    t::geometry::kernel::NDArrayIndexer vertex_indexer(vertex_map, 2);

    // Output
    int64_t rows = vertex_indexer.GetShape(0);
    int64_t cols = vertex_indexer.GetShape(1);

    normal_map =
            core::Tensor::Zeros(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());
    t::geometry::kernel::NDArrayIndexer normal_indexer(normal_map, 2);

    int64_t n = rows * cols;

    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                if (y < rows - 1 && x < cols - 1) {
                    float* v00 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                    float* v10 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x + 1, y);
                    float* v01 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y + 1);
                    float* normal =
                            normal_indexer.GetDataPtrFromCoord<float>(x, y);

                    if (v00[0] == INFINITY || v10[0] == INFINITY ||
                        v01[0] == INFINITY) {
                        normal[0] = INFINITY;
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
                }
            });
}

void ComputePosePointToPlaneCPU(const core::Tensor& source_vertex_map,
                                const core::Tensor& target_vertex_map,
                                const core::Tensor& source_normal_map,
                                const core::Tensor& intrinsics,
                                const core::Tensor& init_source_to_target,
                                core::Tensor& delta,
                                core::Tensor& residual,
                                float depth_diff) {
    t::geometry::kernel::NDArrayIndexer source_vertex_indexer(source_vertex_map,
                                                              2);
    t::geometry::kernel::NDArrayIndexer target_vertex_indexer(target_vertex_map,
                                                              2);
    t::geometry::kernel::NDArrayIndexer source_normal_indexer(source_normal_map,
                                                              2);

    core::Tensor trans = init_source_to_target.Inverse().To(
            source_vertex_map.GetDevice(), core::Dtype::Float32);
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

                    bool valid = GetJacobianLocal(
                            workload_idx, cols, depth_diff,
                            source_vertex_indexer, target_vertex_indexer,
                            source_normal_indexer, ti, J_ij, r);

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

    core::Tensor AtA =
            core::Tensor::Empty({6, 6}, core::Dtype::Float32, device);
    core::Tensor Atb = core::Tensor::Empty({6}, core::Dtype::Float32, device);

    float* AtA_local_ptr = AtA.GetDataPtr<float>();
    float* Atb_local_ptr = Atb.GetDataPtr<float>();

    for (int i = 0, j = 0; j < 6; j++) {
        for (int k = 0; k <= j; k++) {
            AtA_local_ptr[j * 6 + k] = A_1x29[i];
            AtA_local_ptr[k * 6 + j] = A_1x29[i];
            i++;
        }
        Atb_local_ptr[j] = A_1x29[21 + j];
    }

    residual = core::Tensor::Init<float>({A_1x29[27]}, device);

    int count = static_cast<int>(A_1x29[28]);

    utility::LogDebug("avg loss = {}, residual = {}, count = {}",
                      residual.Item<float>() / count, residual.Item<float>(),
                      count);

    // Solve on CPU with double to ensure precision.
    core::Device host(core::Device("CPU:0"));
    delta = AtA.To(host, core::Dtype::Float64)
                    .Solve(Atb.Neg().To(host, core::Dtype::Float64));
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
