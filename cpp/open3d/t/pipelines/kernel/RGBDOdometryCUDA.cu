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
#include "open3d/core/CoreUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryImpl.h"
#include "open3d/t/pipelines/kernel/RGBDOdometryJacobian.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

#define WARPSIZE 32
#define BLOCKSIZE 1024

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void PreprocessDepthCUDA(const core::Tensor& depth,
                         core::Tensor& depth_processed,
                         float depth_scale,
                         float depth_max) {
    NDArrayIndexer depth_in_indexer(depth, 2);

    // Output
    depth_processed = core::Tensor::Empty(
            depth.GetShape(), core::Dtype::Float32, depth.GetDevice());
    NDArrayIndexer depth_out_indexer(depth_processed, 2);

    int64_t rows = depth_in_indexer.GetShape(0);
    int64_t cols = depth_in_indexer.GetShape(1);

    int64_t n = rows * cols;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::kernel::CUDALauncher::LaunchGeneralKernel(
                n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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

void PyrDownDepthCUDA(const core::Tensor& depth,
                      core::Tensor& depth_down,
                      float depth_diff) {
    t::geometry::kernel::NDArrayIndexer depth_indexer(depth, 2);
    int rows = depth_indexer.GetShape(0);
    int cols = depth_indexer.GetShape(1);

    int rows_down = rows / 2;
    int cols_down = cols / 2;
    depth_down = core::Tensor::Zeros({rows_down, cols_down},
                                     core::Dtype::Float32, depth.GetDevice());

    t::geometry::kernel::NDArrayIndexer depth_down_indexer(depth_down, 2);

    int n = rows_down * cols_down;

    const int D = 5;
    const float weights[3] = {0.375f, 0.25f, 0.0625f};

    // Reference:
    // https://github.com/mp3guy/ICPCUDA/blob/master/Cuda/pyrdown.cu#L41
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int y = workload_idx / cols_down;
                int x = workload_idx % cols_down;

                float center =
                        *depth_indexer.GetDataPtrFromCoord<float>(2 * x, 2 * y);
                if (__ISNAN(center)) {
                    *depth_down_indexer.GetDataPtrFromCoord<float>(x, y) = NAN;
                    return;
                }

                int x_mi = max(0, 2 * x - D / 2) - 2 * x;
                int y_mi = max(0, 2 * y - D / 2) - 2 * y;

                int x_ma = min(cols, 2 * x - D / 2 + D) - 2 * x;
                int y_ma = min(rows, 2 * y - D / 2 + D) - 2 * y;

                float sum = 0;
                float sum_weight = 0;
                for (int yi = y_mi; yi < y_ma; ++yi) {
                    for (int xi = x_mi; xi < x_ma; ++xi) {
                        float val = *depth_indexer.GetDataPtrFromCoord<float>(
                                2 * x + xi, 2 * y + yi);
                        if (!__ISNAN(val) && abs(val - center) < depth_diff) {
                            sum += val * weights[abs(xi)] * weights[abs(yi)];
                            sum_weight += weights[abs(xi)] * weights[abs(yi)];
                        }
                    }
                }

                *depth_down_indexer.GetDataPtrFromCoord<float>(x, y) =
                        sum / sum_weight;
            });
}

void CreateVertexMapCUDA(const core::Tensor& depth_map,
                         const core::Tensor& intrinsics,
                         core::Tensor& vertex_map) {
    NDArrayIndexer depth_indexer(depth_map, 2);
    t::geometry::kernel::TransformIndexer ti(intrinsics);

    // Output
    int64_t rows = depth_indexer.GetShape(0);
    int64_t cols = depth_indexer.GetShape(1);

    vertex_map = core::Tensor::Zeros({rows, cols, 3}, core::Dtype::Float32,
                                     depth_map.GetDevice());
    NDArrayIndexer vertex_indexer(vertex_map, 2);

    int64_t n = rows * cols;
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = *depth_indexer.GetDataPtrFromCoord<float>(x, y);

                float* vertex = vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                if (!__ISNAN(d)) {
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, vertex + 0, vertex + 1, vertex + 2);
                } else {
                    vertex[0] = NAN;
                }
            });
}

void CreateNormalMapCUDA(const core::Tensor& vertex_map,
                         core::Tensor& normal_map) {
    NDArrayIndexer vertex_indexer(vertex_map, 2);

    // Output
    int64_t rows = vertex_indexer.GetShape(0);
    int64_t cols = vertex_indexer.GetShape(1);

    normal_map =
            core::Tensor::Zeros(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());
    NDArrayIndexer normal_indexer(normal_map, 2);

    int64_t n = rows * cols;
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
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

                    if (__ISNAN(v00[0]) || __ISNAN(v10[0]) || __ISNAN(v01[0])) {
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
                }
            });
}

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

    core::Tensor trans = init_source_to_target.To(device, core::Dtype::Float32);
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later reduced to {29} where
    // [0, 20] elements are used to construct {6,6} shaped symmetric AtA matrix,
    // [21, 26] elements are used to construct {6} AtB matrix, element [27]
    // stores residual and element [28] stores count.
    core::Tensor A_29xN =
            core::Tensor::Empty({29, n}, core::Dtype::Float32, device);
    float* A_reduction = A_29xN.GetDataPtr<float>();

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float J_ij[6];
                float r;

                bool valid = GetJacobianPointToPlane(
                        workload_idx, cols, depth_diff, source_vertex_indexer,
                        target_vertex_indexer, target_normal_indexer, ti, J_ij,
                        r);

                if (valid) {
                    for (int i = 0, j = 0; j < 6; j++) {
                        for (int k = 0; k <= j; k++) {
                            A_reduction[n * i + workload_idx] =
                                    J_ij[j] * J_ij[k];
                            i++;
                        }
                        A_reduction[n * (21 + j) + workload_idx] = J_ij[j] * r;
                    }
                    A_reduction[n * 27 + workload_idx] = r * r;
                    A_reduction[n * 28 + workload_idx] = 1;
                } else {
                    for (int i = 0; i < 29; i++) {
                        A_reduction[n * i + workload_idx] = 0;
                    }
                }
            });

    ReduceAndSolve6x6(A_reduction, delta, residual, n, device);
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
    core::Tensor trans = init_source_to_target.To(device, core::Dtype::Float32);
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later red[<0;100;16M]uced to
    // {29} where [0, 20] elements are used to construct {6,6} shaped symmetric
    // AtA matrix, [21, 26] elements are used to construct {6} AtB matrix,
    // element [27] stores residual and element [28] stores count.
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
    core::Tensor trans = init_source_to_target.To(device, core::Dtype::Float32);
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later red[<0;100;16M]uced to
    // {29} where [0, 20] elements are used to construct {6,6} shaped symmetric
    // AtA matrix, [21, 26] elements are used to construct {6} AtB matrix,
    // element [27] stores residual and element [28] stores count.
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
