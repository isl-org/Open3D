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
    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later reduced to
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
    core::Tensor trans = init_source_to_target;
    t::geometry::kernel::TransformIndexer ti(intrinsics, trans);

    const int64_t rows = source_vertex_indexer.GetShape(0);
    const int64_t cols = source_vertex_indexer.GetShape(1);
    const int64_t n = rows * cols;

    // A_29xN is a {29, N} shaped tensor, which is later reduced to
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
