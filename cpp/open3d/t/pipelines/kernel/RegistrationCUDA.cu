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

#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace registration {

inline void ReduceAndSolve6x6(float *A_reduction,
                              int64_t n,
                              core::Tensor &delta,
                              double &residual,
                              int64_t &count,
                              const core::Device &device) {
    core::Tensor output_29 =
            core::Tensor::Empty({29}, core::Dtype::Float32, device);
    float *output_29_data = output_29.GetDataPtr<float>();

    // Reduction of {29, N} to {29}.
    for (int i = 0; i < 29; i++) {
        // Determine temporary device storage requirements.
        void *d_temp_storage = NULL;
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

    DecodeAndSolve6x6(output_29, delta, residual, count);
}

void ComputePosePointToPlaneCUDA(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const std::pair<core::Tensor, core::Tensor> &corres,
        core::Tensor &pose,
        double &residual,
        int64_t &count,
        const core::Dtype &dtype,
        const core::Device &device) {
    const float *source_points_ptr = source_points.GetDataPtr<float>();
    const float *target_points_ptr = target_points.GetDataPtr<float>();
    const float *target_normals_ptr = target_normals.GetDataPtr<float>();
    const int64_t *correspondences_first = corres.first.GetDataPtr<int64_t>();
    const float *correspondences_second = corres.second.GetDataPtr<float>();

    int n = corres.first.GetLength();

    // A_29xN is a {29, N} shaped tensor, which is later reduced to {29} where
    // [0, 20] elements are used to construct {6,6} shaped symmetric AtA matrix,
    // and [21, 26] elements are used to construct {6} AtB matrix, [27] is
    // residual or squared_error, [28] is number of correspondences or count.
    core::Tensor A_29xN =
            core::Tensor::Empty({29, n}, core::Dtype::Float32, device);
    float *A_reduction = A_29xN.GetDataPtr<float>();

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float J_ij[6];
                float r;

                bool valid = GetJacobianPointToPlane<float>(
                        workload_idx, source_points_ptr, target_points_ptr,
                        target_normals_ptr, correspondences_first, J_ij, r);

                if (valid) {
                    A_reduction[workload_idx] = J_ij[0] * J_ij[0];
                    A_reduction[n + workload_idx] = J_ij[1] * J_ij[0];
                    A_reduction[n * 2 + workload_idx] = J_ij[1] * J_ij[1];
                    A_reduction[n * 3 + workload_idx] = J_ij[2] * J_ij[0];
                    A_reduction[n * 4 + workload_idx] = J_ij[2] * J_ij[1];
                    A_reduction[n * 5 + workload_idx] = J_ij[2] * J_ij[2];
                    A_reduction[n * 6 + workload_idx] = J_ij[3] * J_ij[0];
                    A_reduction[n * 7 + workload_idx] = J_ij[3] * J_ij[1];
                    A_reduction[n * 8 + workload_idx] = J_ij[3] * J_ij[2];
                    A_reduction[n * 9 + workload_idx] = J_ij[3] * J_ij[3];
                    A_reduction[n * 10 + workload_idx] = J_ij[4] * J_ij[0];
                    A_reduction[n * 11 + workload_idx] = J_ij[4] * J_ij[1];
                    A_reduction[n * 12 + workload_idx] = J_ij[4] * J_ij[2];
                    A_reduction[n * 13 + workload_idx] = J_ij[4] * J_ij[3];
                    A_reduction[n * 14 + workload_idx] = J_ij[4] * J_ij[4];
                    A_reduction[n * 15 + workload_idx] = J_ij[5] * J_ij[0];
                    A_reduction[n * 16 + workload_idx] = J_ij[5] * J_ij[1];
                    A_reduction[n * 17 + workload_idx] = J_ij[5] * J_ij[2];
                    A_reduction[n * 18 + workload_idx] = J_ij[5] * J_ij[3];
                    A_reduction[n * 19 + workload_idx] = J_ij[5] * J_ij[4];
                    A_reduction[n * 20 + workload_idx] = J_ij[5] * J_ij[5];

                    A_reduction[n * 21 + workload_idx] = J_ij[0] * r;
                    A_reduction[n * 22 + workload_idx] = J_ij[1] * r;
                    A_reduction[n * 23 + workload_idx] = J_ij[2] * r;
                    A_reduction[n * 24 + workload_idx] = J_ij[3] * r;
                    A_reduction[n * 25 + workload_idx] = J_ij[4] * r;
                    A_reduction[n * 26 + workload_idx] = J_ij[5] * r;

                    A_reduction[n * 27 + workload_idx] =
                            correspondences_second[workload_idx];
                    A_reduction[n * 28 + workload_idx] = 1;

                } else {
                    A_reduction[n + workload_idx] = 0;
                    A_reduction[n * 2 + workload_idx] = 0;
                    A_reduction[n * 3 + workload_idx] = 0;
                    A_reduction[n * 4 + workload_idx] = 0;
                    A_reduction[n * 5 + workload_idx] = 0;
                    A_reduction[n * 6 + workload_idx] = 0;
                    A_reduction[n * 7 + workload_idx] = 0;
                    A_reduction[n * 8 + workload_idx] = 0;
                    A_reduction[n * 9 + workload_idx] = 0;
                    A_reduction[n * 10 + workload_idx] = 0;
                    A_reduction[n * 11 + workload_idx] = 0;
                    A_reduction[n * 12 + workload_idx] = 0;
                    A_reduction[n * 13 + workload_idx] = 0;
                    A_reduction[n * 14 + workload_idx] = 0;
                    A_reduction[n * 15 + workload_idx] = 0;
                    A_reduction[n * 16 + workload_idx] = 0;
                    A_reduction[n * 17 + workload_idx] = 0;
                    A_reduction[n * 18 + workload_idx] = 0;
                    A_reduction[n * 19 + workload_idx] = 0;
                    A_reduction[n * 20 + workload_idx] = 0;

                    A_reduction[n * 21 + workload_idx] = 0;
                    A_reduction[n * 22 + workload_idx] = 0;
                    A_reduction[n * 23 + workload_idx] = 0;
                    A_reduction[n * 24 + workload_idx] = 0;
                    A_reduction[n * 25 + workload_idx] = 0;
                    A_reduction[n * 26 + workload_idx] = 0;

                    A_reduction[n * 27 + workload_idx] = 0;
                    A_reduction[n * 28 + workload_idx] = 0;
                }
            });

    ReduceAndSolve6x6(A_reduction, n, pose, residual, count, device);
}

}  // namespace registration
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
