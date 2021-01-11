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

#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void FillInRigidAlignmentTermCUDA
#else
void FillInRigidAlignmentTermCPU
#endif
        (core::Tensor &AtA,
         core::Tensor &Atb,
         core::Tensor &residual,
         const core::Tensor &points_i,
         const core::Tensor &points_j,
         const core::Tensor &normals_i,
         int i,
         int j) {

    core::Device device = AtA.GetDevice();
    int64_t n = points_i.GetLength();
    if (points_j.GetLength() != n || normals_i.GetLength() != n) {
        utility::LogError(
                "Unable to setup linear system: input length mismatch.");
    }

    // First fill in a small 12 x 12 linear system
    core::Tensor AtA_local =
            core::Tensor::Zeros({12, 12}, core::Dtype::Float32, device);
    core::Tensor Atb_local =
            core::Tensor::Zeros({12}, core::Dtype::Float32, device);

    float *AtA_local_ptr = static_cast<float *>(AtA_local.GetDataPtr());
    float *Atb_local_ptr = static_cast<float *>(Atb_local.GetDataPtr());
    float *residual_ptr = static_cast<float *>(residual.GetDataPtr());

    const float *points_i_ptr =
            static_cast<const float *>(points_i.GetDataPtr());
    const float *points_j_ptr =
            static_cast<const float *>(points_j.GetDataPtr());
    const float *normals_i_ptr =
            static_cast<const float *>(normals_i.GetDataPtr());

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif
    launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
        const float *point_i = points_i_ptr + 3 * workload_idx;
        const float *point_j = points_j_ptr + 3 * workload_idx;
        const float *normal_i = normals_i_ptr + 3 * workload_idx;

        float r = (point_i[0] - point_j[0]) * normal_i[0] +
                  (point_i[1] - point_j[1]) * normal_i[1] +
                  (point_i[2] - point_j[2]) * normal_i[2];

        float J_ij[12];
        J_ij[0] = -point_j[2] * normal_i[1] + point_j[1] * normal_i[2];
        J_ij[1] = point_j[2] * normal_i[0] - point_j[0] * normal_i[2];
        J_ij[2] = -point_j[1] * normal_i[0] + point_j[0] * normal_i[1];
        J_ij[3] = normal_i[0];
        J_ij[4] = normal_i[1];
        J_ij[5] = normal_i[2];
        for (int k = 0; k < 6; ++k) {
            J_ij[k + 6] = -J_ij[k];
        }

        // Not optimized; Switch to reduction if necessary.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
        for (int i_local = 0; i_local < 12; ++i_local) {
            for (int j_local = 0; j_local < 12; ++j_local) {
                atomicAdd(&AtA_local_ptr[i_local * 12 + j_local],
                          J_ij[i_local] * J_ij[j_local]);
            }
            atomicAdd(&Atb_local_ptr[i_local], J_ij[i_local] * r);
        }
        atomicAdd(residual_ptr, r * r);
#else
#pragma omp critical
        {
            for (int i_local = 0; i_local < 12; ++i_local) {
                for (int j_local = 0; j_local < 12; ++j_local) {
                    AtA_local_ptr[i_local * 12 + j_local]
                      += J_ij[i_local] * J_ij[j_local];
                 }
                 Atb_local_ptr[i_local] += J_ij[i_local] * r;
            }
            *residual_ptr += r * r;
        }
#endif
    });

    // Then fill-in the large linear system
    std::vector<int64_t> indices_vec(12);
    for (int k = 0; k < 6; ++k) {
        indices_vec[k] = i * 6 + k;
        indices_vec[k + 6] = j * 6 + k;
    }

    std::vector<int64_t> indices_i_vec;
    std::vector<int64_t> indices_j_vec;
    for (int local_i = 0; local_i < 12; ++local_i) {
        for (int local_j = 0; local_j < 12; ++local_j) {
            indices_i_vec.push_back(indices_vec[local_i]);
            indices_j_vec.push_back(indices_vec[local_j]);
        }
    }

    core::Tensor indices(indices_vec, {12}, core::Dtype::Int64, device);
    core::Tensor indices_i(indices_i_vec, {12 * 12}, core::Dtype::Int64,
                           device);
    core::Tensor indices_j(indices_j_vec, {12 * 12}, core::Dtype::Int64,
                           device);

    core::Tensor AtA_sub = AtA.IndexGet({indices_i, indices_j});
    AtA.IndexSet({indices_i, indices_j}, AtA_sub + AtA_local.View({12 * 12}));

    core::Tensor Atb_sub = Atb.IndexGet({indices});
    // utility::LogInfo("Atb_sub before = {}", Atb_sub.ToString());
    // utility::LogInfo("Atb_local = {}", Atb_local.ToString());
    Atb.IndexSet({indices}, Atb_sub + Atb_local.View({12, 1}));
    // Atb_sub = Atb.IndexGet({indices});
    // utility::LogInfo("Atb_sub after = {}", Atb_sub.ToString());
}
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
