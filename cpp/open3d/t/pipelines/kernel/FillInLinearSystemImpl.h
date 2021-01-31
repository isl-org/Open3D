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
         const core::Tensor &Ti_ps,
         const core::Tensor &Tj_qs,
         const core::Tensor &Ri_normal_ps,
         int i,
         int j) {

    core::Device device = AtA.GetDevice();
    int64_t n = Ti_ps.GetLength();
    if (Tj_qs.GetLength() != n || Ri_normal_ps.GetLength() != n) {
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

    const float *Ti_ps_ptr = static_cast<const float *>(Ti_ps.GetDataPtr());
    const float *Tj_qs_ptr = static_cast<const float *>(Tj_qs.GetDataPtr());
    const float *Ri_normal_ps_ptr =
            static_cast<const float *>(Ri_normal_ps.GetDataPtr());

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif
    launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
        const float *p_prime = Ti_ps_ptr + 3 * workload_idx;
        const float *q_prime = Tj_qs_ptr + 3 * workload_idx;
        const float *normal_p_prime = Ri_normal_ps_ptr + 3 * workload_idx;

        float r = (p_prime[0] - q_prime[0]) * normal_p_prime[0] +
                  (p_prime[1] - q_prime[1]) * normal_p_prime[1] +
                  (p_prime[2] - q_prime[2]) * normal_p_prime[2];

        float J_ij[12];
        J_ij[0] = -q_prime[2] * normal_p_prime[1] +
                  q_prime[1] * normal_p_prime[2];
        J_ij[1] =
                q_prime[2] * normal_p_prime[0] - q_prime[0] * normal_p_prime[2];
        J_ij[2] = -q_prime[1] * normal_p_prime[0] +
                  q_prime[0] * normal_p_prime[1];
        J_ij[3] = normal_p_prime[0];
        J_ij[4] = normal_p_prime[1];
        J_ij[5] = normal_p_prime[2];
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

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void FillInSLACAlignmentTermCUDA
#else
void FillInSLACAlignmentTermCPU
#endif
        (core::Tensor &AtA,
         core::Tensor &Atb,
         core::Tensor &residual,
         const core::Tensor &Ti_Cps,
         const core::Tensor &Tj_Cqs,
         const core::Tensor &Cnormal_ps,
         const core::Tensor &Ri_Cnormal_ps,
         const core::Tensor &RjT_Ri_Cnormal_ps,
         const core::Tensor &cgrid_idx_ps,
         const core::Tensor &cgrid_idx_qs,
         const core::Tensor &cgrid_ratio_qs,
         const core::Tensor &cgrid_ratio_ps,
         int i,
         int j,
         int n_frags) {
    int64_t n = Ti_Cps.GetLength();
    if (Tj_Cqs.GetLength() != n || Cnormal_ps.GetLength() != n ||
        Ri_Cnormal_ps.GetLength() != n || RjT_Ri_Cnormal_ps.GetLength() != n ||
        cgrid_idx_ps.GetLength() != n || cgrid_ratio_ps.GetLength() != n ||
        cgrid_idx_qs.GetLength() != n || cgrid_ratio_qs.GetLength() != n) {
        utility::LogError(
                "Unable to setup linear system: input length mismatch.");
    }

    int n_vars = Atb.GetLength();
    float *AtA_ptr = static_cast<float *>(AtA.GetDataPtr());
    float *Atb_ptr = static_cast<float *>(Atb.GetDataPtr());
    float *residual_ptr = static_cast<float *>(residual.GetDataPtr());

    // Geometric properties
    const float *Ti_Cps_ptr = static_cast<const float *>(Ti_Cps.GetDataPtr());
    const float *Tj_Cqs_ptr = static_cast<const float *>(Tj_Cqs.GetDataPtr());
    const float *Cnormal_ps_ptr =
            static_cast<const float *>(Cnormal_ps.GetDataPtr());
    const float *Ri_Cnormal_ps_ptr =
            static_cast<const float *>(Ri_Cnormal_ps.GetDataPtr());
    const float *RjT_Ri_Cnormal_ps_ptr =
            static_cast<const float *>(RjT_Ri_Cnormal_ps.GetDataPtr());

    // Association properties
    const int *cgrid_idx_ps_ptr =
            static_cast<const int *>(cgrid_idx_ps.GetDataPtr());
    const int *cgrid_idx_qs_ptr =
            static_cast<const int *>(cgrid_idx_qs.GetDataPtr());
    const float *cgrid_ratio_ps_ptr =
            static_cast<const float *>(cgrid_ratio_ps.GetDataPtr());
    const float *cgrid_ratio_qs_ptr =
            static_cast<const float *>(cgrid_ratio_qs.GetDataPtr());

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif
    launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
        const float *Ti_Cp = Ti_Cps_ptr + 3 * workload_idx;
        const float *Tj_Cq = Tj_Cqs_ptr + 3 * workload_idx;
        const float *Cnormal_p = Cnormal_ps_ptr + 3 * workload_idx;
        const float *Ri_Cnormal_p = Ri_Cnormal_ps_ptr + 3 * workload_idx;
        const float *RjTRi_Cnormal_p = RjT_Ri_Cnormal_ps_ptr + 3 * workload_idx;

        const int *cgrid_idx_p = cgrid_idx_ps_ptr + 8 * workload_idx;
        const int *cgrid_idx_q = cgrid_idx_qs_ptr + 8 * workload_idx;
        const float *cgrid_ratio_p = cgrid_ratio_ps_ptr + 8 * workload_idx;
        const float *cgrid_ratio_q = cgrid_ratio_qs_ptr + 8 * workload_idx;

        float r = (Ti_Cp[0] - Tj_Cq[0]) * Ri_Cnormal_p[0] +
                  (Ti_Cp[1] - Tj_Cq[1]) * Ri_Cnormal_p[1] +
                  (Ti_Cp[2] - Tj_Cq[2]) * Ri_Cnormal_p[2];

        // Now we fill in a 60 x 60 sub-matrix: 2 x (6 + 8 x 3)
        float J[60];
        int idx[60];

        // Jacobian w.r.t. Ti: 0-6
        J[0] = -Tj_Cq[2] * Ri_Cnormal_p[1] + Tj_Cq[1] * Ri_Cnormal_p[2];
        J[1] = Tj_Cq[2] * Ri_Cnormal_p[0] - Tj_Cq[0] * Ri_Cnormal_p[2];
        J[2] = -Tj_Cq[1] * Ri_Cnormal_p[0] + Tj_Cq[0] * Ri_Cnormal_p[1];
        J[3] = Ri_Cnormal_p[0];
        J[4] = Ri_Cnormal_p[1];
        J[5] = Ri_Cnormal_p[2];

        // Jacobian w.r.t. Tj: 6-12
        for (int k = 0; k < 6; ++k) {
            J[k + 6] = -J[k];

            idx[k + 0] = 6 * i + k;
            idx[k + 6] = 6 * j + k;
        }

        // Jacobian w.r.t. C over p: 12-36
        for (int k = 0; k < 8; ++k) {
            J[12 + k * 3 + 0] = cgrid_ratio_p[k] * Cnormal_p[0];
            J[12 + k * 3 + 1] = cgrid_ratio_p[k] * Cnormal_p[1];
            J[12 + k * 3 + 2] = cgrid_ratio_p[k] * Cnormal_p[2];

            idx[12 + k * 3 + 0] = 6 * n_frags + cgrid_idx_p[k] * 3 + 0;
            idx[12 + k * 3 + 1] = 6 * n_frags + cgrid_idx_p[k] * 3 + 1;
            idx[12 + k * 3 + 2] = 6 * n_frags + cgrid_idx_p[k] * 3 + 2;
        }

        // Jacobian w.r.t. C over q: 36-60
        for (int k = 0; k < 8; ++k) {
            J[36 + k * 3 + 0] = -cgrid_ratio_q[k] * RjTRi_Cnormal_p[0];
            J[36 + k * 3 + 1] = -cgrid_ratio_q[k] * RjTRi_Cnormal_p[1];
            J[36 + k * 3 + 2] = -cgrid_ratio_q[k] * RjTRi_Cnormal_p[2];

            idx[36 + k * 3 + 0] = 6 * n_frags + cgrid_idx_q[k] * 3 + 0;
            idx[36 + k * 3 + 1] = 6 * n_frags + cgrid_idx_q[k] * 3 + 1;
            idx[36 + k * 3 + 2] = 6 * n_frags + cgrid_idx_q[k] * 3 + 2;
        }

        // Not optimized; Switch to reduction if necessary.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
        for (int ki = 0; ki < 60; ++ki) {
            for (int kj = 0; kj < 60; ++kj) {
                float AtA_ij = J[ki] * J[kj];
                int ij = idx[ki] * n_vars + idx[kj];
                atomicAdd(AtA_ptr + ij, AtA_ij);
            }
            float Atb_i = J[ki] * r;
            atomicAdd(Atb_ptr + idx[ki], Atb_i);
        }
        atomicAdd(residual_ptr, r * r);
#else
#pragma omp critical
        {
            for (int ki = 0; ki < 12; ++ki) {
                for (int kj = 0; kj < 12; ++kj) {
                    AtA_ptr[idx[ki] * n_vars + idx[kj]]
                      += J[ki] * J[kj];
                 }
                 Atb_ptr[ki] += J[ki] * r;
            }
            *residual_ptr += r * r;
        }
#endif
    });
}
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
