// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Skip the CPU/CUDA main function definitions; include only helper includes.
#define OPEN3D_SKIP_FILL_IN_LS_MAIN
#include "open3d/t/pipelines/kernel/FillInLinearSystemImpl.h"
#undef OPEN3D_SKIP_FILL_IN_LS_MAIN

#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void FillInRigidAlignmentTermSYCL(core::Tensor &AtA,
                                   core::Tensor &Atb,
                                   core::Tensor &residual,
                                   const core::Tensor &Ti_ps,
                                   const core::Tensor &Tj_qs,
                                   const core::Tensor &Ri_normal_ps,
                                   int i,
                                   int j,
                                   float threshold) {
    core::Device device = AtA.GetDevice();
    int64_t n = Ti_ps.GetLength();
    if (Tj_qs.GetLength() != n || Ri_normal_ps.GetLength() != n) {
        utility::LogError(
                "Unable to setup linear system: input length mismatch.");
    }

    // First fill in a small 12 x 12 linear system.
    core::Tensor AtA_local =
            core::Tensor::Zeros({12, 12}, core::Float32, device);
    core::Tensor Atb_local = core::Tensor::Zeros({12}, core::Float32, device);

    float *AtA_local_ptr = static_cast<float *>(AtA_local.GetDataPtr());
    float *Atb_local_ptr = static_cast<float *>(Atb_local.GetDataPtr());
    float *residual_ptr = static_cast<float *>(residual.GetDataPtr());

    const float *Ti_ps_ptr = static_cast<const float *>(Ti_ps.GetDataPtr());
    const float *Tj_qs_ptr = static_cast<const float *>(Tj_qs.GetDataPtr());
    const float *Ri_normal_ps_ptr =
            static_cast<const float *>(Ri_normal_ps.GetDataPtr());

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     int64_t workload_idx = id[0];
                     const float *p_prime = Ti_ps_ptr + 3 * workload_idx;
                     const float *q_prime = Tj_qs_ptr + 3 * workload_idx;
                     const float *normal_p_prime =
                             Ri_normal_ps_ptr + 3 * workload_idx;

                     float r = (p_prime[0] - q_prime[0]) *
                                       normal_p_prime[0] +
                               (p_prime[1] - q_prime[1]) *
                                       normal_p_prime[1] +
                               (p_prime[2] - q_prime[2]) * normal_p_prime[2];
                     if (sycl::fabs(r) > threshold) return;

                     float J_ij[12];
                     J_ij[0] = -q_prime[2] * normal_p_prime[1] +
                               q_prime[1] * normal_p_prime[2];
                     J_ij[1] = q_prime[2] * normal_p_prime[0] -
                               q_prime[0] * normal_p_prime[2];
                     J_ij[2] = -q_prime[1] * normal_p_prime[0] +
                               q_prime[0] * normal_p_prime[1];
                     J_ij[3] = normal_p_prime[0];
                     J_ij[4] = normal_p_prime[1];
                     J_ij[5] = normal_p_prime[2];
                     for (int k = 0; k < 6; ++k) {
                         J_ij[k + 6] = -J_ij[k];
                     }

                     for (int i_local = 0; i_local < 12; ++i_local) {
                         for (int j_local = 0; j_local < 12; ++j_local) {
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_local_ptr[i_local * 12 + j_local]) +=
                                     J_ij[i_local] * J_ij[j_local];
                         }
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 Atb_local_ptr[i_local]) +=
                                 J_ij[i_local] * r;
                     }
                     sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                      sycl::memory_scope::device>(
                             *residual_ptr) += r * r;
                 })
            .wait_and_throw();

    // Then fill-in the large linear system.
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

    core::Tensor indices(indices_vec, {12}, core::Int64, device);
    core::Tensor indices_i(indices_i_vec, {12 * 12}, core::Int64, device);
    core::Tensor indices_j(indices_j_vec, {12 * 12}, core::Int64, device);

    core::Tensor AtA_sub = AtA.IndexGet({indices_i, indices_j});
    AtA.IndexSet({indices_i, indices_j}, AtA_sub + AtA_local.View({12 * 12}));

    core::Tensor Atb_sub = Atb.IndexGet({indices});
    Atb.IndexSet({indices}, Atb_sub + Atb_local.View({12, 1}));
}

void FillInSLACAlignmentTermSYCL(core::Tensor &AtA,
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
                                  int n_frags,
                                  float threshold) {
    int64_t n = Ti_Cps.GetLength();
    if (Tj_Cqs.GetLength() != n || Cnormal_ps.GetLength() != n ||
        Ri_Cnormal_ps.GetLength() != n ||
        RjT_Ri_Cnormal_ps.GetLength() != n ||
        cgrid_idx_ps.GetLength() != n || cgrid_ratio_ps.GetLength() != n ||
        cgrid_idx_qs.GetLength() != n || cgrid_ratio_qs.GetLength() != n) {
        utility::LogError(
                "Unable to setup linear system: input length mismatch.");
    }

    core::Device device = AtA.GetDevice();
    int n_vars = Atb.GetLength();
    float *AtA_ptr = static_cast<float *>(AtA.GetDataPtr());
    float *Atb_ptr = static_cast<float *>(Atb.GetDataPtr());
    float *residual_ptr = static_cast<float *>(residual.GetDataPtr());

    const float *Ti_Cps_ptr =
            static_cast<const float *>(Ti_Cps.GetDataPtr());
    const float *Tj_Cqs_ptr =
            static_cast<const float *>(Tj_Cqs.GetDataPtr());
    const float *Cnormal_ps_ptr =
            static_cast<const float *>(Cnormal_ps.GetDataPtr());
    const float *Ri_Cnormal_ps_ptr =
            static_cast<const float *>(Ri_Cnormal_ps.GetDataPtr());
    const float *RjT_Ri_Cnormal_ps_ptr =
            static_cast<const float *>(RjT_Ri_Cnormal_ps.GetDataPtr());

    const int *cgrid_idx_ps_ptr =
            static_cast<const int *>(cgrid_idx_ps.GetDataPtr());
    const int *cgrid_idx_qs_ptr =
            static_cast<const int *>(cgrid_idx_qs.GetDataPtr());
    const float *cgrid_ratio_ps_ptr =
            static_cast<const float *>(cgrid_ratio_ps.GetDataPtr());
    const float *cgrid_ratio_qs_ptr =
            static_cast<const float *>(cgrid_ratio_qs.GetDataPtr());

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     int64_t workload_idx = id[0];
                     const float *Ti_Cp = Ti_Cps_ptr + 3 * workload_idx;
                     const float *Tj_Cq = Tj_Cqs_ptr + 3 * workload_idx;
                     const float *Cnormal_p =
                             Cnormal_ps_ptr + 3 * workload_idx;
                     const float *Ri_Cnormal_p =
                             Ri_Cnormal_ps_ptr + 3 * workload_idx;
                     const float *RjTRi_Cnormal_p =
                             RjT_Ri_Cnormal_ps_ptr + 3 * workload_idx;

                     const int *cgrid_idx_p =
                             cgrid_idx_ps_ptr + 8 * workload_idx;
                     const int *cgrid_idx_q =
                             cgrid_idx_qs_ptr + 8 * workload_idx;
                     const float *cgrid_ratio_p =
                             cgrid_ratio_ps_ptr + 8 * workload_idx;
                     const float *cgrid_ratio_q =
                             cgrid_ratio_qs_ptr + 8 * workload_idx;

                     float r = (Ti_Cp[0] - Tj_Cq[0]) * Ri_Cnormal_p[0] +
                               (Ti_Cp[1] - Tj_Cq[1]) * Ri_Cnormal_p[1] +
                               (Ti_Cp[2] - Tj_Cq[2]) * Ri_Cnormal_p[2];
                     if (sycl::fabs(r) > threshold) return;

                     float J[60];
                     int idx[60];

                     J[0] = -Tj_Cq[2] * Ri_Cnormal_p[1] +
                            Tj_Cq[1] * Ri_Cnormal_p[2];
                     J[1] = Tj_Cq[2] * Ri_Cnormal_p[0] -
                            Tj_Cq[0] * Ri_Cnormal_p[2];
                     J[2] = -Tj_Cq[1] * Ri_Cnormal_p[0] +
                            Tj_Cq[0] * Ri_Cnormal_p[1];
                     J[3] = Ri_Cnormal_p[0];
                     J[4] = Ri_Cnormal_p[1];
                     J[5] = Ri_Cnormal_p[2];

                     for (int k = 0; k < 6; ++k) {
                         J[k + 6] = -J[k];
                         idx[k + 0] = 6 * i + k;
                         idx[k + 6] = 6 * j + k;
                     }

                     for (int k = 0; k < 8; ++k) {
                         J[12 + k * 3 + 0] =
                                 cgrid_ratio_p[k] * Cnormal_p[0];
                         J[12 + k * 3 + 1] =
                                 cgrid_ratio_p[k] * Cnormal_p[1];
                         J[12 + k * 3 + 2] =
                                 cgrid_ratio_p[k] * Cnormal_p[2];
                         idx[12 + k * 3 + 0] =
                                 6 * n_frags + cgrid_idx_p[k] * 3 + 0;
                         idx[12 + k * 3 + 1] =
                                 6 * n_frags + cgrid_idx_p[k] * 3 + 1;
                         idx[12 + k * 3 + 2] =
                                 6 * n_frags + cgrid_idx_p[k] * 3 + 2;
                     }

                     for (int k = 0; k < 8; ++k) {
                         J[36 + k * 3 + 0] =
                                 -cgrid_ratio_q[k] * RjTRi_Cnormal_p[0];
                         J[36 + k * 3 + 1] =
                                 -cgrid_ratio_q[k] * RjTRi_Cnormal_p[1];
                         J[36 + k * 3 + 2] =
                                 -cgrid_ratio_q[k] * RjTRi_Cnormal_p[2];
                         idx[36 + k * 3 + 0] =
                                 6 * n_frags + cgrid_idx_q[k] * 3 + 0;
                         idx[36 + k * 3 + 1] =
                                 6 * n_frags + cgrid_idx_q[k] * 3 + 1;
                         idx[36 + k * 3 + 2] =
                                 6 * n_frags + cgrid_idx_q[k] * 3 + 2;
                     }

                     for (int ki = 0; ki < 60; ++ki) {
                         for (int kj = 0; kj < 60; ++kj) {
                             float AtA_ij = J[ki] * J[kj];
                             int ij = idx[ki] * n_vars + idx[kj];
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_ptr[ij]) += AtA_ij;
                         }
                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 Atb_ptr[idx[ki]]) += J[ki] * r;
                     }
                     sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                      sycl::memory_scope::device>(
                             *residual_ptr) += r * r;
                 })
            .wait_and_throw();
}

void FillInSLACRegularizerTermSYCL(core::Tensor &AtA,
                                    core::Tensor &Atb,
                                    core::Tensor &residual,
                                    const core::Tensor &grid_idx,
                                    const core::Tensor &grid_nbs_idx,
                                    const core::Tensor &grid_nbs_mask,
                                    const core::Tensor &positions_init,
                                    const core::Tensor &positions_curr,
                                    float weight,
                                    int n_frags,
                                    int anchor_idx) {
    int64_t n = grid_idx.GetLength();
    int64_t n_vars = Atb.GetLength();

    core::Device device = AtA.GetDevice();
    float *AtA_ptr = static_cast<float *>(AtA.GetDataPtr());
    float *Atb_ptr = static_cast<float *>(Atb.GetDataPtr());
    float *residual_ptr = static_cast<float *>(residual.GetDataPtr());

    const int *grid_idx_ptr =
            static_cast<const int *>(grid_idx.GetDataPtr());
    const int *grid_nbs_idx_ptr =
            static_cast<const int *>(grid_nbs_idx.GetDataPtr());
    const bool *grid_nbs_mask_ptr =
            static_cast<const bool *>(grid_nbs_mask.GetDataPtr());
    const float *positions_init_ptr =
            static_cast<const float *>(positions_init.GetDataPtr());
    const float *positions_curr_ptr =
            static_cast<const float *>(positions_curr.GetDataPtr());

    sycl::queue queue =
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
                 sycl::range<1>{(size_t)n},
                 [=](sycl::id<1> id) {
                     int64_t workload_idx = id[0];
                     int idx_i = grid_idx_ptr[workload_idx];

                     const int *idx_nbs =
                             grid_nbs_idx_ptr + 6 * workload_idx;
                     const bool *mask_nbs =
                             grid_nbs_mask_ptr + 6 * workload_idx;

                     float cov[3][3] = {{0}};
                     float U[3][3], V[3][3], S[3];

                     int cnt = 0;
                     for (int k = 0; k < 6; ++k) {
                         if (!mask_nbs[k]) continue;
                         int idx_k = idx_nbs[k];

                         float diff_ik_init[3] = {
                                 positions_init_ptr[idx_i * 3 + 0] -
                                         positions_init_ptr[idx_k * 3 + 0],
                                 positions_init_ptr[idx_i * 3 + 1] -
                                         positions_init_ptr[idx_k * 3 + 1],
                                 positions_init_ptr[idx_i * 3 + 2] -
                                         positions_init_ptr[idx_k * 3 + 2]};
                         float diff_ik_curr[3] = {
                                 positions_curr_ptr[idx_i * 3 + 0] -
                                         positions_curr_ptr[idx_k * 3 + 0],
                                 positions_curr_ptr[idx_i * 3 + 1] -
                                         positions_curr_ptr[idx_k * 3 + 1],
                                 positions_curr_ptr[idx_i * 3 + 2] -
                                         positions_curr_ptr[idx_k * 3 + 2]};

                         for (int ii = 0; ii < 3; ++ii) {
                             for (int jj = 0; jj < 3; ++jj) {
                                 cov[ii][jj] += diff_ik_init[ii] *
                                                diff_ik_curr[jj];
                             }
                         }
                         ++cnt;
                     }

                     if (cnt < 3) return;

                     core::linalg::kernel::svd3x3(*cov, *U, S, *V);

                     float R[3][3];
                     core::linalg::kernel::transpose3x3_(*U);
                     core::linalg::kernel::matmul3x3_3x3(*V, *U, *R);

                     float d = core::linalg::kernel::det3x3(*R);
                     if (d < 0) {
                         U[2][0] = -U[2][0];
                         U[2][1] = -U[2][1];
                         U[2][2] = -U[2][2];
                         core::linalg::kernel::matmul3x3_3x3(*V, *U, *R);
                     }

                     if (idx_i == anchor_idx) {
                         R[0][0] = R[1][1] = R[2][2] = 1;
                         R[0][1] = R[0][2] = R[1][0] = R[1][2] = R[2][0] =
                                 R[2][1] = 0;
                     }

                     for (int k = 0; k < 6; ++k) {
                         if (!mask_nbs[k]) continue;
                         int idx_k = idx_nbs[k];

                         float diff_ik_init[3] = {
                                 positions_init_ptr[idx_i * 3 + 0] -
                                         positions_init_ptr[idx_k * 3 + 0],
                                 positions_init_ptr[idx_i * 3 + 1] -
                                         positions_init_ptr[idx_k * 3 + 1],
                                 positions_init_ptr[idx_i * 3 + 2] -
                                         positions_init_ptr[idx_k * 3 + 2]};
                         float diff_ik_curr[3] = {
                                 positions_curr_ptr[idx_i * 3 + 0] -
                                         positions_curr_ptr[idx_k * 3 + 0],
                                 positions_curr_ptr[idx_i * 3 + 1] -
                                         positions_curr_ptr[idx_k * 3 + 1],
                                 positions_curr_ptr[idx_i * 3 + 2] -
                                         positions_curr_ptr[idx_k * 3 + 2]};
                         float R_diff_ik_curr[3];
                         core::linalg::kernel::matmul3x3_3x1(*R, diff_ik_init,
                                                              R_diff_ik_curr);

                         float local_r[3];
                         local_r[0] = diff_ik_curr[0] - R_diff_ik_curr[0];
                         local_r[1] = diff_ik_curr[1] - R_diff_ik_curr[1];
                         local_r[2] = diff_ik_curr[2] - R_diff_ik_curr[2];

                         int offset_idx_i = 3 * idx_i + 6 * n_frags;
                         int offset_idx_k = 3 * idx_k + 6 * n_frags;

                         sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                                          sycl::memory_scope::device>(
                                 *residual_ptr) +=
                                 weight * (local_r[0] * local_r[0] +
                                           local_r[1] * local_r[1] +
                                           local_r[2] * local_r[2]);

                         for (int axis = 0; axis < 3; ++axis) {
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_ptr[(offset_idx_i + axis) * n_vars +
                                             offset_idx_i + axis]) += weight;
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_ptr[(offset_idx_k + axis) * n_vars +
                                             offset_idx_k + axis]) += weight;
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_ptr[(offset_idx_i + axis) * n_vars +
                                             offset_idx_k + axis]) -= weight;
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     AtA_ptr[(offset_idx_k + axis) * n_vars +
                                             offset_idx_i + axis]) -= weight;

                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     Atb_ptr[offset_idx_i + axis]) +=
                                     weight * local_r[axis];
                             sycl::atomic_ref<
                                     float, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device>(
                                     Atb_ptr[offset_idx_k + axis]) -=
                                     weight * local_r[axis];
                         }
                     }
                 })
            .wait_and_throw();
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
