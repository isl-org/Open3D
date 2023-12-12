// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.

#pragma once

#include <cmath>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/kernel/Matrix.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

#ifndef __CUDACC__
using std::abs;
#endif

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const core::Tensor &source_points,
                                const core::Tensor &target_points,
                                const core::Tensor &target_normals,
                                const core::Tensor &correspondence_indices,
                                core::Tensor &pose,
                                float &residual,
                                int &inlier_count,
                                const core::Dtype &dtype,
                                const core::Device &device,
                                const registration::RobustKernel &kernel);

void ComputePoseColoredICPCPU(const core::Tensor &source_points,
                              const core::Tensor &source_colors,
                              const core::Tensor &target_points,
                              const core::Tensor &target_normals,
                              const core::Tensor &target_colors,
                              const core::Tensor &target_color_gradients,
                              const core::Tensor &correspondence_indices,
                              core::Tensor &pose,
                              float &residual,
                              int &inlier_count,
                              const core::Dtype &dtype,
                              const core::Device &device,
                              const registration::RobustKernel &kernel,
                              const double &lambda_geometric);

void ComputePoseDopplerICPCPU(
        const core::Tensor &source_points,
        const core::Tensor &source_dopplers,
        const core::Tensor &source_directions,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const core::Tensor &correspondence_indices,
        core::Tensor &output_pose,
        float &residual,
        int &inlier_count,
        const core::Dtype &dtype,
        const core::Device &device,
        const core::Tensor &R_S_to_V,
        const core::Tensor &r_v_to_s_in_V,
        const core::Tensor &w_v_in_V,
        const core::Tensor &v_v_in_V,
        const double period,
        const bool reject_dynamic_outliers,
        const double doppler_outlier_threshold,
        const registration::RobustKernel &kernel_geometric,
        const registration::RobustKernel &kernel_doppler,
        const double lambda_doppler);

#ifdef BUILD_CUDA_MODULE
void ComputePosePointToPlaneCUDA(const core::Tensor &source_points,
                                 const core::Tensor &target_points,
                                 const core::Tensor &target_normals,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &pose,
                                 float &residual,
                                 int &inlier_count,
                                 const core::Dtype &dtype,
                                 const core::Device &device,
                                 const registration::RobustKernel &kernel);

void ComputePoseColoredICPCUDA(const core::Tensor &source_points,
                               const core::Tensor &source_colors,
                               const core::Tensor &target_points,
                               const core::Tensor &target_normals,
                               const core::Tensor &target_colors,
                               const core::Tensor &target_color_gradients,
                               const core::Tensor &correspondence_indices,
                               core::Tensor &pose,
                               float &residual,
                               int &inlier_count,
                               const core::Dtype &dtype,
                               const core::Device &device,
                               const registration::RobustKernel &kernel,
                               const double &lambda_geometric);

void ComputePoseDopplerICPCUDA(
        const core::Tensor &source_points,
        const core::Tensor &source_dopplers,
        const core::Tensor &source_directions,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const core::Tensor &correspondence_indices,
        core::Tensor &output_pose,
        float &residual,
        int &inlier_count,
        const core::Dtype &dtype,
        const core::Device &device,
        const core::Tensor &R_S_to_V,
        const core::Tensor &r_v_to_s_in_V,
        const core::Tensor &w_v_in_V,
        const core::Tensor &v_v_in_V,
        const double period,
        const bool reject_dynamic_outliers,
        const double doppler_outlier_threshold,
        const registration::RobustKernel &kernel_geometric,
        const registration::RobustKernel &kernel_doppler,
        const double lambda_doppler);
#endif

void ComputeRtPointToPointCPU(const core::Tensor &source_points,
                              const core::Tensor &target_points,
                              const core::Tensor &correspondence_indices,
                              core::Tensor &R,
                              core::Tensor &t,
                              int &inlier_count,
                              const core::Dtype &dtype,
                              const core::Device &device);

void ComputeInformationMatrixCPU(const core::Tensor &target_points,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &information_matrix,
                                 const core::Dtype &dtype,
                                 const core::Device &device);

#ifdef BUILD_CUDA_MODULE
void ComputeInformationMatrixCUDA(const core::Tensor &target_points,
                                  const core::Tensor &correspondence_indices,
                                  core::Tensor &information_matrix,
                                  const core::Dtype &dtype,
                                  const core::Device &device);
#endif

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int64_t workload_idx,
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        scalar_t *J_ij,
        scalar_t &r) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];
    const int64_t source_idx = 3 * workload_idx;

    const scalar_t &sx = source_points_ptr[source_idx + 0];
    const scalar_t &sy = source_points_ptr[source_idx + 1];
    const scalar_t &sz = source_points_ptr[source_idx + 2];
    const scalar_t &tx = target_points_ptr[target_idx + 0];
    const scalar_t &ty = target_points_ptr[target_idx + 1];
    const scalar_t &tz = target_points_ptr[target_idx + 2];
    const scalar_t &nx = target_normals_ptr[target_idx + 0];
    const scalar_t &ny = target_normals_ptr[target_idx + 1];
    const scalar_t &nz = target_normals_ptr[target_idx + 2];

    r = (sx - tx) * nx + (sy - ty) * ny + (sz - tz) * nz;

    J_ij[0] = nz * sy - ny * sz;
    J_ij[1] = nx * sz - nz * sx;
    J_ij[2] = ny * sx - nx * sy;
    J_ij[3] = nx;
    J_ij[4] = ny;
    J_ij[5] = nz;

    return true;
}

template bool GetJacobianPointToPlane(int64_t workload_idx,
                                      const float *source_points_ptr,
                                      const float *target_points_ptr,
                                      const float *target_normals_ptr,
                                      const int64_t *correspondence_indices,
                                      float *J_ij,
                                      float &r);

template bool GetJacobianPointToPlane(int64_t workload_idx,
                                      const double *source_points_ptr,
                                      const double *target_points_ptr,
                                      const double *target_normals_ptr,
                                      const int64_t *correspondence_indices,
                                      double *J_ij,
                                      double &r);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetJacobianColoredICP(
        const int64_t workload_idx,
        const scalar_t *source_points_ptr,
        const scalar_t *source_colors_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const scalar_t *target_colors_ptr,
        const scalar_t *target_color_gradients_ptr,
        const int64_t *correspondence_indices,
        const scalar_t &sqrt_lambda_geometric,
        const scalar_t &sqrt_lambda_photometric,
        scalar_t *J_G,
        scalar_t *J_I,
        scalar_t &r_G,
        scalar_t &r_I) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];
    const int64_t source_idx = 3 * workload_idx;

    const scalar_t vs[3] = {source_points_ptr[source_idx],
                            source_points_ptr[source_idx + 1],
                            source_points_ptr[source_idx + 2]};

    const scalar_t vt[3] = {target_points_ptr[target_idx],
                            target_points_ptr[target_idx + 1],
                            target_points_ptr[target_idx + 2]};

    const scalar_t nt[3] = {target_normals_ptr[target_idx],
                            target_normals_ptr[target_idx + 1],
                            target_normals_ptr[target_idx + 2]};

    const scalar_t d = (vs[0] - vt[0]) * nt[0] + (vs[1] - vt[1]) * nt[1] +
                       (vs[2] - vt[2]) * nt[2];

    J_G[0] = sqrt_lambda_geometric * (-vs[2] * nt[1] + vs[1] * nt[2]);
    J_G[1] = sqrt_lambda_geometric * (vs[2] * nt[0] - vs[0] * nt[2]);
    J_G[2] = sqrt_lambda_geometric * (-vs[1] * nt[0] + vs[0] * nt[1]);
    J_G[3] = sqrt_lambda_geometric * nt[0];
    J_G[4] = sqrt_lambda_geometric * nt[1];
    J_G[5] = sqrt_lambda_geometric * nt[2];
    r_G = sqrt_lambda_geometric * d;

    const scalar_t vs_proj[3] = {vs[0] - d * nt[0], vs[1] - d * nt[1],
                                 vs[2] - d * nt[2]};

    const scalar_t intensity_source =
            (source_colors_ptr[source_idx] + source_colors_ptr[source_idx + 1] +
             source_colors_ptr[source_idx + 2]) /
            3.0;

    const scalar_t intensity_target =
            (target_colors_ptr[target_idx] + target_colors_ptr[target_idx + 1] +
             target_colors_ptr[target_idx + 2]) /
            3.0;

    const scalar_t dit[3] = {target_color_gradients_ptr[target_idx],
                             target_color_gradients_ptr[target_idx + 1],
                             target_color_gradients_ptr[target_idx + 2]};

    const scalar_t is_proj = dit[0] * (vs_proj[0] - vt[0]) +
                             dit[1] * (vs_proj[1] - vt[1]) +
                             dit[2] * (vs_proj[2] - vt[2]) + intensity_target;

    const scalar_t s = dit[0] * nt[0] + dit[1] * nt[1] + dit[2] * nt[2];
    const scalar_t ditM[3] = {s * nt[0] - dit[0], s * nt[1] - dit[1],
                              s * nt[2] - dit[2]};

    J_I[0] = sqrt_lambda_photometric * (-vs[2] * ditM[1] + vs[1] * ditM[2]);
    J_I[1] = sqrt_lambda_photometric * (vs[2] * ditM[0] - vs[0] * ditM[2]);
    J_I[2] = sqrt_lambda_photometric * (-vs[1] * ditM[0] + vs[0] * ditM[1]);
    J_I[3] = sqrt_lambda_photometric * ditM[0];
    J_I[4] = sqrt_lambda_photometric * ditM[1];
    J_I[5] = sqrt_lambda_photometric * ditM[2];
    r_I = sqrt_lambda_photometric * (intensity_source - is_proj);

    return true;
}

template bool GetJacobianColoredICP(const int64_t workload_idx,
                                    const float *source_points_ptr,
                                    const float *source_colors_ptr,
                                    const float *target_points_ptr,
                                    const float *target_normals_ptr,
                                    const float *target_colors_ptr,
                                    const float *target_color_gradients_ptr,
                                    const int64_t *correspondence_indices,
                                    const float &sqrt_lambda_geometric,
                                    const float &sqrt_lambda_photometric,
                                    float *J_G,
                                    float *J_I,
                                    float &r_G,
                                    float &r_I);

template bool GetJacobianColoredICP(const int64_t workload_idx,
                                    const double *source_points_ptr,
                                    const double *source_colors_ptr,
                                    const double *target_points_ptr,
                                    const double *target_normals_ptr,
                                    const double *target_colors_ptr,
                                    const double *target_color_gradients_ptr,
                                    const int64_t *correspondence_indices,
                                    const double &sqrt_lambda_geometric,
                                    const double &sqrt_lambda_photometric,
                                    double *J_G,
                                    double *J_I,
                                    double &r_G,
                                    double &r_I);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline void PreComputeForDopplerICP(
        const scalar_t *R_S_to_V,
        const scalar_t *r_v_to_s_in_V,
        const scalar_t *w_v_in_V,
        const scalar_t *v_v_in_V,
        scalar_t *v_s_in_S) {
    // Compute v_s_in_V = v_v_in_V + w_v_in_V.cross(r_v_to_s_in_V).
    scalar_t v_s_in_V[3] = {0};
    core::linalg::kernel::cross_3x1(w_v_in_V, r_v_to_s_in_V, v_s_in_V);
    v_s_in_V[0] += v_v_in_V[0];
    v_s_in_V[1] += v_v_in_V[1];
    v_s_in_V[2] += v_v_in_V[2];

    // Compute v_s_in_S = R_S_to_V * v_s_in_V.
    core::linalg::kernel::matmul3x3_3x1(R_S_to_V, v_s_in_V, v_s_in_S);
}

template void PreComputeForDopplerICP(const float *R_S_to_V,
                                      const float *r_v_to_s_in_V,
                                      const float *w_v_in_V,
                                      const float *v_v_in_V,
                                      float *v_s_in_S);

template void PreComputeForDopplerICP(const double *R_S_to_V,
                                      const double *r_v_to_s_in_V,
                                      const double *w_v_in_V,
                                      const double *v_v_in_V,
                                      double *v_s_in_S);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetJacobianDopplerICP(
        const int64_t workload_idx,
        const scalar_t *source_points_ptr,
        const scalar_t *source_dopplers_ptr,
        const scalar_t *source_directions_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        const scalar_t *R_S_to_V,
        const scalar_t *r_v_to_s_in_V,
        const scalar_t *v_s_in_S,
        const bool reject_dynamic_outliers,
        const scalar_t doppler_outlier_threshold,
        const scalar_t &sqrt_lambda_geometric,
        const scalar_t &sqrt_lambda_doppler,
        const scalar_t &sqrt_lambda_doppler_by_dt,
        scalar_t *J_G,
        scalar_t *J_D,
        scalar_t &r_G,
        scalar_t &r_D) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];
    const int64_t source_idx = 3 * workload_idx;

    const scalar_t &doppler_in_S = source_dopplers_ptr[workload_idx];

    const scalar_t ds_in_V[3] = {source_directions_ptr[source_idx],
                                 source_directions_ptr[source_idx + 1],
                                 source_directions_ptr[source_idx + 2]};

    // Compute predicted Doppler velocity (in sensor frame).
    scalar_t ds_in_S[3] = {0};
    core::linalg::kernel::matmul3x3_3x1(R_S_to_V, ds_in_V, ds_in_S);
    const scalar_t doppler_pred_in_S =
            -core::linalg::kernel::dot_3x1(ds_in_S, v_s_in_S);

    // Compute Doppler error.
    const double doppler_error = doppler_in_S - doppler_pred_in_S;

    // Dynamic point outlier rejection.
    if (reject_dynamic_outliers &&
        abs(doppler_error) > doppler_outlier_threshold) {
        // Jacobian and residual are set to 0 by default.
        return true;
    }

    // Compute Doppler residual and Jacobian.
    scalar_t J_D_w[3] = {0};
    core::linalg::kernel::cross_3x1(ds_in_V, r_v_to_s_in_V, J_D_w);
    J_D[0] = sqrt_lambda_doppler_by_dt * J_D_w[0];
    J_D[1] = sqrt_lambda_doppler_by_dt * J_D_w[1];
    J_D[2] = sqrt_lambda_doppler_by_dt * J_D_w[2];
    J_D[3] = sqrt_lambda_doppler_by_dt * -ds_in_V[0];
    J_D[4] = sqrt_lambda_doppler_by_dt * -ds_in_V[1];
    J_D[5] = sqrt_lambda_doppler_by_dt * -ds_in_V[2];
    r_D = sqrt_lambda_doppler * doppler_error;

    const scalar_t ps[3] = {source_points_ptr[source_idx],
                            source_points_ptr[source_idx + 1],
                            source_points_ptr[source_idx + 2]};

    const scalar_t pt[3] = {target_points_ptr[target_idx],
                            target_points_ptr[target_idx + 1],
                            target_points_ptr[target_idx + 2]};

    const scalar_t nt[3] = {target_normals_ptr[target_idx],
                            target_normals_ptr[target_idx + 1],
                            target_normals_ptr[target_idx + 2]};

    // Compute geometric point-to-plane error.
    const scalar_t p2p_error = (ps[0] - pt[0]) * nt[0] +
                               (ps[1] - pt[1]) * nt[1] +
                               (ps[2] - pt[2]) * nt[2];

    // Compute geometric point-to-plane residual and Jacobian.
    J_G[0] = sqrt_lambda_geometric * (-ps[2] * nt[1] + ps[1] * nt[2]);
    J_G[1] = sqrt_lambda_geometric * (ps[2] * nt[0] - ps[0] * nt[2]);
    J_G[2] = sqrt_lambda_geometric * (-ps[1] * nt[0] + ps[0] * nt[1]);
    J_G[3] = sqrt_lambda_geometric * nt[0];
    J_G[4] = sqrt_lambda_geometric * nt[1];
    J_G[5] = sqrt_lambda_geometric * nt[2];
    r_G = sqrt_lambda_geometric * p2p_error;

    return true;
}

template bool GetJacobianDopplerICP(const int64_t workload_idx,
                                    const float *source_points_ptr,
                                    const float *source_dopplers_ptr,
                                    const float *source_directions_ptr,
                                    const float *target_points_ptr,
                                    const float *target_normals_ptr,
                                    const int64_t *correspondence_indices,
                                    const float *R_S_to_V,
                                    const float *r_v_to_s_in_V,
                                    const float *v_s_in_S,
                                    const bool reject_dynamic_outliers,
                                    const float doppler_outlier_threshold,
                                    const float &sqrt_lambda_geometric,
                                    const float &sqrt_lambda_doppler,
                                    const float &sqrt_lambda_doppler_by_dt,
                                    float *J_G,
                                    float *J_D,
                                    float &r_G,
                                    float &r_D);

template bool GetJacobianDopplerICP(const int64_t workload_idx,
                                    const double *source_points_ptr,
                                    const double *source_dopplers_ptr,
                                    const double *source_directions_ptr,
                                    const double *target_points_ptr,
                                    const double *target_normals_ptr,
                                    const int64_t *correspondence_indices,
                                    const double *R_S_to_V,
                                    const double *r_v_to_s_in_V,
                                    const double *v_s_in_S,
                                    const bool reject_dynamic_outliers,
                                    const double doppler_outlier_threshold,
                                    const double &sqrt_lambda_geometric,
                                    const double &sqrt_lambda_doppler,
                                    const double &sqrt_lambda_doppler_by_dt,
                                    double *J_G,
                                    double *J_D,
                                    double &r_G,
                                    double &r_D);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetInformationJacobians(
        int64_t workload_idx,
        const scalar_t *target_points_ptr,
        const int64_t *correspondence_indices,
        scalar_t *jacobian_x,
        scalar_t *jacobian_y,
        scalar_t *jacobian_z) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];

    jacobian_x[0] = jacobian_x[4] = jacobian_x[5] = 0.0;
    jacobian_x[1] = target_points_ptr[target_idx + 2];
    jacobian_x[2] = -target_points_ptr[target_idx + 1];
    jacobian_x[3] = 1.0;

    jacobian_y[1] = jacobian_y[3] = jacobian_y[5] = 0.0;
    jacobian_y[0] = -target_points_ptr[target_idx + 2];
    jacobian_y[2] = target_points_ptr[target_idx];
    jacobian_y[4] = 1.0;

    jacobian_z[2] = jacobian_z[3] = jacobian_z[4] = 0.0;
    jacobian_z[0] = target_points_ptr[target_idx + 1];
    jacobian_z[1] = -target_points_ptr[target_idx];
    jacobian_z[5] = 1.0;

    return true;
}

template bool GetInformationJacobians(int64_t workload_idx,
                                      const float *target_points_ptr,
                                      const int64_t *correspondence_indices,
                                      float *jacobian_x,
                                      float *jacobian_y,
                                      float *jacobian_z);

template bool GetInformationJacobians(int64_t workload_idx,
                                      const double *target_points_ptr,
                                      const int64_t *correspondence_indices,
                                      double *jacobian_x,
                                      double *jacobian_y,
                                      double *jacobian_z);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
