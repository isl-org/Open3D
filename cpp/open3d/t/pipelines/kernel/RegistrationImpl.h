// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

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
