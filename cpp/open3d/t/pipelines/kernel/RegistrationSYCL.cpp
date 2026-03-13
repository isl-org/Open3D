// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

static constexpr int kReduceDim =
        29;  // 21 (JtJ) + 6 (Jtr) + 1 (r) + 1 (inlier)

/// Perform a group reduction over a fixed-size private array using
/// sycl::reduce_over_group, then have work item 0 atomically accumulate each
/// group partial sum into the device-side global buffer.
template <int N, typename scalar_t>
static inline void GroupReduceAndAdd(sycl::nd_item<1> item,
                                     const scalar_t (&local_sum)[N],
                                     scalar_t *global_sum_ptr) {
    auto grp = item.get_group();
    for (int k = 0; k < N; ++k) {
        scalar_t grp_val = sycl::reduce_over_group(grp, local_sum[k],
                                                   sycl::plus<scalar_t>{});
        if (item.get_local_id(0) == 0) {
            sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                             sycl::memory_scope::device>(global_sum_ptr[k]) +=
                    grp_val;
        }
    }
}

void ComputePosePointToPlaneSYCL(const core::Tensor &source_points,
                                 const core::Tensor &target_points,
                                 const core::Tensor &target_normals,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &pose,
                                 float &residual,
                                 int &inlier_count,
                                 const core::Dtype &dtype,
                                 const core::Device &device,
                                 const registration::RobustKernel &kernel) {
    const int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({kReduceDim}, dtype, device);

    auto device_props =
            core::sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    sycl::queue queue = device_props.queue;
    const size_t wgs = device_props.max_work_group_size;
    const size_t num_groups = ((size_t)n + wgs - 1) / wgs;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    const scalar_t *source_points_ptr =
                            source_points.GetDataPtr<scalar_t>();
                    const scalar_t *target_points_ptr =
                            target_points.GetDataPtr<scalar_t>();
                    const scalar_t *target_normals_ptr =
                            target_normals.GetDataPtr<scalar_t>();
                    const int64_t *correspondence_indices_ptr =
                            correspondence_indices.GetDataPtr<int64_t>();

                    queue.submit([&](sycl::handler &cgh) {
                             cgh.parallel_for(
                                     sycl::nd_range<1>{num_groups * wgs, wgs},
                                     [=](sycl::nd_item<1> item) {
                                         const int gid =
                                                 item.get_global_id(0);
                                         scalar_t local_sum[kReduceDim] = {};

                                         if (gid < n) {
                                             scalar_t J_ij[6] = {0};
                                             scalar_t r = 0;
                                             const bool valid =
                                                     GetJacobianPointToPlane<
                                                             scalar_t>(
                                                             gid,
                                                             source_points_ptr,
                                                             target_points_ptr,
                                                             target_normals_ptr,
                                                             correspondence_indices_ptr,
                                                             J_ij, r);

                                             if (valid) {
                                                 const scalar_t w =
                                                         GetWeightFromRobustKernel(
                                                                 r);
                                                 int i = 0;
                                                 for (int j = 0; j < 6; ++j) {
                                                     for (int k = 0; k <= j;
                                                          ++k) {
                                                         local_sum[i++] +=
                                                                 J_ij[j] * w *
                                                                 J_ij[k];
                                                     }
                                                     local_sum[21 + j] +=
                                                             J_ij[j] * w * r;
                                                 }
                                                 local_sum[27] += r;
                                                 local_sum[28] +=
                                                         scalar_t(1);
                                             }
                                         }

                                         GroupReduceAndAdd<kReduceDim,
                                                           scalar_t>(
                                                 item, local_sum,
                                                 global_sum_ptr);
                                     });
                         }).wait_and_throw();
                });
    });

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

void ComputePoseColoredICPSYCL(const core::Tensor &source_points,
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
                               const double &lambda_geometric) {
    const int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({kReduceDim}, dtype, device);

    auto device_props =
            core::sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    sycl::queue queue = device_props.queue;
    const size_t wgs = device_props.max_work_group_size;
    const size_t num_groups = ((size_t)n + wgs - 1) / wgs;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t sqrt_lambda_geometric =
                static_cast<scalar_t>(sqrt(lambda_geometric));
        const scalar_t sqrt_lambda_photometric =
                static_cast<scalar_t>(sqrt(1.0 - lambda_geometric));
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_ROBUST_KERNEL_FUNCTION(
                kernel.type_, scalar_t, kernel.scaling_parameter_,
                kernel.shape_parameter_, [&]() {
                    const scalar_t *source_points_ptr =
                            source_points.GetDataPtr<scalar_t>();
                    const scalar_t *source_colors_ptr =
                            source_colors.GetDataPtr<scalar_t>();
                    const scalar_t *target_points_ptr =
                            target_points.GetDataPtr<scalar_t>();
                    const scalar_t *target_normals_ptr =
                            target_normals.GetDataPtr<scalar_t>();
                    const scalar_t *target_colors_ptr =
                            target_colors.GetDataPtr<scalar_t>();
                    const scalar_t *target_color_gradients_ptr =
                            target_color_gradients.GetDataPtr<scalar_t>();
                    const int64_t *correspondence_indices_ptr =
                            correspondence_indices.GetDataPtr<int64_t>();

                    queue.submit([&](sycl::handler &cgh) {
                             cgh.parallel_for(
                                     sycl::nd_range<1>{num_groups * wgs, wgs},
                                     [=](sycl::nd_item<1> item) {
                                         const int gid =
                                                 item.get_global_id(0);
                                         scalar_t local_sum[kReduceDim] = {};

                                         if (gid < n) {
                                             scalar_t J_G[6] = {0},
                                                      J_I[6] = {0};
                                             scalar_t r_G = 0, r_I = 0;

                                             const bool valid =
                                                     GetJacobianColoredICP<
                                                             scalar_t>(
                                                             gid,
                                                             source_points_ptr,
                                                             source_colors_ptr,
                                                             target_points_ptr,
                                                             target_normals_ptr,
                                                             target_colors_ptr,
                                                             target_color_gradients_ptr,
                                                             correspondence_indices_ptr,
                                                             sqrt_lambda_geometric,
                                                             sqrt_lambda_photometric,
                                                             J_G, J_I, r_G,
                                                             r_I);

                                             if (valid) {
                                                 const scalar_t w_G =
                                                         GetWeightFromRobustKernel(
                                                                 r_G);
                                                 const scalar_t w_I =
                                                         GetWeightFromRobustKernel(
                                                                 r_I);

                                                 int i = 0;
                                                 for (int j = 0; j < 6; ++j) {
                                                     for (int k = 0; k <= j;
                                                          ++k) {
                                                         local_sum[i++] +=
                                                                 J_G[j] * w_G *
                                                                         J_G[k] +
                                                                 J_I[j] * w_I *
                                                                         J_I[k];
                                                     }
                                                     local_sum[21 + j] +=
                                                             J_G[j] * w_G *
                                                                     r_G +
                                                             J_I[j] * w_I * r_I;
                                                 }
                                                 local_sum[27] +=
                                                         r_G * r_G + r_I * r_I;
                                                 local_sum[28] += scalar_t(1);
                                             }
                                         }

                                         GroupReduceAndAdd<kReduceDim,
                                                           scalar_t>(
                                                 item, local_sum,
                                                 global_sum_ptr);
                                     });
                         }).wait_and_throw();
                });
    });

    DecodeAndSolve6x6(global_sum, pose, residual, inlier_count);
}

void ComputePoseDopplerICPSYCL(
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
        const double lambda_doppler) {
    const int n = source_points.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({kReduceDim}, dtype, device);
    core::Tensor v_s_in_S = core::Tensor::Zeros({3}, dtype, device);

    auto device_props =
            core::sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    sycl::queue queue = device_props.queue;
    const size_t wgs = device_props.max_work_group_size;
    const size_t num_groups = ((size_t)n + wgs - 1) / wgs;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        const scalar_t sqrt_lambda_geometric =
                sqrt(1.0 - static_cast<scalar_t>(lambda_doppler));
        const scalar_t sqrt_lambda_doppler = sqrt(lambda_doppler);
        const scalar_t sqrt_lambda_doppler_by_dt =
                sqrt_lambda_doppler / static_cast<scalar_t>(period);

        // Pre-compute v_s_in_S using a single-task kernel.
        {
            const scalar_t *R_S_to_V_ptr = R_S_to_V.GetDataPtr<scalar_t>();
            const scalar_t *r_v_to_s_in_V_ptr =
                    r_v_to_s_in_V.GetDataPtr<scalar_t>();
            const scalar_t *w_v_in_V_ptr = w_v_in_V.GetDataPtr<scalar_t>();
            const scalar_t *v_v_in_V_ptr = v_v_in_V.GetDataPtr<scalar_t>();
            scalar_t *v_s_in_S_ptr = v_s_in_S.GetDataPtr<scalar_t>();
            queue.single_task([=]() {
                     PreComputeForDopplerICP(R_S_to_V_ptr, r_v_to_s_in_V_ptr,
                                            w_v_in_V_ptr, v_v_in_V_ptr,
                                            v_s_in_S_ptr);
                 }).wait_and_throw();
        }

        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();

        DISPATCH_DUAL_ROBUST_KERNEL_FUNCTION(
                scalar_t, kernel_geometric.type_,
                kernel_geometric.scaling_parameter_, kernel_doppler.type_,
                kernel_doppler.scaling_parameter_, [&]() {
                    const scalar_t *source_points_ptr =
                            source_points.GetDataPtr<scalar_t>();
                    const scalar_t *source_dopplers_ptr =
                            source_dopplers.GetDataPtr<scalar_t>();
                    const scalar_t *source_directions_ptr =
                            source_directions.GetDataPtr<scalar_t>();
                    const scalar_t *target_points_ptr =
                            target_points.GetDataPtr<scalar_t>();
                    const scalar_t *target_normals_ptr =
                            target_normals.GetDataPtr<scalar_t>();
                    const int64_t *correspondence_indices_ptr =
                            correspondence_indices.GetDataPtr<int64_t>();
                    const scalar_t *R_S_to_V_ptr =
                            R_S_to_V.GetDataPtr<scalar_t>();
                    const scalar_t *r_v_to_s_in_V_ptr =
                            r_v_to_s_in_V.GetDataPtr<scalar_t>();
                    const scalar_t *v_s_in_S_ptr =
                            v_s_in_S.GetDataPtr<scalar_t>();

                    queue.submit([&](sycl::handler &cgh) {
                             cgh.parallel_for(
                                     sycl::nd_range<1>{num_groups * wgs, wgs},
                                     [=](sycl::nd_item<1> item) {
                                         const int gid =
                                                 item.get_global_id(0);
                                         scalar_t local_sum[kReduceDim] = {};

                                         if (gid < n) {
                                             scalar_t J_G[6] = {0},
                                                      J_D[6] = {0};
                                             scalar_t r_G = 0, r_D = 0;

                                             const bool valid =
                                                     GetJacobianDopplerICP<
                                                             scalar_t>(
                                                             gid,
                                                             source_points_ptr,
                                                             source_dopplers_ptr,
                                                             source_directions_ptr,
                                                             target_points_ptr,
                                                             target_normals_ptr,
                                                             correspondence_indices_ptr,
                                                             R_S_to_V_ptr,
                                                             r_v_to_s_in_V_ptr,
                                                             v_s_in_S_ptr,
                                                             reject_dynamic_outliers,
                                                             static_cast<scalar_t>(
                                                                     doppler_outlier_threshold),
                                                             sqrt_lambda_geometric,
                                                             sqrt_lambda_doppler,
                                                             sqrt_lambda_doppler_by_dt,
                                                             J_G, J_D, r_G,
                                                             r_D);

                                             if (valid) {
                                                 const scalar_t w_G =
                                                         GetWeightFromRobustKernelFirst(
                                                                 r_G);
                                                 const scalar_t w_D =
                                                         GetWeightFromRobustKernelSecond(
                                                                 r_D);

                                                 int i = 0;
                                                 for (int j = 0; j < 6; ++j) {
                                                     for (int k = 0; k <= j;
                                                          ++k) {
                                                         local_sum[i++] +=
                                                                 J_G[j] * w_G *
                                                                         J_G[k] +
                                                                 J_D[j] * w_D *
                                                                         J_D[k];
                                                     }
                                                     local_sum[21 + j] +=
                                                             J_G[j] * w_G *
                                                                     r_G +
                                                             J_D[j] * w_D * r_D;
                                                 }
                                                 local_sum[27] +=
                                                         r_G * r_G + r_D * r_D;
                                                 local_sum[28] += scalar_t(1);
                                             }
                                         }

                                         GroupReduceAndAdd<kReduceDim,
                                                           scalar_t>(
                                                 item, local_sum,
                                                 global_sum_ptr);
                                     });
                         }).wait_and_throw();
                });
    });

    DecodeAndSolve6x6(global_sum, output_pose, residual, inlier_count);
}

void ComputeInformationMatrixSYCL(const core::Tensor &target_points,
                                  const core::Tensor &correspondence_indices,
                                  core::Tensor &information_matrix,
                                  const core::Dtype &dtype,
                                  const core::Device &device) {
    const int n = correspondence_indices.GetLength();

    core::Tensor global_sum = core::Tensor::Zeros({21}, dtype, device);

    auto device_props =
            core::sy::SYCLContext::GetInstance().GetDeviceProperties(device);
    sycl::queue queue = device_props.queue;
    const size_t wgs = device_props.max_work_group_size;
    const size_t num_groups = ((size_t)n + wgs - 1) / wgs;

    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t *global_sum_ptr = global_sum.GetDataPtr<scalar_t>();
        const scalar_t *target_points_ptr =
                target_points.GetDataPtr<scalar_t>();
        const int64_t *correspondence_indices_ptr =
                correspondence_indices.GetDataPtr<int64_t>();

        queue.submit([&](sycl::handler &cgh) {
                 cgh.parallel_for(
                         sycl::nd_range<1>{num_groups * wgs, wgs},
                         [=](sycl::nd_item<1> item) {
                             const int gid = item.get_global_id(0);
                             scalar_t local_sum[21] = {};

                             if (gid < n) {
                                 scalar_t J_x[6] = {0}, J_y[6] = {0},
                                          J_z[6] = {0};
                                 const bool valid =
                                         GetInformationJacobians<scalar_t>(
                                                 gid, target_points_ptr,
                                                 correspondence_indices_ptr,
                                                 J_x, J_y, J_z);

                                 if (valid) {
                                     int i = 0;
                                     for (int j = 0; j < 6; ++j) {
                                         for (int k = 0; k <= j; ++k) {
                                             local_sum[i++] +=
                                                     J_x[j] * J_x[k] +
                                                     J_y[j] * J_y[k] +
                                                     J_z[j] * J_z[k];
                                         }
                                     }
                                 }
                             }

                             GroupReduceAndAdd<21, scalar_t>(item, local_sum,
                                                             global_sum_ptr);
                         });
             }).wait_and_throw();

        const core::Device host(core::Device("CPU:0"));
        core::Tensor global_sum_cpu = global_sum.To(host, core::Float64);
        double *sum_ptr = global_sum_cpu.GetDataPtr<double>();

        // Information matrix is on CPU of type Float64.
        double *GTG_ptr = information_matrix.GetDataPtr<double>();

        int i = 0;
        for (int j = 0; j < 6; j++) {
            for (int k = 0; k <= j; k++) {
                GTG_ptr[j * 6 + k] = GTG_ptr[k * 6 + j] = sum_ptr[i];
                ++i;
            }
        }
    });
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
