// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/Registration.h"

#include "open3d/core/Dispatch.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/t/pipelines/kernel/RegistrationImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor ComputePosePointToPlane(const core::Tensor &source_points,
                                     const core::Tensor &target_points,
                                     const core::Tensor &target_normals,
                                     const core::Tensor &correspondence_indices,
                                     const registration::RobustKernel &kernel) {
    const core::Device device = source_points.GetDevice();

    // Pose {6,} tensor [output].
    core::Tensor pose = core::Tensor::Empty({6}, core::Float64, device);

    float residual = 0;
    int inlier_count = 0;

    if (source_points.IsCPU()) {
        ComputePosePointToPlaneCPU(
                source_points.Contiguous(), target_points.Contiguous(),
                target_normals.Contiguous(),
                correspondence_indices.Contiguous(), pose, residual,
                inlier_count, source_points.GetDtype(), device, kernel);
    } else if (source_points.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_points.GetDevice());
        CUDA_CALL(ComputePosePointToPlaneCUDA, source_points.Contiguous(),
                  target_points.Contiguous(), target_normals.Contiguous(),
                  correspondence_indices.Contiguous(), pose, residual,
                  inlier_count, source_points.GetDtype(), device, kernel);
    } else {
        utility::LogError("Unimplemented device.");
    }

    utility::LogDebug("PointToPlane Transform: residual {}, inlier_count {}",
                      residual, inlier_count);

    return pose;
}

core::Tensor ComputePoseColoredICP(const core::Tensor &source_points,
                                   const core::Tensor &source_colors,
                                   const core::Tensor &target_points,
                                   const core::Tensor &target_normals,
                                   const core::Tensor &target_colors,
                                   const core::Tensor &target_color_gradients,
                                   const core::Tensor &correspondence_indices,
                                   const registration::RobustKernel &kernel,
                                   const double &lambda_geometric) {
    const core::Device device = source_points.GetDevice();

    // Pose {6,} tensor [output].
    core::Tensor pose = core::Tensor::Empty({6}, core::Dtype::Float64, device);

    float residual = 0;
    int inlier_count = 0;

    if (source_points.IsCPU()) {
        ComputePoseColoredICPCPU(
                source_points.Contiguous(), source_colors.Contiguous(),
                target_points.Contiguous(), target_normals.Contiguous(),
                target_colors.Contiguous(), target_color_gradients.Contiguous(),
                correspondence_indices.Contiguous(), pose, residual,
                inlier_count, source_points.GetDtype(), device, kernel,
                lambda_geometric);
    } else if (source_points.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(source_points.GetDevice());
        CUDA_CALL(ComputePoseColoredICPCUDA, source_points.Contiguous(),
                  source_colors.Contiguous(), target_points.Contiguous(),
                  target_normals.Contiguous(), target_colors.Contiguous(),
                  target_color_gradients.Contiguous(),
                  correspondence_indices.Contiguous(), pose, residual,
                  inlier_count, source_points.GetDtype(), device, kernel,
                  lambda_geometric);
    } else {
        utility::LogError("Unimplemented device.");
    }

    utility::LogDebug("PointToPlane Transform: residual {}, inlier_count {}",
                      residual, inlier_count);

    return pose;
}

core::Tensor ComputePoseDopplerICP(
        const core::Tensor &source_points,
        const core::Tensor &source_dopplers,
        const core::Tensor &source_directions,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const core::Tensor &correspondence_indices,
        const core::Tensor &current_transform,
        const core::Tensor &transform_vehicle_to_sensor,
        const std::size_t iteration,
        const double period,
        const double lambda_doppler,
        const bool reject_dynamic_outliers,
        const double doppler_outlier_threshold,
        const std::size_t outlier_rejection_min_iteration,
        const std::size_t geometric_robust_loss_min_iteration,
        const std::size_t doppler_robust_loss_min_iteration,
        const registration::RobustKernel &geometric_kernel,
        const registration::RobustKernel &doppler_kernel) {
    const core::Device device = source_points.GetDevice();
    const core::Dtype dtype = source_points.GetDtype();

    // Pose {6,} tensor [ouput].
    core::Tensor output_pose =
            core::Tensor::Empty({6}, core::Dtype::Float64, device);

    float residual = 0;
    int inlier_count = 0;

    // Use robust kernels only after a specified minimum number of iterations.
    const auto kernel_default = registration::RobustKernel(
            registration::RobustKernelMethod::L2Loss, 1.0, 1.0);
    const auto kernel_geometric =
            (iteration >= geometric_robust_loss_min_iteration)
                    ? geometric_kernel
                    : kernel_default;
    const auto kernel_doppler = (iteration >= doppler_robust_loss_min_iteration)
                                        ? doppler_kernel
                                        : kernel_default;

    // Enable outlier rejection based on the current iteration count.
    const bool reject_outliers = reject_dynamic_outliers &&
                                 (iteration >= outlier_rejection_min_iteration);

    // Extract the rotation and translation parts from the matrix.
    const core::Tensor R_S_to_V =
            transform_vehicle_to_sensor
                    .GetItem({core::TensorKey::Slice(0, 3, 1),
                              core::TensorKey::Slice(0, 3, 1)})
                    .Inverse()
                    .Flatten()
                    .To(device, dtype);
    const core::Tensor r_v_to_s_in_V =
            transform_vehicle_to_sensor
                    .GetItem({core::TensorKey::Slice(0, 3, 1),
                              core::TensorKey::Slice(3, 4, 1)})
                    .Flatten()
                    .To(device, dtype);

    // Compute the pose (rotation + translation) vector.
    const core::Tensor state_vector =
            pipelines::kernel::TransformationToPose(current_transform)
                    .To(device, dtype);

    // Compute the linear and angular velocity from the pose vector.
    const core::Tensor w_v_in_V =
            (state_vector.GetItem(core::TensorKey::Slice(0, 3, 1)).Neg() /
             period)
                    .To(device, dtype);
    const core::Tensor v_v_in_V =
            (state_vector.GetItem(core::TensorKey::Slice(3, 6, 1)).Neg() /
             period)
                    .To(device, dtype);

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputePoseDopplerICPCPU(
                source_points.Contiguous(), source_dopplers.Contiguous(),
                source_directions.Contiguous(), target_points.Contiguous(),
                target_normals.Contiguous(),
                correspondence_indices.Contiguous(), output_pose, residual,
                inlier_count, dtype, device, R_S_to_V.Contiguous(),
                r_v_to_s_in_V.Contiguous(), w_v_in_V.Contiguous(),
                v_v_in_V.Contiguous(), period, reject_outliers,
                doppler_outlier_threshold, kernel_geometric, kernel_doppler,
                lambda_doppler);
    } else if (device_type == core::Device::DeviceType::CUDA) {
        CUDA_CALL(ComputePoseDopplerICPCUDA, source_points.Contiguous(),
                  source_dopplers.Contiguous(), source_directions.Contiguous(),
                  target_points.Contiguous(), target_normals.Contiguous(),
                  correspondence_indices.Contiguous(), output_pose, residual,
                  inlier_count, dtype, device, R_S_to_V.Contiguous(),
                  r_v_to_s_in_V.Contiguous(), w_v_in_V.Contiguous(),
                  v_v_in_V.Contiguous(), period, reject_outliers,
                  doppler_outlier_threshold, kernel_geometric, kernel_doppler,
                  lambda_doppler);
    } else {
        utility::LogError("Unimplemented device.");
    }

    utility::LogDebug(
            "DopplerPointToPlane Transform: residual {}, inlier_count {}",
            residual, inlier_count);

    return output_pose;
}

std::tuple<core::Tensor, core::Tensor> ComputeRtPointToPoint(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &correspondence_indices) {
    const core::Device device = source_points.GetDevice();

    // [Output] Rotation and translation tensor of type Float64.
    core::Tensor R, t;

    int inlier_count = 0;

    if (source_points.IsCPU()) {
        // Pointer to point cloud data - indexed according to correspondences.
        ComputeRtPointToPointCPU(
                source_points.Contiguous(), target_points.Contiguous(),
                correspondence_indices.Contiguous(), R, t, inlier_count,
                source_points.GetDtype(), device);
    } else if (source_points.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        core::CUDAScopedDevice scoped_device(source_points.GetDevice());
        // TODO: Implement optimized CUDA reduction kernel.
        core::Tensor valid = correspondence_indices.Ne(-1).Reshape({-1});
        // correpondence_set : (i, corres[i]).

        if (valid.GetLength() == 0) {
            utility::LogError("No valid correspondence present.");
        }

        // source[i] and target[corres[i]] is a correspondence.
        core::Tensor source_indices =
                core::Tensor::Arange(0, source_points.GetShape()[0], 1,
                                     core::Int64, device)
                        .IndexGet({valid});
        // Only take valid indices.
        core::Tensor target_indices =
                correspondence_indices.IndexGet({valid}).Reshape({-1});

        // Number of good correspondences (C).
        inlier_count = source_indices.GetLength();

        core::Tensor source_select = source_points.IndexGet({source_indices});
        core::Tensor target_select = target_points.IndexGet({target_indices});

        // https://ieeexplore.ieee.org/document/88573
        core::Tensor mean_s = source_select.Mean({0}, true);
        core::Tensor mean_t = target_select.Mean({0}, true);

        // Compute linear system on CPU as Float64.
        core::Device host("CPU:0");
        core::Tensor Sxy = (target_select - mean_t)
                                   .T()
                                   .Matmul(source_select - mean_s)
                                   .Div_(static_cast<float>(inlier_count))
                                   .To(host, core::Float64);

        mean_s = mean_s.To(host, core::Float64);
        mean_t = mean_t.To(host, core::Float64);

        core::Tensor U, D, VT;
        std::tie(U, D, VT) = Sxy.SVD();
        core::Tensor S = core::Tensor::Eye(3, core::Float64, host);
        if (U.Det() * (VT.T()).Det() < 0) {
            S[-1][-1] = -1;
        }
        R = U.Matmul(S.Matmul(VT));
        t = mean_t.Reshape({-1}) - R.Matmul(mean_s.T()).Reshape({-1});
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
    return std::make_tuple(R, t);
}

core::Tensor ComputeInformationMatrix(
        const core::Tensor &target_points,
        const core::Tensor &correspondence_indices) {
    const core::Device device = target_points.GetDevice();

    core::Tensor information_matrix =
            core::Tensor::Empty({6, 6}, core::Float64, core::Device("CPU:0"));

    if (target_points.IsCPU()) {
        ComputeInformationMatrixCPU(
                target_points.Contiguous(), correspondence_indices.Contiguous(),
                information_matrix, target_points.GetDtype(), device);
    } else if (target_points.IsCUDA()) {
        core::CUDAScopedDevice scoped_device(target_points.GetDevice());
        CUDA_CALL(ComputeInformationMatrixCUDA, target_points.Contiguous(),
                  correspondence_indices.Contiguous(), information_matrix,
                  target_points.GetDtype(), device);
    } else {
        utility::LogError("Unimplemented device.");
    }

    return information_matrix;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
