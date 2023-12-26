// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

/// \brief Computes pose for point to plane registration method.
///
/// \param source_positions source point positions of Float32 or Float64 dtype.
/// \param target_positions target point positions of same dtype as source point
/// positions.
/// \param target_normals target point normals of same dtype as source point
/// positions.
/// \param correspondence_indices Tensor of type Int64 containing indices of
/// corresponding target positions, where the value is the target index and the
/// index of the value itself is the source index. It contains -1 as value at
/// index with no correspondence.
/// \param kernel statistical robust kernel for outlier rejection.
/// \return Pose [alpha beta gamma, tx, ty, tz], a shape {6} tensor of dtype
/// Float64, where alpha, beta, gamma are the Euler angles in the ZYX order.
core::Tensor ComputePosePointToPlane(const core::Tensor &source_positions,
                                     const core::Tensor &target_positions,
                                     const core::Tensor &target_normals,
                                     const core::Tensor &correspondence_indices,
                                     const registration::RobustKernel &kernel);

/// \brief Computes pose for colored-icp registration method.
///
/// \param source_positions source point positions of Float32 or Float64 dtype.
/// \param source_colors source point colors of same dtype as source point
/// positions.
/// \param target_positions target point positions of same dtype as source point
/// positions.
/// \param target_normals target point normals of same dtype as source point
/// positions.
/// \param target_colors target point colors of same dtype as source point
/// positions.
/// \param target_color_gradients targets point color gradients of same dtype as
/// source point positions.
/// \param correspondence_indices Tensor of type Int64 containing indices of
/// corresponding target positions, where the value is the target index and the
/// index of the value itself is the source index. It contains -1 as value at
/// index with no correspondence.
/// \param kernel statistical robust kernel for outlier rejection.
/// \param lambda_geometric `λ ∈ [0,1]` in the overall energy `λEG + (1−λ)EC`.
/// Refer the documentation of Colored-ICP for more information.
/// \return Pose [alpha beta gamma, tx, ty, tz], a shape {6} tensor of dtype
/// Float64, where alpha, beta, gamma are the Euler angles in the ZYX order.
core::Tensor ComputePoseColoredICP(const core::Tensor &source_positions,
                                   const core::Tensor &source_colors,
                                   const core::Tensor &target_positions,
                                   const core::Tensor &target_normals,
                                   const core::Tensor &target_colors,
                                   const core::Tensor &target_color_gradients,
                                   const core::Tensor &correspondence_indices,
                                   const registration::RobustKernel &kernel,
                                   const double &lambda_geometric);

/// \brief Computes pose for DopplerICP registration method.
///
/// \param source_points source point positions of Float32 or Float64 dtype.
/// \param source_dopplers source point Dopplers of same dtype as source point
/// positions.
/// \param source_directions source point direction of same dtype as source
/// point positions.
/// \param target_points target point positions of same dtype as source point
/// positions.
/// \param target_normals target point normals of same dtype as source point
/// positions.
/// \param correspondence_indices Tensor of type Int64 containing indices of
/// corresponding target positions, where the value is the target index and the
/// index of the value itself is the source index. It contains -1 as value at
/// index with no correspondence.
/// \param current_transform The current pose estimate of ICP.
/// \param transform_vehicle_to_sensor The 4x4 extrinsic transformation matrix
/// between the vehicle and the sensor frames.
/// \param iteration current iteration number of the ICP algorithm.
/// \param period Time period (in seconds) between the source and the target
/// point clouds.
/// \param lambda_doppler weight for the Doppler objective.
/// \param reject_dynamic_outliers Whether or not to prune dynamic point
/// outlier correspondences.
/// \param doppler_outlier_threshold Correspondences with Doppler error
/// greater than this threshold are rejected from optimization.
/// \param outlier_rejection_min_iteration Number of iterations of ICP after
/// which outlier rejection is enabled.
/// \param geometric_robust_loss_min_iteration Number of iterations of ICP
/// after which robust loss for geometric term kicks in.
/// \param doppler_robust_loss_min_iteration Number of iterations of ICP
/// after which robust loss for Doppler term kicks in.
/// \param geometric_kernel statistical robust kernel for outlier rejection.
/// \param doppler_kernel statistical robust kernel for outlier rejection.
/// \return Pose [alpha beta gamma, tx, ty, tz], a shape {6} tensor of dtype
/// Float64, where alpha, beta, gamma are the Euler angles in the ZYX order.
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
        const registration::RobustKernel &doppler_kernel);

/// \brief Computes (R) Rotation {3,3} and (t) translation {3,}
/// for point to point registration method.
///
/// \param source_positions source point positions of Float32 or Float64 dtype.
/// \param target_positions target point positions of same dtype as source point
/// positions.
/// \param correspondence_indices Tensor of type Int64 containing indices of
/// corresponding target positions, where the value is the target index and the
/// index of the value itself is the source index. It contains -1 as value at
/// index with no correspondence.
/// \return tuple of (R, t). [Dtype: Float64].
std::tuple<core::Tensor, core::Tensor> ComputeRtPointToPoint(
        const core::Tensor &source_positions,
        const core::Tensor &target_positions,
        const core::Tensor &correspondence_indices);

/// \brief Computes `Information Matrix` of shape {6, 6}, of dtype `Float64` on
/// device `CPU:0`, from the target point cloud and correspondence indices
/// w.r.t. target point cloud.
/// Only target positions and correspondence indices are required.
///
/// \param target_positions The target point positions.
/// \param correspondence_indices Tensor of type Int64 containing indices of
/// corresponding target positions, where the value is the target index and the
/// index of the value itself is the source index. It contains -1 as value
/// at index with no correspondence.
core::Tensor ComputeInformationMatrix(
        const core::Tensor &target_positions,
        const core::Tensor &correspondence_indices);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
