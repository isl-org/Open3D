// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.
#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void ComputeOdometryResultPointToPlaneCPU(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& source_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

void ComputeOdometryResultIntensityCPU(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float intensity_huber_delta);

void ComputeOdometryResultHybridCPU(const core::Tensor& source_depth,
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
                                    float& inlier_residual,
                                    int& inlier_count,
                                    float depth_outlier_trunc,
                                    const float depth_huber_delta,
                                    const float intensity_huber_delta);

void ComputeOdometryInformationMatrixCPU(const core::Tensor& source_depth,
                                         const core::Tensor& target_depth,
                                         const core::Tensor& intrinsic,
                                         const core::Tensor& source_to_target,
                                         const float depth_outlier_trunc,
                                         core::Tensor& information);

#ifdef BUILD_CUDA_MODULE

void ComputeOdometryResultPointToPlaneCUDA(
        const core::Tensor& source_vertex_map,
        const core::Tensor& target_vertex_map,
        const core::Tensor& source_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float depth_huber_delta);

void ComputeOdometryResultIntensityCUDA(
        const core::Tensor& source_depth,
        const core::Tensor& target_depth,
        const core::Tensor& source_intensity,
        const core::Tensor& target_intensity,
        const core::Tensor& target_intensity_dx,
        const core::Tensor& target_intensity_dy,
        const core::Tensor& source_vertex_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        core::Tensor& delta,
        float& inlier_residual,
        int& inlier_count,
        const float depth_outlier_trunc,
        const float intensity_huber_delta);

void ComputeOdometryResultHybridCUDA(const core::Tensor& source_depth,
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
                                     float& inlier_residual,
                                     int& inlier_count,
                                     const float depth_outlier_trunc,
                                     const float depth_huber_delta,
                                     const float intensity_huber_delta);

void ComputeOdometryInformationMatrixCUDA(const core::Tensor& source_depth,
                                          const core::Tensor& target_depth,
                                          const core::Tensor& intrinsic,
                                          const core::Tensor& source_to_target,
                                          const float square_dist_thr,
                                          core::Tensor& information);
#endif

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
