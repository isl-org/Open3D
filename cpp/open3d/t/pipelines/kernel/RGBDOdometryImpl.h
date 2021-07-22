// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#endif

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
