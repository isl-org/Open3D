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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

enum class LossType {
    PointToPlane,
    DirectIntensity,
    DirectHybrid,
};

/// Note: all the 4x4 transformation in this file, from params to returns, are
/// Float64. Only convert to Float32 in kernel calls.

/// Create an RGBD image pyramid given the original source and target and
/// perform hierarchical odometry.
/// Used for offline odometry where we do not care performance (too much) and
/// not reuse vertex/normal map computed before.
core::Tensor RGBDOdometryMultiScale(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0")),
        float depth_factor = 1000.0f,
        float depth_diff = 0.07f,
        const std::vector<int>& iterations = {10, 5, 3},
        const LossType method = LossType::PointToPlane);

/// Create a vertex map (image) from a depth image. Useful for point-to-plane
/// odometry and point-to-point.
core::Tensor CreateVertexMap(const t::geometry::Image& depth,
                             const core::Tensor& intrinsics,
                             float depth_factor = 1000.0,
                             float depth_max = 3.0);

/// Create a normal map (image) from a vertex map (image). Useful for
/// point-to-plane odometry.
core::Tensor CreateNormalMap(const core::Tensor& vertex_map);

/// Perform single scale odometry using loss function
/// [(V_p - V_q)^T N_p]^2,
/// requiring normal map generation.
/// KinectFusion, ISMAR 2011
core::Tensor ComputePosePointToPlane(const core::Tensor& source_vtx_map,
                                     const core::Tensor& target_vtx_map,
                                     const core::Tensor& source_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff);

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2 + lambda(D_p - (D_q)')^2,
/// requiring the gradient images of target color and depth.
/// Colored ICP Revisited, ICCV 2017
core::Tensor ComputePoseDirectHybrid(const core::Tensor& source_vtx_map,
                                     const core::Tensor& target_vtx_map,
                                     const core::Tensor& source_color,
                                     const core::Tensor& target_color,
                                     const core::Tensor& source_color_dx,
                                     const core::Tensor& target_color_dy,
                                     const core::Tensor& source_depth_dx,
                                     const core::Tensor& target_depth_dy,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff);

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2,
/// requiring the gradient image of target color.
/// Real-time visual odometry from dense RGB-D images, ICCV Workshops, 2011
core::Tensor ComputePoseDirectIntensity(
        const core::Tensor& source_vtx_map,
        const core::Tensor& target_vtx_map,
        const core::Tensor& source_color,
        const core::Tensor& target_color,
        const core::Tensor& source_color_dx,
        const core::Tensor& source_color_dy,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target,
        float depth_diff);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
