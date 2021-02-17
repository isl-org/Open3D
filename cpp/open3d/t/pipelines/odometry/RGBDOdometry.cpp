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

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"

#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/kernel/RGBDOdometry.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

core::Tensor CreateVertexMap(const t::geometry::Image& depth,
                             const core::Tensor& intrinsics,
                             float depth_factor,
                             float depth_max) {
    core::Tensor vertex_map;
    kernel::odometry::CreateVertexMap(depth.AsTensor(), intrinsics, vertex_map,
                                      depth_factor, depth_max);
    return vertex_map;
}

core::Tensor CreateNormalMap(const core::Tensor& vertex_map,
                             float depth_scale,
                             float depth_max,
                             float depth_diff) {
    core::Tensor normal_map;
    kernel::odometry::CreateNormalMap(vertex_map, normal_map, depth_scale,
                                      depth_max, depth_diff);
    return normal_map;
}

/// Perform single scale odometry using loss function
/// [(V_p - V_q)^T N_p]^2,
/// requiring normal map generation.
/// KinectFusion, ISMAR 2011
core::Tensor ComputePosePointToPlane(const core::Tensor& source_vtx_map,
                                     const core::Tensor& target_vtx_map,
                                     const core::Tensor& source_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff) {
    // Delta target_to_source
    core::Tensor se3_delta;
    core::Tensor residual;
    kernel::odometry::ComputePosePointToPlane(
            source_vtx_map, target_vtx_map, source_normal_map, intrinsics,
            init_source_to_target, se3_delta, residual, depth_diff);
    return pipelines::kernel::PoseToTransformation(se3_delta).Inverse();
}

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2 + lambda(D_p - (D_q)')^2,
/// requiring the gradient images of target color and depth.
/// Colored ICP Revisited, ICCV 2017
// core::Tensor RGBDOdometryJoint(const t::geometry::RGBDImage& source,
//                                const t::geometry::RGBDImage& target,
//                                const t::geometry::Image& source_color_dx,
//                                const t::geometry::Image& source_color_dy,
//                                const t::geometry::Image& source_depth_dx,
//                                const t::geometry::Image& source_depth_dy,
//                                const core::Tensor& intrinsics,
//                                const core::Tensor& init_source_to_target);

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2,
/// requiring the gradient image of target color.
/// Real-time visual odometry from dense RGB-D images, ICCV Workshops, 2011
// core::Tensor RGBDOdometryColor(const t::geometry::RGBDImage& source,
//                                const t::geometry::RGBDImage& target,
//                                const t::geometry::Image& source_color_dx,
//                                const t::geometry::Image& source_color_dy,
//                                const core::Tensor& intrinsics,
//                                const core::Tensor& init_source_to_target);
}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
