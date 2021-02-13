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

namespace open3d {
namespace t {

namespace geometry {
class RGBDImage;
class Image;
}  // namespace geometry

namespace pipelines {
namespace odometry {

/// Create an RGBD image pyramid given the original source and target and
/// perform hierarchical odometry.
core::Tensor RGBDOdometryMultiScale(const RGBDImage& source,
                                    const RGBDImage& target,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    const std::vector<int>& iterations = {10, 5,
                                                                          3});

/// Perform single scale odometry using loss function
/// [(V_p - V_q)^T N_p]^2,
/// requiring normal map generation.
/// KinectFusion, ISMAR 2011
core::Tensor RGBDOdometryPointToPlane(
        const Image& source_vtx_map,
        const Image& target_vtx_map,
        const Image& source_normal_map,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target);

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2 + lambda(D_p - (D_q)')^2,
/// requiring the gradient images of target color and depth.
/// Colored ICP Revisited, ICCV 2017
core::Tensor RGBDOdometryJoint(const RGBDImage& source,
                               const RGBDImage& target,
                               const Image& source_color_dx,
                               const Image& source_color_dy,
                               const Image& source_depth_dx,
                               const Image& source_depth_dy,
                               const core::Tensor& intrinsics,
                               const core::Tensor& init_source_to_target);

/// Perform single scale odometry using loss function
/// (I_p - I_q)^2,
/// requiring the gradient image of target color.
/// Real-time visual odometry from dense RGB-D images, ICCV Workshops, 2011
core::Tensor RGBDOdometryColor(const RGBDImage& source,
                               const RGBDImage& target,
                               const Image& source_color_dx,
                               const Image& source_color_dy,
                               const core::Tensor& intrinsics,
                               const core::Tensor& init_source_to_target);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
