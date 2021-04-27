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

/// \file RGBDOdometry.h
/// All the 4x4 transformation in this file, from params to returns, are
/// Float64. Only convert to Float32 in kernel calls.

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

enum class Method {
    PointToPlane,  // Implemented and commented in ComputePosePointToPlane
    Intensity,     // Implemented and commented in ComputePoseIntensity
    Hybrid,        // Implemented and commented in ComputePoseHybrid
};

// TODO (Wei): Encapsule shared params (depth_max, depth_diff, intrinsic, etc)
// in an option, similar to Registration.

/// \brief Create an RGBD image pyramid given the original source and target
/// RGBD images, and perform hierarchical odometry using specified \p
/// method.
/// Can be used for offline odometry where we do not expect to push performance
/// to the extreme and not reuse vertex/normal map computed before.
/// Input RGBD images hold a depth image (UInt16 or Float32) with a scale
/// factor and a color image (UInt8 x 3).
/// \param source Source RGBD image.
/// \param target Target RGBD image.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_scale Converts depth pixel values to meters by dividing the
/// scale factor.
/// \param depth_diff Depth difference threshold used to filter projective
/// associations.
/// \param iterations Iterations in multiscale odometry, from coarse to fine.
/// \param method Method used to apply RGBD odometry.
core::Tensor RGBDOdometryMultiScale(
        const t::geometry::RGBDImage& source,
        const t::geometry::RGBDImage& target,
        const core::Tensor& intrinsics,
        const core::Tensor& init_source_to_target = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0")),
        float depth_scale = 1000.0f,
        float depth_max = 3.0f,
        float depth_diff = 0.07f,
        const std::vector<int>& iterations = {10, 5, 3},
        const Method method = Method::Hybrid);

/// \brief Estimates the 4x4 rigid transformation T from source to target.
/// Performs one iteration of RGBD odometry using loss function
/// \f$[(V_p - V_q)^T N_p]^2\f$, where
/// \f$ V_p \f$ denotes the vertex at pixel p in the source,
/// \f$ V_q \f$ denotes the vertex at pixel q in the target,
/// \f$ N_p \f$ denotes the normal at pixel p in the source.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// KinectFusion, ISMAR 2011
///
/// \param source_vertex_map (H, W, 3) Float32 source vertex image obtained by
/// CreateVertexMap before calling this function.
/// \param target_vertex_map (H, W, 3) Float32 target vertex image obtained by
/// CreateVertexMap before calling this function.
/// \param target_normal_map (H, W, 3) Float32 target normal image obtained by
/// CreateNormalMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_diff Depth difference threshold used to filter projective
/// associations.
/// \return (4, 4) optimized transformation matrix from source to target.
core::Tensor ComputePosePointToPlane(const core::Tensor& source_vertex_map,
                                     const core::Tensor& target_vertex_map,
                                     const core::Tensor& target_normal_map,
                                     const core::Tensor& intrinsics,
                                     const core::Tensor& init_source_to_target,
                                     float depth_diff);

/// \brief Estimates the 4x4 rigid transformation T from source to target.
/// Performs one iteration of RGBD odometry using loss function
/// \f$(I_p - I_q)^2\f$, where
/// \f$ I_p \f$ denotes the intensity at pixel p in the source,
/// \f$ I_q \f$ denotes the intensity at pixel q in the target.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// Real-time visual odometry from dense RGB-D images, ICCV Workshops, 2011
///
/// \param source_depth_map (H, W, 1) Float32 source depth image obtained by
/// PreprocessDepth before calling this function.
/// \param target_depth_map (H, W, 1) Float32 target depth image obtained by
/// PreprocessDepth before calling this function.
/// \param source_intensity (H, W, 1) Float32 source intensity image obtained by
/// RGBToGray before calling this function.
/// \param target_intensity (H, W, 1) Float32 target intensity image obtained by
/// RGBToGray before calling this function.
/// \param target_intensity_dx (H, W, 1) Float32 target intensity gradient image
/// at x-axis obtained by FilterSobel before calling this function.
/// \param target_intensity_dy (H, W, 1) Float32 target intensity gradient image
/// at y-axis obtained by FilterSobel before calling this function.
/// \param source_vertex_map (H, W, 3) Float32 source vertex image obtained by
/// CreateVertexMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_diff Depth difference threshold used to filter projective
/// associations.
/// \return (4, 4) optimized transformation matrix from source to target.
core::Tensor ComputePoseIntensity(const core::Tensor& source_depth_map,
                                  const core::Tensor& target_depth_map,
                                  const core::Tensor& source_intensity,
                                  const core::Tensor& target_intensity,
                                  const core::Tensor& target_intensity_dx,
                                  const core::Tensor& target_intensity_dy,
                                  const core::Tensor& source_vertex_map,
                                  const core::Tensor& intrinsics,
                                  const core::Tensor& init_source_to_target,
                                  float depth_diff);

/// \brief Estimates the 4x4 rigid transformation T from source to target.
/// Performs one iteration of RGBD odometry using loss function
/// \f$(I_p - I_q)^2 + \lambda(D_p - (D_q)')^2\f$, where
/// \f$ I_p \f$ denotes the intensity at pixel p in the source,
/// \f$ I_q \f$ denotes the intensity at pixel q in the target.
/// \f$ D_p \f$ denotes the depth pixel p in the source,
/// \f$ D_q \f$ denotes the depth pixel q in the target.
/// q is obtained by transforming p with \p init_source_to_target then
/// projecting with \p intrinsics.
/// Colored ICP Revisited, ICCV 2017
///
/// \param source_depth (H, W, 1) Float32 source depth image obtained by
/// PreprocessDepth before calling this function.
/// \param target_depth (H, W, 1) Float32 target depth image obtained by
/// PreprocessDepth before calling this function.
/// \param source_intensity (H, W, 1) Float32 source intensity image obtained by
/// RGBToGray before calling this function.
/// \param target_intensity (H, W, 1) Float32 target intensity image obtained by
/// RGBToGray before calling this function.
/// \param source_depth_dx (H, W, 1) Float32 source depth gradient image
/// at x-axis obtained by FilterSobel before calling this function.
/// \param source_depth_dy (H, W, 1) Float32 source depth gradient image
/// at y-axis obtained by FilterSobel before calling this function.
/// \param source_intensity_dx (H, W, 1) Float32 source intensity gradient image
/// at x-axis obtained by FilterSobel before calling this function.
/// \param source_intensity_dy (H, W, 1) Float32 source intensity gradient image
/// at y-axis obtained by FilterSobel before calling this function.
/// \param target_vertex_map (H, W, 3) Float32 target vertex image obtained by
/// CreateVertexMap before calling this function.
/// \param intrinsics (3, 3) intrinsic matrix for projection.
/// \param init_source_to_target (4, 4) initial transformation matrix from
/// source to target.
/// \param depth_diff Depth difference threshold used to filter projective
/// associations.
/// \return (4, 4) optimized transformation matrix from source to target.
core::Tensor ComputePoseHybrid(const core::Tensor& source_depth,
                               const core::Tensor& target_depth,
                               const core::Tensor& source_intensity,
                               const core::Tensor& target_intensity,
                               const core::Tensor& source_depth_dx,
                               const core::Tensor& source_depth_dy,
                               const core::Tensor& source_intensity_dx,
                               const core::Tensor& source_intensity_dy,
                               const core::Tensor& target_vertex_map,
                               const core::Tensor& intrinsics,
                               const core::Tensor& init_source_to_target,
                               float depth_diff);

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
