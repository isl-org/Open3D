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

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/kernel/RGBDOdometry.h"
#include "open3d/t/pipelines/kernel/TransformationConverter.h"
#include "open3d/visualization/utility/DrawGeometry.h"

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

    core::Tensor T_delta_inv = pipelines::kernel::PoseToTransformation(
            se3_delta.To(core::Dtype::Float32));
    utility::LogInfo("T_delta_inv = {}", T_delta_inv.ToString());

    core::Tensor R_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 0, 3);
    utility::LogInfo("RTR = {}", R_inv.T().Matmul(R_inv).ToString());
    core::Tensor t_inv = T_delta_inv.Slice(0, 0, 3).Slice(1, 3, 4);

    core::Tensor T_delta = core::Tensor::Zeros({4, 4}, core::Dtype::Float32);
    T_delta.Slice(0, 0, 3).Slice(1, 0, 3) = R_inv.T();
    T_delta.Slice(0, 0, 3).Slice(1, 3, 4) = R_inv.T().Matmul(t_inv).Neg();
    T_delta[-1][-1] = 1;

    return T_delta;
}

core::Tensor RGBDOdometryMultiScale(const t::geometry::RGBDImage& source,
                                    const t::geometry::RGBDImage& target,
                                    const core::Tensor& intrinsics,
                                    const core::Tensor& init_source_to_target,
                                    float depth_factor,
                                    float depth_diff,
                                    const std::vector<int>& iterations,
                                    const LossType method) {
    core::Device device = source.depth_.GetDevice();
    if (target.depth_.GetDevice() != device) {
        utility::LogError(
                "Device mismatch, got {} for source and {} for target.",
                device.ToString(), target.depth_.GetDevice().ToString());
    }

    core::Tensor intrinsics_d = intrinsics.To(device, true);

    core::Device host("CPU:0");
    core::Tensor trans_d = init_source_to_target.To(host);
    if (method == LossType::PointToPlane) {
        int64_t n = int64_t(iterations.size());

        std::vector<core::Tensor> src_vertex_maps(iterations.size());
        std::vector<core::Tensor> src_normal_maps(iterations.size());
        std::vector<core::Tensor> dst_vertex_maps(iterations.size());
        std::vector<core::Tensor> intrinsic_matrices(iterations.size());

        t::geometry::Image src_depth = source.depth_;
        t::geometry::Image dst_depth = target.depth_;

        // Create image pyramid
        for (int64_t i = 0; i < n; ++i) {
            core::Tensor src_vertex_map =
                    t::pipelines::odometry::CreateVertexMap(
                            src_depth, intrinsics_d, depth_factor);

            // t::geometry::Image src_depth_filtered =
            //         src_depth.FilterBilateral(5, 50, 50);
            // core::Tensor src_vertex_map_filtered =
            //         t::pipelines::odometry::CreateVertexMap(
            //                 src_depth_filtered, intrinsics_d, depth_factor);
            core::Tensor src_normal_map =
                    t::pipelines::odometry::CreateNormalMap(src_vertex_map);

            core::Tensor dst_vertex_map =
                    t::pipelines::odometry::CreateVertexMap(
                            dst_depth, intrinsics_d, depth_factor);

            src_vertex_maps[n - 1 - i] = src_vertex_map;
            src_normal_maps[n - 1 - i] = src_normal_map;
            dst_vertex_maps[n - 1 - i] = dst_vertex_map;

            intrinsic_matrices[n - 1 - i] = intrinsics_d.Clone();

            if (i != n - 1) {
                src_depth = src_depth.PyrDown();
                dst_depth = dst_depth.PyrDown();

                intrinsics_d /= 2;
                intrinsics_d[-1][-1] = 1;
            }
        }

        // Odometry
        for (int64_t i = 0; i < n; ++i) {
            // utility::LogInfo("level {}, intrinsics {}", i,
            //                  intrinsic_matrices[i].ToString());

            // utility::LogInfo("src_vtx_maps[{}].sum = {}", i,
            //                  src_vertex_maps[i].Sum({0, 1}).ToString());
            // utility::LogInfo("dst_vtx_maps[{}].sum = {}", i,
            //                  dst_vertex_maps[i].Sum({0, 1}).ToString());
            // utility::LogInfo("src_normal_maps[{}].sum = {}", i,
            //                  src_normal_maps[i].Sum({0, 1}).ToString());
            for (int iter = 0; iter < iterations[i]; ++iter) {
                core::Tensor delta_src_to_dst =
                        t::pipelines::odometry::ComputePosePointToPlane(
                                src_vertex_maps[i], dst_vertex_maps[i],
                                src_normal_maps[i], intrinsic_matrices[i],
                                trans_d.To(device), depth_diff);
                trans_d = delta_src_to_dst.Matmul(trans_d);
                utility::LogInfo("T = {}", trans_d.ToString());
                utility::LogInfo("TTT = {}",
                                 trans_d.T().Matmul(trans_d).ToString());
            }
        }
    } else {
        utility::LogError("Odometry method not implemented.");
    }

    return trans_d;
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
