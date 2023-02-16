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

#include "open3d/t/pipelines/slam/Model.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/VoxelBlockGrid.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "open3d/t/pipelines/slam/Frame.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slam {

Model::Model(float voxel_size,
             int block_resolution,
             int est_block_count,
             const core::Tensor& T_init,
             const core::Device& device)
    : voxel_grid_(std::vector<std::string>({"tsdf", "weight", "color"}),
                  std::vector<core::Dtype>(
                          {core::Float32, core::UInt16, core::UInt16}),
                  std::vector<core::SizeVector>({{1}, {1}, {3}}),
                  voxel_size,
                  block_resolution,
                  est_block_count,
                  device),
      T_frame_to_world_(T_init.To(core::Device("CPU:0"))) {}

void Model::SynthesizeModelFrame(Frame& raycast_frame,
                                 float depth_scale,
                                 float depth_min,
                                 float depth_max,
                                 float trunc_voxel_multiplier,
                                 bool enable_color,
                                 float weight_threshold) {
    if (weight_threshold < 0) {
        weight_threshold = std::min(frame_id_ * 1.0f, 3.0f);
    }

    auto result = voxel_grid_.RayCast(
            frustum_block_coords_, raycast_frame.GetIntrinsics(),
            t::geometry::InverseTransformation(GetCurrentFramePose()),
            raycast_frame.GetWidth(), raycast_frame.GetHeight(),
            {"depth", "color"}, depth_scale, depth_min, depth_max,
            weight_threshold, trunc_voxel_multiplier);
    raycast_frame.SetData("depth", result["depth"]);

    if (enable_color) {
        raycast_frame.SetData("color", result["color"]);
    } else if (raycast_frame.GetData("color").NumElements() == 0) {
        // Put a dummy RGB frame to enable RGBD odometry in TrackFrameToModel
        raycast_frame.SetData("color",
                              core::Tensor({raycast_frame.GetHeight(),
                                            raycast_frame.GetWidth(), 3},
                                           core::Float32, core::Device()));
    }
}

odometry::OdometryResult Model::TrackFrameToModel(
        const Frame& input_frame,
        const Frame& raycast_frame,
        float depth_scale,
        float depth_max,
        float depth_diff,
        const odometry::Method method,
        const std::vector<odometry::OdometryConvergenceCriteria>& criteria) {
    // TODO: Expose init_source_to_target as param, and make the input sequence
    // consistent with RGBDOdometryMultiScale.
    const static core::Tensor init_source_to_target =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

    return odometry::RGBDOdometryMultiScale(
            t::geometry::RGBDImage(input_frame.GetDataAsImage("color"),
                                   input_frame.GetDataAsImage("depth")),
            t::geometry::RGBDImage(raycast_frame.GetDataAsImage("color"),
                                   raycast_frame.GetDataAsImage("depth")),
            raycast_frame.GetIntrinsics(), init_source_to_target, depth_scale,
            depth_max, criteria, method,
            odometry::OdometryLossParams(depth_diff));
}

void Model::Integrate(const Frame& input_frame,
                      float depth_scale,
                      float depth_max,
                      float trunc_voxel_multiplier) {
    t::geometry::Image depth = input_frame.GetDataAsImage("depth");
    t::geometry::Image color = input_frame.GetDataAsImage("color");
    core::Tensor intrinsic = input_frame.GetIntrinsics();
    core::Tensor extrinsic =
            t::geometry::InverseTransformation(GetCurrentFramePose());
    frustum_block_coords_ = voxel_grid_.GetUniqueBlockCoordinates(
            depth, intrinsic, extrinsic, depth_scale, depth_max,
            trunc_voxel_multiplier);
    voxel_grid_.Integrate(frustum_block_coords_, depth, color, intrinsic,
                          extrinsic, depth_scale, depth_max,
                          trunc_voxel_multiplier);
}

t::geometry::PointCloud Model::ExtractPointCloud(float weight_threshold,
                                                 int estimated_number) {
    return voxel_grid_.ExtractPointCloud(weight_threshold, estimated_number);
}

t::geometry::TriangleMesh Model::ExtractTriangleMesh(float weight_threshold,
                                                     int estimated_number) {
    return voxel_grid_.ExtractTriangleMesh(weight_threshold, estimated_number);
}

core::HashMap Model::GetHashMap() { return voxel_grid_.GetHashMap(); }

}  // namespace slam
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
