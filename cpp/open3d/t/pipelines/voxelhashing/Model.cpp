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

#include "open3d/t/pipelines/voxelhashing/Model.h"

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/geometry/TSDFVoxelGrid.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "open3d/t/pipelines/voxelhashing/Frame.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace voxelhashing {

Model::Model(float voxel_size,
             float sdf_trunc,
             int block_resolution,
             int est_block_count,
             const core::Tensor& T_init,
             const core::Device& device)
    : voxel_grid_({{"tsdf", core::Dtype::Float32},
                   {"weight", core::Dtype::UInt16},
                   {"color", core::Dtype::UInt16}},
                  voxel_size,
                  sdf_trunc,
                  block_resolution,
                  est_block_count,
                  device),
      T_frame_to_world_(T_init.To(core::Device("CPU:0"))) {}

void Model::SynthesizeModelFrame(Frame& raycast_frame, float depth_scale) {
    using MaskCode = t::geometry::TSDFVoxelGrid::SurfaceMaskCode;
    auto result = voxel_grid_.RayCast(
            raycast_frame.GetIntrinsics(), GetCurrentFramePose().Inverse(),
            raycast_frame.GetWidth(), raycast_frame.GetHeight(), 80,
            depth_scale, 0.1, 4.0, std::min(frame_id_ * 1.0f, 3.0f),
            MaskCode::DepthMap | MaskCode::ColorMap);
    raycast_frame.SetData("depth", result[MaskCode::DepthMap]);
    raycast_frame.SetData("color", result[MaskCode::ColorMap]);
}

core::Tensor Model::TrackFrameToModel(const Frame& input_frame,
                                      const Frame& raycast_frame,
                                      float depth_scale,
                                      float depth_max,
                                      float depth_diff) {
    const static core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float32, core::Device("CPU:0"));

    // TODO: more customized / optimized
    return t::pipelines::odometry::RGBDOdometryMultiScale(
            t::geometry::RGBDImage(input_frame.GetDataAsImage("color"),
                                   input_frame.GetDataAsImage("depth")),
            t::geometry::RGBDImage(raycast_frame.GetDataAsImage("color"),
                                   raycast_frame.GetDataAsImage("depth")),
            raycast_frame.GetIntrinsics(), identity, depth_scale, depth_max,
            depth_diff, {10, 0, 0},
            t::pipelines::odometry::Method::PointToPlane);
}

void Model::Integrate(const Frame& input_frame,
                      float depth_scale,
                      float depth_max) {
    voxel_grid_.Integrate(
            input_frame.GetDataAsImage("depth"),
            input_frame.GetDataAsImage("color"), input_frame.GetIntrinsics(),
            GetCurrentFramePose().Inverse(), depth_scale, depth_max);
}

t::geometry::PointCloud Model::ExtractPointCloud(float weight_threshold) {
    return voxel_grid_.ExtractSurfacePoints(weight_threshold);
}

}  // namespace voxelhashing
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
