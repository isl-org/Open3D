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

// void Model::SynthesizeModelFrame(const Frame& input_frame, int down_factor) {
//     // using MaskCode = t::geometry::TSDFVoxelGrid::RayCastMaskCode;
//     core::Tensor intrinsics = input_frame.GetIntrinsics().Clone();
//     intrinsics /= down_factor;
//     intrinsics[2][2] = 1.0;

//     // clang-format off
//     // auto result = voxel_grid_.RayCast(
//     //         intrinsics,
//     //         T_frame_to_world_.Inverse(),
//     //         input_frame.GetCols() / down_factor,
//     //         input_frame.GetRows() / down_factor,
//     //         100, // steps
//     //         0.1 // start,
//     //         3.0, // stop
//     //         std::min(i * 1.0f, 3.0f), // weight_thr
//     //         MaskCode::DepthMap);
//     // clang-format on

//     // return Frame(result);
// }

// // Track using RGBD odometry - default color + depth
// core::Tensor TrackFrameToModel(const Frame& model_frame,
//                                const Frame& input_frame,
//                                int down_factor) {
//     ComputePosePointToPlane(source_vertex_map, target_vertex_map,
//                             target_normal_map, intrinsics,
//                             init_source_to_target, depth_diff);
// }

// core::Tensor Integrate(const Frame& input_frame);

}  // namespace voxelhashing
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
