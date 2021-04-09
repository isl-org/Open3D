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
#include "open3d/t/geometry/TSDFVoxelGrid.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "open3d/t/pipelines/voxelhashing/Frame.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace voxelhashing {
class Model {
public:
    Model() {}
    Model(float voxel_size,
          float sdf_trunc,
          int est_block_count,
          const core::Tensor& T_init = core::Tensor::Eye(4,
                                                         core::Dtype::Float64,
                                                         core::Device("CPU:0")),
          const core::Device& device = core::Device("CUDA:0"));

    // Apply ray casting to obtain a new frame - default color + depth
    void SynthesizeModelFrame(const Frame& input_frame, int down_factor);

    // Track using RGBD odometry - default color + depth
    core::Tensor TrackFrameToModel(const Frame& model_frame,
                                   const Frame& input_frame);

    // Integrate RGBD
    core::Tensor Integrate(const Frame& input_frame);

public:
    // Maintained volumetric map
    t::geometry::TSDFVoxelGrid voxel_grid_;

    // T_frame_to_model, maintained tracking state
    core::Tensor T_frame_to_world_;
};
}  // namespace voxelhashing
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
