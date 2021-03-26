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
namespace pipelines {
namespace kernel {
namespace odometry {

void CreateVertexMap(const core::Tensor &depth_map,
                     const core::Tensor &intrinsics,
                     core::Tensor &vertex_map,
                     float depth_scale,
                     float depth_max);

void CreateNormalMap(const core::Tensor &vertex_map, core::Tensor &normal_map);

void ComputePosePointToPlane(const core::Tensor &source_vertex_map,
                             const core::Tensor &target_vertex_map,
                             const core::Tensor &source_normal_map,
                             const core::Tensor &intrinsics,
                             const core::Tensor &init_source_to_target,
                             core::Tensor &delta,
                             core::Tensor &residual,
                             float depth_diff);

void CreateVertexMapCPU(const core::Tensor &depth_map,
                        const core::Tensor &intrinsics,
                        core::Tensor &vertex_map,
                        float depth_scale,
                        float depth_max);

void CreateNormalMapCPU(const core::Tensor &vertex_map,
                        core::Tensor &normal_map);

void ComputePosePointToPlaneCPU(const core::Tensor &source_vertex_map,
                                const core::Tensor &source_normal_map,
                                const core::Tensor &target_vertex_map,
                                const core::Tensor &intrinsics,
                                const core::Tensor &init_source_to_target,
                                core::Tensor &delta,
                                core::Tensor &residual,
                                float depth_diff);

#ifdef BUILD_CUDA_MODULE
void CreateVertexMapCUDA(const core::Tensor &depth_map,
                         const core::Tensor &intrinsics,
                         core::Tensor &vertex_map,
                         float depth_scale,
                         float depth_max);

void CreateNormalMapCUDA(const core::Tensor &vertex_map,
                         core::Tensor &normal_map);

void ComputePosePointToPlaneCUDA(const core::Tensor &source_vertex_map,
                                 const core::Tensor &source_normal_map,
                                 const core::Tensor &target_vertex_map,
                                 const core::Tensor &intrinsics,
                                 const core::Tensor &init_source_to_target,
                                 core::Tensor &delta,
                                 core::Tensor &residual,
                                 float depth_diff);
#endif

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
