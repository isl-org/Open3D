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

// Private header. Do not include in Open3d.h.

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

void CreateVertexMapCPU(const core::Tensor& depth_map,
                        const core::Tensor& intrinsics,
                        core::Tensor& vertex_map,
                        float depth_scale,
                        float depth_max);

void CreateNormalMapCPU(const core::Tensor& vertex_map,
                        core::Tensor& normal_map);

void ComputePosePointToPlaneCPU(const core::Tensor& source_vertex_map,
                                const core::Tensor& target_vertex_map,
                                const core::Tensor& source_normal_map,
                                const core::Tensor& intrinsics,
                                const core::Tensor& init_source_to_target,
                                core::Tensor& delta,
                                core::Tensor& residual,
                                float depth_diff);
#ifdef BUILD_CUDA_MODULE
void CreateVertexMapCUDA(const core::Tensor& depth_map,
                         const core::Tensor& intrinsics,
                         core::Tensor& vertex_map,
                         float depth_scale,
                         float depth_max);

void CreateNormalMapCUDA(const core::Tensor& vertex_map,
                         core::Tensor& normal_map);

void ComputePosePointToPlaneCUDA(const core::Tensor& source_vertex_map,
                                 const core::Tensor& target_vertex_map,
                                 const core::Tensor& source_normal_map,
                                 const core::Tensor& intrinsics,
                                 const core::Tensor& init_source_to_target,
                                 core::Tensor& delta,
                                 core::Tensor& residual,
                                 float depth_diff);
#endif

OPEN3D_HOST_DEVICE inline bool GetJacobianLocal(
        int64_t workload_idx,
        int64_t cols,
        float depth_diff,
        const t::geometry::kernel::NDArrayIndexer& source_vertex_indexer,
        const t::geometry::kernel::NDArrayIndexer& target_vertex_indexer,
        const t::geometry::kernel::NDArrayIndexer& source_normal_indexer,
        const t::geometry::kernel::TransformIndexer& ti,
        float* J_ij,
        float& r) {
    int64_t y = workload_idx / cols;
    int64_t x = workload_idx % cols;

    float* dst_v = target_vertex_indexer.GetDataPtrFromCoord<float>(x, y);
    if (dst_v[0] == INFINITY) {
        return false;
    }

    float T_dst_v[3], u, v;
    ti.RigidTransform(dst_v[0], dst_v[1], dst_v[2], &T_dst_v[0], &T_dst_v[1],
                      &T_dst_v[2]);
    ti.Project(T_dst_v[0], T_dst_v[1], T_dst_v[2], &u, &v);
    u = round(u);
    v = round(v);

    if (T_dst_v[2] < 0 || !source_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int64_t ui = static_cast<int64_t>(u);
    int64_t vi = static_cast<int64_t>(v);
    float* src_v = source_vertex_indexer.GetDataPtrFromCoord<float>(ui, vi);
    float* src_n = source_normal_indexer.GetDataPtrFromCoord<float>(ui, vi);
    if (src_v[0] == INFINITY || src_n[0] == INFINITY) {
        return false;
    }

    r = (T_dst_v[0] - src_v[0]) * src_n[0] +
        (T_dst_v[1] - src_v[1]) * src_n[1] + (T_dst_v[2] - src_v[2]) * src_n[2];
    if (abs(r) > depth_diff) {
        return false;
    }

    J_ij[0] = -T_dst_v[2] * src_n[1] + T_dst_v[1] * src_n[2];
    J_ij[1] = T_dst_v[2] * src_n[0] - T_dst_v[0] * src_n[2];
    J_ij[2] = -T_dst_v[1] * src_n[0] + T_dst_v[0] * src_n[1];
    J_ij[3] = src_n[0];
    J_ij[4] = src_n[1];
    J_ij[5] = src_n[2];

    return true;
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
