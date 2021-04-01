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

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/CoreUtil.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {

namespace pipelines {
namespace kernel {
namespace registration {

void ComputePosePointToPlaneCPU(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const std::pair<core::Tensor, core::Tensor> &corres,
        core::Tensor &pose,
        const core::Dtype &dtype,
        const core::Device &device);

#ifdef BUILD_CUDA_MODULE
void ComputePosePointToPlaneCUDA(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const std::pair<core::Tensor, core::Tensor> &corres,
        core::Tensor &pose,
        const core::Dtype &dtype,
        const core::Device &device);
#endif

void ComputeRtPointToPointCPU(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const std::pair<core::Tensor, core::Tensor> &corres,
        core::Tensor &R,
        core::Tensor &t,
        const core::Dtype &dtype,
        const core::Device &device);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int64_t workload_idx,
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondences_first,
        const int64_t *correspondences_second,
        scalar_t *J_ij,
        scalar_t &r) {
    const int64_t &source_idx = 3 * correspondences_first[workload_idx];
    const int64_t &target_idx = 3 * correspondences_second[workload_idx];

    const scalar_t &sx = source_points_ptr[source_idx + 0];
    const scalar_t &sy = source_points_ptr[source_idx + 1];
    const scalar_t &sz = source_points_ptr[source_idx + 2];
    const scalar_t &tx = target_points_ptr[target_idx + 0];
    const scalar_t &ty = target_points_ptr[target_idx + 1];
    const scalar_t &tz = target_points_ptr[target_idx + 2];
    const scalar_t &nx = target_normals_ptr[target_idx + 0];
    const scalar_t &ny = target_normals_ptr[target_idx + 1];
    const scalar_t &nz = target_normals_ptr[target_idx + 2];

    r = (sx - tx) * nx + (sy - ty) * ny + (sz - tz) * nz;
    J_ij[0] = nz * sy - ny * sz;
    J_ij[1] = nx * sz - nz * sx;
    J_ij[2] = ny * sx - nx * sy;
    J_ij[3] = nx;
    J_ij[4] = ny;
    J_ij[5] = nz;

    return true;
}

}  // namespace registration
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
