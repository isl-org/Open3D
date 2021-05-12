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
#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void ComputePosePointToPlaneCPU(const float *source_points_ptr,
                                const float *target_points_ptr,
                                const float *target_normals_ptr,
                                const int64_t *correspondences_first,
                                const int64_t *correspondences_second,
                                const int n,
                                core::Tensor &pose,
                                const core::Dtype &dtype,
                                const core::Device &device);

#ifdef BUILD_CUDA_MODULE
void ComputePosePointToPlaneCUDA(const float *source_points_ptr,
                                 const float *target_points_ptr,
                                 const float *target_normals_ptr,
                                 const int64_t *correspondences_first,
                                 const int64_t *correspondences_second,
                                 const int n,
                                 core::Tensor &pose,
                                 const core::Dtype &dtype,
                                 const core::Device &device);
#endif

void ComputeRtPointToPointCPU(const float *source_points_ptr,
                              const float *target_points_ptr,
                              const int64_t *correspondences_first,
                              const int64_t *correspondences_second,
                              const int n,
                              core::Tensor &R,
                              core::Tensor &t,
                              const core::Dtype dtype,
                              const core::Device device);

OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int64_t workload_idx,
        const float *source_points_ptr,
        const float *target_points_ptr,
        const float *target_normals_ptr,
        const int64_t *correspondence_first,
        const int64_t *correspondence_second,
        float *J_ij,
        float &r) {
    // TODO (@rishabh): Pass correspondence without eliminating -1
    // in registration::GetRegistationResultAndCorrespondences,
    // and directly check if valid (index != -1) here.
    // In that case, only use correspondence_second as index for
    // target, and workload_idx as index for source pointcloud.

    const int64_t source_idx = 3 * correspondence_first[workload_idx];
    const int64_t target_idx = 3 * correspondence_second[workload_idx];

    const float &sx = source_points_ptr[source_idx + 0];
    const float &sy = source_points_ptr[source_idx + 1];
    const float &sz = source_points_ptr[source_idx + 2];
    const float &tx = target_points_ptr[target_idx + 0];
    const float &ty = target_points_ptr[target_idx + 1];
    const float &tz = target_points_ptr[target_idx + 2];
    const float &nx = target_normals_ptr[target_idx + 0];
    const float &ny = target_normals_ptr[target_idx + 1];
    const float &nz = target_normals_ptr[target_idx + 2];

    r = (sx - tx) * nx + (sy - ty) * ny + (sz - tz) * nz;
    J_ij[0] = nz * sy - ny * sz;
    J_ij[1] = nx * sz - nz * sx;
    J_ij[2] = ny * sx - nx * sy;
    J_ij[3] = nx;
    J_ij[4] = ny;
    J_ij[5] = nz;

    return true;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
