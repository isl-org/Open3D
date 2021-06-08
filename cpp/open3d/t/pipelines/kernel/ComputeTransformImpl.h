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

void ComputePosePointToPlaneCPU(const core::Tensor &source_points,
                                const core::Tensor &target_points,
                                const core::Tensor &target_normals,
                                const core::Tensor &correspondence_indices,
                                core::Tensor &pose,
                                float &residual,
                                int &inlier_count,
                                const core::Dtype &dtype,
                                const core::Device &device);

#ifdef BUILD_CUDA_MODULE
void ComputePosePointToPlaneCUDA(const core::Tensor &source_points,
                                 const core::Tensor &target_points,
                                 const core::Tensor &target_normals,
                                 const core::Tensor &correspondence_indices,
                                 core::Tensor &pose,
                                 float &residual,
                                 int &inlier_count,
                                 const core::Dtype &dtype,
                                 const core::Device &device);
#endif

void ComputeRtPointToPointCPU(const core::Tensor &source_points,
                              const core::Tensor &target_points,
                              const core::Tensor &correspondence_indices,
                              core::Tensor &R,
                              core::Tensor &t,
                              int &inlier_count,
                              const core::Dtype &dtype,
                              const core::Device &device);

template <typename scalar_t>
OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int64_t workload_idx,
        const scalar_t *source_points_ptr,
        const scalar_t *target_points_ptr,
        const scalar_t *target_normals_ptr,
        const int64_t *correspondence_indices,
        scalar_t *J_ij,
        scalar_t &r) {
    utility::LogError(" GetJacobianPointToPlane: Dtype not supported.");
}

template <>
OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane<float>(
        int64_t workload_idx,
        const float *source_points_ptr,
        const float *target_points_ptr,
        const float *target_normals_ptr,
        const int64_t *correspondence_indices,
        float *J_ij,
        float &r) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];
    const int64_t source_idx = 3 * workload_idx;

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

template <>
OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane<double>(
        int64_t workload_idx,
        const double *source_points_ptr,
        const double *target_points_ptr,
        const double *target_normals_ptr,
        const int64_t *correspondence_indices,
        double *J_ij,
        double &r) {
    if (correspondence_indices[workload_idx] == -1) {
        return false;
    }

    const int64_t target_idx = 3 * correspondence_indices[workload_idx];
    const int64_t source_idx = 3 * workload_idx;

    const double &sx = source_points_ptr[source_idx + 0];
    const double &sy = source_points_ptr[source_idx + 1];
    const double &sz = source_points_ptr[source_idx + 2];
    const double &tx = target_points_ptr[target_idx + 0];
    const double &ty = target_points_ptr[target_idx + 1];
    const double &tz = target_points_ptr[target_idx + 2];
    const double &nx = target_normals_ptr[target_idx + 0];
    const double &ny = target_normals_ptr[target_idx + 1];
    const double &nz = target_normals_ptr[target_idx + 2];

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
