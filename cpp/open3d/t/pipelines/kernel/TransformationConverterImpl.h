// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.

#pragma once

#include <cmath>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

/// Shared implementation for PoseToTransformation function.
template <typename scalar_t>
OPEN3D_HOST_DEVICE inline void PoseToTransformationImpl(
        scalar_t *transformation_ptr, const scalar_t *pose_ptr) {
    transformation_ptr[0] = cos(pose_ptr[2]) * cos(pose_ptr[1]);
    transformation_ptr[1] =
            -1 * sin(pose_ptr[2]) * cos(pose_ptr[0]) +
            cos(pose_ptr[2]) * sin(pose_ptr[1]) * sin(pose_ptr[0]);
    transformation_ptr[2] =
            sin(pose_ptr[2]) * sin(pose_ptr[0]) +
            cos(pose_ptr[2]) * sin(pose_ptr[1]) * cos(pose_ptr[0]);
    transformation_ptr[4] = sin(pose_ptr[2]) * cos(pose_ptr[1]);
    transformation_ptr[5] =
            cos(pose_ptr[2]) * cos(pose_ptr[0]) +
            sin(pose_ptr[2]) * sin(pose_ptr[1]) * sin(pose_ptr[0]);
    transformation_ptr[6] =
            -1 * cos(pose_ptr[2]) * sin(pose_ptr[0]) +
            sin(pose_ptr[2]) * sin(pose_ptr[1]) * cos(pose_ptr[0]);
    transformation_ptr[8] = -1 * sin(pose_ptr[1]);
    transformation_ptr[9] = cos(pose_ptr[1]) * sin(pose_ptr[0]);
    transformation_ptr[10] = cos(pose_ptr[1]) * cos(pose_ptr[0]);
}

#ifdef BUILD_CUDA_MODULE
/// \brief Helper function for PoseToTransformationCUDA.
/// Do not call this independently, as it only sets the transformation part
/// in transformation matrix, using the Pose, the rest is set in
/// the parent function PoseToTransformation.
template <typename scalar_t>
void PoseToTransformationCUDA(scalar_t *transformation_ptr,
                              const scalar_t *pose_ptr);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
