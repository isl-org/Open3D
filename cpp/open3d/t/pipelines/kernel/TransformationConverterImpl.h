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

/// Shared implementation for TransformationToPose function.
/// Reference method: utility::TransformMatrix4dToVector6d.
template <typename scalar_t>
OPEN3D_HOST_DEVICE inline void TransformationToPoseImpl(
        scalar_t *pose_ptr, const scalar_t *transformation_ptr) {
    const scalar_t sy = sqrt(transformation_ptr[0] * transformation_ptr[0] +
                             transformation_ptr[4] * transformation_ptr[4]);
    if (!(sy < 1e-6)) {
        pose_ptr[0] = atan2(transformation_ptr[9], transformation_ptr[10]);
        pose_ptr[1] = atan2(-transformation_ptr[8], sy);
        pose_ptr[2] = atan2(transformation_ptr[4], transformation_ptr[0]);
    } else {
        pose_ptr[0] = atan2(-transformation_ptr[6], transformation_ptr[5]);
        pose_ptr[1] = atan2(-transformation_ptr[8], sy);
        pose_ptr[2] = 0;
    }
}

#ifdef BUILD_CUDA_MODULE
/// \brief Helper function for PoseToTransformationCUDA.
/// Do not call this independently, as it only sets the transformation part
/// in transformation matrix, using the Pose, the rest is set in
/// the parent function PoseToTransformation.
template <typename scalar_t>
void PoseToTransformationCUDA(scalar_t *transformation_ptr,
                              const scalar_t *pose_ptr);

/// \brief Helper function for TransformationToPoseCUDA.
/// Do not call this independently, as it only sets the rotation part in the
/// pose, using the Transformation, the rest is set in the parent function
/// TransformationToPose.
template <typename scalar_t>
void TransformationToPoseCUDA(scalar_t *pose_ptr,
                              const scalar_t *transformation_ptr);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
