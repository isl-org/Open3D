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

#include <cmath>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

/// Shared implementation for PoseToTransformation function.
OPEN3D_HOST_DEVICE inline void PoseToTransformationImpl(
        float *transformation_ptr, const float *pose_ptr) {
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
void PoseToTransformationCUDA(float *transformation_ptr, const float *pose_ptr);
#endif

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
