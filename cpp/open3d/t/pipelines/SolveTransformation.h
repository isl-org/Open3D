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

// THIS FILE IS FOR TEMPORARY PURPOSE, FOR HELPING TRANSFORMATIONESTIMATION
// WITH FUNCTIONS TO SOLVE EQUATIONS, AND WILL BE REMOVED WHEN THE
// PROPER FUNCTIONALITIES ARE ADDED TO THE CORE

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace pipelines {

// Utility functions
// Accessible directly as
// t::pipelines::ComputeTransformationFromRt
// t::pipelines::ComputeTransformationFromPose

/// \brief Functions for Computing Transformation Matrix {4,4}
/// from Rotation {3,3} and Translation {3}
/// \param R Rotation Tensor {3,3} Float32
/// \param t Translation Tensor {3} Float32
core::Tensor ComputeTransformationFromRt(const core::Tensor &R,
                                         const core::Tensor &t);

/// \brief Functions for Computing Transformation Matrix {4,4}
/// from Pose {6} [alpha, beta, gamma, tx, ty, tz]
/// \param X Pose {6} Float32
core::Tensor ComputeTransformationFromPose(const core::Tensor &X);

/// \brief Helper function for ComputeTransformationFromPose CUDA
/// Do not call this independendtly, as it only sets the Rotation part
/// in Transformation matrix, using the Pose, the rest is set in
/// the parent function ComputeTransformationFromPose
void ComputeTransformationFromPoseCUDA(float *transformation_ptr, float *X_ptr);

/// \brief Helper function for ComputeTransformationFromPose CPU
/// Do not call this independendtly, as it only sets the Rotation part
/// in Transformation matrix, using the Pose, the rest is set in
/// the parent function ComputeTransformationFromPose
void ComputeTransformationFromPoseCPU(float *transformation_ptr, float *X_ptr);

}  // namespace pipelines
}  // namespace t
}  // namespace open3d
