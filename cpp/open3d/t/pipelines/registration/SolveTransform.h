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
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/registration/CorrespondenceChecker.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace registration {

typedef std::pair<core::Tensor, core::Tensor> CorrespondenceSet;

// NOT A CLASS, only some helper functions for Solving Transformations

core::Tensor ComputeTransformationFromRt(const core::Tensor &R,
                                         const core::Tensor &t,
                                         const core::Dtype &dtype,
                                         const core::Device &device);
double det_(const core::Tensor D);

core::Tensor ComputeTransformationFromPose(const core::Tensor &X,
                                           const core::Dtype &dtype,
                                           const core::Device &device);

core::Tensor Compute_A(const core::Tensor &source_select,
                       const core::Tensor &target_n_select,
                       const core::Dtype &dtype,
                       const core::Device &device);

core::Tensor SolvePointToPlaneTransformation(const geometry::PointCloud &source,
                                             const geometry::PointCloud &target,
                                             CorrespondenceSet &corres,
                                             const core::Dtype dtype);

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
