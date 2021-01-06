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

/// \brief Convert rotation and translation to the transformation matrix.
///
/// \param R Rotation, a tensor of shape {3, 3} and dtype Float32.
/// \param t Translation, a tensor of shape {3, 3} and dtype Float32.
/// \return Transformation, a tensor of shape {4, 4}, dtype Float32.
core::Tensor RtToTransformation(const core::Tensor &R, const core::Tensor &t);

/// \brief Convert pose to the transformation matrix.
///
/// \param pose Pose [alpha, beta, gamma, tx, ty, tz], a tensor of
/// shape {6} and dtype Float32.
/// \return Transformation, a tensor of shape {4, 4}, dtype Float32.
core::Tensor PoseToTransformation(const core::Tensor &pose);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
