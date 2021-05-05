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
#include "open3d/t/pipelines/registration/Registration.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

/// \brief Computes pose for point to plane registration method.
/// \param source_points source points indexed according to correspondences.
/// \param target_points target points indexed according to correspondences.
/// \param target_normals target normals indexed according to correspondences.
/// \param correspondences CorrespondenceSet. [refer to definition in
/// `/cpp/open3d/t/pipelines/registration/TransformationEstimation.h`].
/// \return Pose [alpha beta gamma, tx, ty, tz], a shape {6} tensor of dtype
/// Float32, where alpha, beta, gamma are the Euler angles in the ZYX order.
core::Tensor ComputePosePointToPlane(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const pipelines::registration::CorrespondenceSet &correspondences);

/// \brief Computes (R) Rotation {3,3} and (t) translation {3,}
/// for point to point registration method.
/// \param source_points source points indexed according to correspondences.
/// \param target_points target points indexed according to correspondences.
/// \param correspondences CorrespondenceSet. [refer to definition in
/// `/cpp/open3d/t/pipelines/registration/TransformationEstimation.h`].
/// \return tuple of (R, t). [Dtype: Float32].
std::tuple<core::Tensor, core::Tensor> ComputeRtPointToPoint(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const pipelines::registration::CorrespondenceSet &correspondences);

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
