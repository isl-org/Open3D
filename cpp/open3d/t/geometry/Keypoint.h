// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Optional.h"

namespace open3d {
namespace t {
namespace geometry {

/// \brief Function that computes the ISS Keypoints from an input point
/// cloud. This implements the keypoint detection module proposed in Yu
/// Zhong ,"Intrinsic Shape Signatures: A Shape Descriptor for 3D Object
/// Recognition", 2009. The implementation is inspired by the PCL one.
///
/// If either salient_radius or non_max_radius is not provided, the default
/// values of them 6 * pointcloud resolution and 4* pointcloud resolution will
/// be used respectively.
///
/// \param input The input PointCloud where to compute the ISS Keypoints.
/// \param salient_radius [optional] The radius of the spherical neighborhood
/// used to detect the keypoints.
/// \param non_max_radius [optional] The non maxima suppression radius.
/// \param gamma_21 The upper bound on the ratio between the second and the
/// first eigenvalue
/// \param gamma_32 The upper bound on the ratio between the third and the
/// second eigenvalue
/// \param min_neighbors Minimum number of neighbors that has to be found to
/// consider a keypoint.
/// \return The computed ISS Keypoints and its boolean mask tensor of the input
/// PointCloud.
std::tuple<PointCloud, core::Tensor> ComputeISSKeypoints(
        const PointCloud &input,
        const utility::optional<double> salient_radius = utility::nullopt,
        const utility::optional<double> non_max_radius = utility::nullopt,
        double gamma_21 = 0.975,
        double gamma_32 = 0.975,
        int min_neighbors = 5);

}  // namespace geometry
}  // namespace t
}  // namespace open3d