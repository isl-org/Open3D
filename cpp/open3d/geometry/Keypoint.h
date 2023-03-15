// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace open3d {
namespace geometry {

class PointCloud;

namespace keypoint {

/// \brief Function that computes the ISS Keypoints from an input point
/// cloud. This implements the keypoint detection module proposed in Yu
/// Zhong ,"Intrinsic Shape Signatures: A Shape Descriptor for 3D Object
/// Recognition", 2009. The implementation is inspired by the PCL one.
///
/// \param input The input PointCloud where to compute the ISS Keypoints.
/// \param salient_radius The radius of the spherical neighborhood used to
/// detect the keypoints
/// \param non_max_radius The non maxima suppression radius. If non of
/// the input parameters are specified or are 0.0, then they will be computed
/// from the input data, taking into account the Model Resolution.
/// \param gamma_21 The upper bound on the ratio between the second and the
/// first eigenvalue
/// \param gamma_32 The upper bound on the ratio between the third and the
/// second eigenvalue
/// \param min_neighbors Minimum number of neighbors that has to be found to
/// consider a keypoint.
/// \authors Ignacio Vizzo and Cyrill Stachniss, University of Bonn.
std::shared_ptr<PointCloud> ComputeISSKeypoints(const PointCloud &input,
                                                double salient_radius = 0.0,
                                                double non_max_radius = 0.0,
                                                double gamma_21 = 0.975,
                                                double gamma_32 = 0.975,
                                                int min_neighbors = 5);

}  // namespace keypoint
}  // namespace geometry
}  // namespace open3d
