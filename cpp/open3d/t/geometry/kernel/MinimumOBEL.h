// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace minimum_obel {

/// Creates the oriented bounding ellipsoid with the smallest volume.
/// Initially, the algorithm treats every point of the point cloud as equally 
/// important by assigning them the same weight. It creates an initial 
/// ellipsoid guess based on these weights. Then it checks which point is the 
/// worst fit, the one that sticks out the most from the ellipsoid. The algorithm 
/// slightly increases the weight of that problematic point and slightly decreases
/// the weights of the others. This update makes the ellipsoid expand a bit around 
/// the troublesome dot, improving the overall fit. It keeps doing this, 
/// recalculating the ellipsoid and adjusting weights, until the changes become 
/// very small, meaning the ellipsoid is "almost" as tight as possible around all 
/// the dots.
///
/// \param points A list of points with data type of float32 or float64 (N x
/// 3 tensor, where N must be larger than 3).
/// \param robust If set to true uses a more robust method which works
///               in degenerate cases but introduces noise to the points
///               coordinates.
OrientedBoundingEllipsoid ComputeMinimumOBELApprox(const core::Tensor &points,
                                             bool robust);

}  // namespace minimum_obel
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d