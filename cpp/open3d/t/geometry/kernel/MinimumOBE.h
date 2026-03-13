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

// Forward declaration
class OrientedBoundingEllipsoid;

namespace kernel {
namespace minimum_obe {

/// Creates the oriented bounding ellipsoid with the smallest volume using
/// Khachiyan's algorithm. This algorithm computes the minimum volume enclosing
/// ellipsoid (MVEE) around a set of points. The MVEE is unique and can be
/// computed to arbitrary precision using an iterative algorithm.
///
/// The algorithm works by:
/// 1. Computing the convex hull of the input points
/// 2. Running Khachiyan's algorithm to find the optimal ellipsoid
/// 3. Extracting the ellipsoid parameters (center, orientation, radii)
///
/// \param points A list of points with data type of float32 or float64 (N x
/// 3 tensor, where N must be larger than 3).
/// \param robust If set to true uses a more robust method which works
///               in degenerate cases but introduces noise to the points
///               coordinates.
OrientedBoundingEllipsoid ComputeMinimumOBEKhachiyan(const core::Tensor &points,
                                                     bool robust);

}  // namespace minimum_obe
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d