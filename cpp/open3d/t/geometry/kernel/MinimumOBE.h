// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <Eigen/Core>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/geometry/BoundingVolume.h"

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
/// All computation is in Eigen (Float64) on the CPU.
///
/// \param points  Convex-hull or raw point set (each element is a 3D point).
///                Must contain at least 4 non-coplanar points.
/// \param robust  If true, joggle the convex-hull computation to handle
///                degenerate / near-planar inputs.
/// \return Legacy OrientedBoundingEllipsoid (Eigen types, Float64).
open3d::geometry::OrientedBoundingEllipsoid ComputeMinimumOBEKhachiyan(
        const std::vector<Eigen::Vector3d> &points, bool robust);

/// Tensor wrapper: accepts an (N, 3) tensor of any supported float dtype on
/// any device.  Converts to Eigen on the CPU, calls the Eigen core above,
/// and converts the result back to a tensor OBE with the *same* dtype and
/// device as the input points.
OrientedBoundingEllipsoid ComputeMinimumOBEKhachiyan(const core::Tensor &points,
                                                     bool robust);

}  // namespace minimum_obe
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d