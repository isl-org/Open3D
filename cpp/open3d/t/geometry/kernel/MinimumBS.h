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
class BoundingSphere;

namespace kernel {
namespace bounding_sphere {

/// Creates the minimum enclosing sphere of a set of 3D points using Welzl's
/// algorithm. This algorithm computes the smallest sphere that encloses all
/// points in the input set. The algorithm is randomized with expected linear
/// time complexity.
///
/// The algorithm works by:
/// 1. Randomizing the input point order
/// 2. Recursively computing the minimum sphere
/// 3. Checking if each point lies inside the current sphere
/// 4. If not, adding it to the boundary set and recomputing
///
/// All computation is in Eigen (Float64) on the CPU.
///
/// \param points  A set of 3D points. Each element is a 3D point.
///                Must contain at least 1 point.
/// \param robust  Whether to use robust computation for the convex hull.
/// \return Legacy BoundingSphere with center and radius (Eigen types, Float64).
open3d::geometry::BoundingSphere ComputeMinimumBSWelzl(
        const std::vector<Eigen::Vector3d> &points, bool robust);

/// Tensor wrapper: accepts an (N, 3) tensor of any supported float dtype on
/// any device. Converts to Eigen on the CPU, calls the Eigen core above,
/// and converts the result back to a tensor BoundingSphere with the *same*
/// dtype and device as the input points.
BoundingSphere ComputeMinimumBSWelzl(const core::Tensor &points, bool robust);

/// Creates an approximate minimum bounding sphere using Ritter's algorithm.
/// This is a fast, non-iterative algorithm that provides a bounding sphere
/// that encloses all input points.
///
/// \param points  A set of 3D points. Each element is a 3D point.
///                Must contain at least 1 point.
/// \return Legacy BoundingSphere with center and radius (Eigen types, Float64).
open3d::geometry::BoundingSphere ComputeApproximateBSRitter(
        const std::vector<Eigen::Vector3d> &points);

/// Tensor wrapper: accepts an (N, 3) tensor of any supported float dtype on
/// any device. Converts to Eigen on the CPU, calls the Eigen core above,
/// and converts the result back to a tensor BoundingSphere with the *same*
/// dtype and device as the input points.
BoundingSphere ComputeApproximateBSRitter(
        const core::Tensor &points);

}  // namespace bounding_sphere
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
