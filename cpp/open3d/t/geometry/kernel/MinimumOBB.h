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
namespace minimum_obb {

/// Creates the oriented bounding box with the smallest volume.
/// This algorithm is inspired by the article "An Exact Algorithm for
/// Finding Minimum Oriented Bounding Boxes" written by Jukka Jylänki.
/// The original implementation can be found at the following address:
/// https://github.com/juj/MathGeoLib/blob/55053da5e3e55a83043af7324944407b174c3724/src/Geometry/OBB.cpp#L987
///
/// \param points A list of points with data type of float32 or float64 (N x
/// 3 tensor, where N must be larger than 3).
/// \param robust If set to true uses a more robust method which works
///               in degenerate cases but introduces noise to the points
///               coordinates.
OrientedBoundingBox ComputeMinimumOBBJylanki(const core::Tensor &points,
                                             bool robust);

/// Fast approximation of the oriented bounding box with the smallest volume.
/// The algorithm makes use of the fact that at least one edge of
/// the convex hull must be collinear with an edge of the minimum
/// bounding box: for each triangle in the convex hull, calculate
/// the minimal axis aligned box in the frame of that triangle.
/// at the end, return the box with the smallest volume found.
/// \param points A list of points with data type of float32 or float64 (N x
/// 3 tensor, where N must be larger than 3).
/// \param robust If set to true uses a more robust method which works
///               in degenerate cases but introduces noise to the points
///               coordinates.
OrientedBoundingBox ComputeMinimumOBBApprox(const core::Tensor &points,
                                            bool robust);

}  // namespace minimum_obb
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d