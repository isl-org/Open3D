// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace open3d {
namespace geometry {

class TriangleMesh;
class TetraMesh;

class Qhull {
public:
    /// Computes the convex hull
    /// \param points Input points.
    /// \param joggle_inputs If true allows the algorithm to add random noise
    ///        to the points to work around degenerate inputs. This adds the
    ///        'QJ' option to the qhull command.
    /// \returns The triangle mesh of the convex hull and the list of point
    ///          indices that are part of the convex hull.
    static std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
    ComputeConvexHull(const std::vector<Eigen::Vector3d>& points,
                      bool joggle_inputs = false);

    static std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
    ComputeDelaunayTetrahedralization(
            const std::vector<Eigen::Vector3d>& points);
};

}  // namespace geometry
}  // namespace open3d
