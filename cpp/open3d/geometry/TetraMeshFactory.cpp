// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TetraMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
TetraMesh::CreateFromPointCloud(const PointCloud& point_cloud) {
    if (point_cloud.points_.size() < 4) {
        utility::LogError("Not enough points to create a tetrahedral mesh.");
    }
    return Qhull::ComputeDelaunayTetrahedralization(point_cloud.points_);
}

}  // namespace geometry
}  // namespace open3d
