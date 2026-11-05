// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

#include "open3d/Macro.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {
namespace detail {

inline void AssertTriangleIndicesInRange(const Eigen::Vector3i& triangle,
                                         int64_t vertex_count) {
    OPEN3D_ASSERT(
            triangle.minCoeff() >= 0 &&
                    static_cast<int64_t>(triangle.maxCoeff()) < vertex_count,
            "Triangle vertex index is out of range.");
}

inline void AssertTriangleIndicesInRange(
        const std::vector<Eigen::Vector3i>& triangles, int64_t vertex_count) {
    for (const Eigen::Vector3i& triangle : triangles) {
        AssertTriangleIndicesInRange(triangle, vertex_count);
    }
}

}  // namespace detail
}  // namespace geometry
}  // namespace open3d