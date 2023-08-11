// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/Open3D.h"

namespace open3d {
namespace apps {
namespace offline_reconstruction {

static const Eigen::Matrix4d flip_transformation = Eigen::Matrix4d({
        {1, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1},
});

void DrawRegistrationResult(const geometry::PointCloud& src,
                            const geometry::PointCloud& dst,
                            const Eigen::Matrix4d& transformation,
                            bool keep_color = false) {
    auto transformed_src = std::make_shared<geometry::PointCloud>(src);
    auto transformed_dst = std::make_shared<geometry::PointCloud>(dst);
    if (!keep_color) {
        transformed_src->PaintUniformColor(Eigen::Vector3d(1, 0.706, 0));
        transformed_dst->PaintUniformColor(Eigen::Vector3d(0, 0.651, 0.929));
    }
    transformed_src->Transform(flip_transformation * transformation);
    transformed_dst->Transform(flip_transformation);
    visualization::DrawGeometries({transformed_src, transformed_dst});
}

}  // namespace offline_reconstruction
}  // namespace apps
}  // namespace open3d