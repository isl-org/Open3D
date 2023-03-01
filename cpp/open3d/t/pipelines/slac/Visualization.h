// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/EigenConverter.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

/// \brief Visualize pairs with correspondences.
///
/// \param tpcd_i, source point cloud.
/// \param tpcd_j, target point cloud.
/// \param correspondences Putative correspondence between tcpd_i and tpcd_j.
/// \param T_ij Transformation from tpcd_i to tpcd_j. Use T_j.Inverse() @ T_i
/// (node transformation in a pose graph) to check global correspondences , and
/// T_ij (edge transformation) to check pairwise correspondences.
void VisualizePointCloudCorrespondences(const t::geometry::PointCloud& tpcd_i,
                                        const t::geometry::PointCloud& tpcd_j,
                                        const core::Tensor correspondences,
                                        const core::Tensor& T_ij);

void VisualizePointCloudEmbedding(t::geometry::PointCloud& tpcd_param,
                                  ControlGrid& ctr_grid,
                                  bool show_lines = true);

void VisualizePointCloudDeformation(const geometry::PointCloud& tpcd_param,
                                    ControlGrid& ctr_grid);

void VisualizeGridDeformation(ControlGrid& cgrid);

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
