// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/pipelines/registration/GlobalOptimizationConvergenceCriteria.h"
#include "open3d/pipelines/registration/GlobalOptimizationMethod.h"

namespace open3d {
namespace pipelines {
namespace registration {

class PoseGraph;

/// Function to optimize a PoseGraph
/// Reference:
/// [Kümmerle et al 2011]
///    R Kümmerle, G. Grisetti, H. Strasdat, K. Konolige, W. Burgard
///    g2o: A General Framework for Graph Optimization, ICRA 2011
/// [Choi et al 2015]
///    S. Choi, Q.-Y. Zhou, V. Koltun,
///    Robust Reconstruction of Indoor Scenes, CVPR 2015
/// [M. Lourakis 2009]
///    M. Lourakis,
///    SBA: A Software Package for Generic Sparse Bundle Adjustment,
///    Transactions on Mathematical Software, 2009
void GlobalOptimization(
        PoseGraph &pose_graph,
        const GlobalOptimizationMethod &method =
                GlobalOptimizationLevenbergMarquardt(),
        const GlobalOptimizationConvergenceCriteria &criteria =
                GlobalOptimizationConvergenceCriteria(),
        const GlobalOptimizationOption &option = GlobalOptimizationOption());

/// Function to prune out uncertain edges having
/// confidence_ < .edge_prune_threshold_
std::shared_ptr<PoseGraph> CreatePoseGraphWithoutInvalidEdges(
        const PoseGraph &pose_graph, const GlobalOptimizationOption &option);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
