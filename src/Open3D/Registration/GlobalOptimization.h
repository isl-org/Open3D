// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "Open3D/Registration/GlobalOptimizationConvergenceCriteria.h"
#include "Open3D/Registration/GlobalOptimizationMethod.h"

namespace open3d {
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
}  // namespace open3d
