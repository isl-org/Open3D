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

#include <string>
#include <vector>

#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"
#include "open3d/t/pipelines/slac/Visualization.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

using PoseGraph = open3d::pipelines::registration::PoseGraph;

struct SLACOptimizerParams {
    /// Number of iterations.
    int max_iterations_ = 5;

    /// Voxel size to downsample input point cloud.
    /// Note: it is not control grid resolution, which is fixed to be 0.375.
    float voxel_size_ = 0.05;

    /// Distance threshold to filter inconsistent correspondences.
    float distance_threshold_ = 0.07;

    /// Fitness threshold to filter inconsistent pairs.
    float fitness_threshold_ = 0.3;

    /// Weight of the regularizor.
    float regularizor_weight_ = 1;

    /// Device to use.
    std::string device_ = "CPU:0";

    /// Relative directory to store SLAC results in the dataset folder.
    std::string slac_folder_ = "";
    std::string GetSubfolderName() const {
        if (voxel_size_ < 0) {
            return fmt::format("{}/original", slac_folder_);
        }
        return fmt::format("{}/{:.3f}", slac_folder_, voxel_size_);
    }
};

struct SLACDebugOption {
    /// Enable debug
    bool debug_ = false;

    /// The node id to start debugging with. Smaller nodes will be skipped for
    /// visualization.
    int debug_start_node_idx_ = 0;
};

/// \brief Read pose graph containing loop closures and odometry to compute
/// correspondences. Uses aggressive pruning -- reject any suspicious pair.
///
/// \param fnames_processed Vector of filenames for processed pointcloud
/// fragments.
///  \param fragment_pose_graph Legacy PoseGraph for pointcloud
/// fragments.
/// \param option SLACOptimizerParams containing the configurations.
/// \param option SLACDebugOption containing the debug options.
void SaveCorrespondencesForPointClouds(
        const std::vector<std::string>& fnames_processed,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params);

/// \brief Simultaneous Localization and Calibration: Self-Calibration of
/// Consumer Depth Cameras, CVPR 2014 Qian-Yi Zhou and Vladlen Koltun Estimate a
/// shared control grid for all fragments for scene reconstruction, implemented
/// in https://github.com/qianyizh/ElasticReconstruction.
///
/// \param fragment_fnames Vector of filenames for pointcloud fragments.
/// \param fragment_pose_graph Legacy PoseGraph for pointcloud fragments.
/// \param option SLACOptimizerOption containing the configurations.
/// \return pair of registraion::PoseGraph and slac::ControlGrid.
std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fragment_fnames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params,
        const SLACDebugOption& debug_option);

PoseGraph RunRigidOptimizerForFragments(
        const std::vector<std::string>& fragment_fnames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params,
        const SLACDebugOption& debug_option);

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
