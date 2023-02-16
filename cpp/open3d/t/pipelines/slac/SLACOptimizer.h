// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
    int max_iterations_;

    /// Voxel size to downsample input point cloud.
    /// Note: it is not control grid resolution, which is fixed to be 0.375.
    float voxel_size_;

    /// Distance threshold to filter inconsistent correspondences.
    float distance_threshold_;

    /// Fitness threshold to filter inconsistent pairs.
    float fitness_threshold_;

    /// Weight of the regularizer.
    float regularizer_weight_;

    /// Device to use.
    core::Device device_;

    /// Relative directory to store SLAC results in the dataset folder.
    std::string slac_folder_ = "";
    std::string GetSubfolderName() const {
        if (voxel_size_ < 0) {
            return fmt::format("{}/original", slac_folder_);
        }
        return fmt::format("{}/{:.3f}", slac_folder_, voxel_size_);
    }

    /// Default constructor.
    ///
    /// \param max_iterations Number of iterations. [Default: 5].
    /// \param voxel_size Voxel size to downsample input point cloud.
    /// [Default: 0.05].
    /// \param distance_threshold Distance threshold to filter inconsistent
    /// correspondences. [Default: 0.07].
    /// \param fitness_threshold Fitness threshold to filter inconsistent pairs.
    /// [Default: 0.3].
    /// \param regularizer_weight_ Weight of the regularizer. [Default: 1].
    /// \param device Device to use. [Default: CPU:0].
    /// \param slac_folder Relative directory to store SLAC results in the
    /// dataset folder. [Default: ""].
    SLACOptimizerParams(const int max_iterations = 5,
                        const float voxel_size = 0.05,
                        const float distance_threshold = 0.07,
                        const float fitness_threshold = 0.3,
                        const float regularizer_weight = 1,
                        const core::Device device = core::Device("CPU:0"),
                        const std::string slac_folder = "") {
        if (fitness_threshold < 0) {
            utility::LogError("fitness threshold must be positive.");
        }
        if (distance_threshold < 0) {
            utility::LogError("distance threshold must be positive.");
        }

        max_iterations_ = max_iterations;
        voxel_size_ = voxel_size;
        distance_threshold_ = distance_threshold;
        fitness_threshold_ = fitness_threshold;
        regularizer_weight_ = regularizer_weight;
        device_ = device;
        slac_folder_ = slac_folder;
    }
};

struct SLACDebugOption {
    /// Enable debug
    bool debug_ = false;

    /// The node id to start debugging with. Smaller nodes will be skipped for
    /// visualization.
    int debug_start_node_idx_ = 0;

    /// Default Constructor.
    ///
    /// \param debug Enable debug. [Default: False].
    /// \param debug_start_node_idx The node id to start debugging with. Smaller
    /// nodes will be skipped for visualization. [Default: 0].
    SLACDebugOption(const bool debug = false,
                    const int debug_start_node_idx = 0) {
        if (debug_start_node_idx < 0) {
            utility::LogError("debug_start_node_idx must be positive integer.");
        }

        debug_ = debug;
        debug_start_node_idx_ = debug_start_node_idx;
    }

    /// Parameterized Constructor.
    ///
    /// \param debug_start_node_idx The node id to start debugging with. Smaller
    /// nodes will be skipped for visualization. Debug is enabled by default.
    SLACDebugOption(const int debug_start_node_idx) {
        if (debug_start_node_idx < 0) {
            utility::LogError("debug_start_node_idx must be positive integer.");
        }

        debug_ = true;
        debug_start_node_idx_ = debug_start_node_idx;
    }
};

/// \brief Read pose graph containing loop closures and odometry to compute
/// putative correspondences between pairs of pointclouds.
///
/// \param fnames_processed Vector of filenames for processed pointcloud
/// fragments.
///  \param fragment_pose_graph Legacy PoseGraph for pointcloud
/// fragments.
/// \param params Parameters to tune in finding correspondences.
/// \param debug_option SLACDebugOption containing the debug options.
void SaveCorrespondencesForPointClouds(
        const std::vector<std::string>& fnames_processed,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params = SLACOptimizerParams(),
        const SLACDebugOption& debug_option = SLACDebugOption());

/// \brief Simultaneous Localization and Calibration: Self-Calibration of
/// Consumer Depth Cameras, CVPR 2014 Qian-Yi Zhou and Vladlen Koltun
/// Estimate a shared control grid for all fragments for scene reconstruction,
/// implemented in https://github.com/qianyizh/ElasticReconstruction.
///
/// \param fragment_filenames Vector of filenames for pointcloud fragments.
/// \param fragment_pose_graph Legacy PoseGraph for pointcloud fragments.
/// \param params  Parameters to tune in SLAC.
/// \param debug_option Debug options.
/// \return pair of optimized registration::PoseGraph and slac::ControlGrid.
std::pair<PoseGraph, ControlGrid> RunSLACOptimizerForFragments(
        const std::vector<std::string>& fragment_filenames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params = SLACOptimizerParams(),
        const SLACDebugOption& debug_option = SLACDebugOption());

/// \brief Extended ICP to simultaneously align multiple point clouds with dense
/// pairwise point-to-plane distances.
///
/// \param fragment_fnames Vector of filenames for pointcloud fragments.
/// \param fragment_pose_graph Legacy PoseGraph for pointcloud fragments.
/// \param params Parameters to tune in rigid optimization.
/// \param debug_option Debug options.
/// \return Updated pose graph.
PoseGraph RunRigidOptimizerForFragments(
        const std::vector<std::string>& fragment_filenames,
        const PoseGraph& fragment_pose_graph,
        const SLACOptimizerParams& params = SLACOptimizerParams(),
        const SLACDebugOption& debug_option = SLACDebugOption());

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
