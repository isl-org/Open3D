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

#include <json/json.h>

#include "DebugUtil.h"
#include "FileSystemUtil.h"
#include "open3d/Open3D.h"

namespace open3d {
namespace apps {
namespace offline_reconstruction {

/// \class MatchingResult
/// \brief Result of matching with two fragments.
///
class MatchingResult {
public:
    explicit MatchingResult(int s, int t)
        : s_(s),
          t_(t),
          success_(false),
          transformation_(Eigen::Matrix4d::Identity()),
          information_(Eigen::Matrix6d::Identity()) {}

    virtual ~MatchingResult() {}

public:
    int s_;
    int t_;
    bool success_;
    Eigen::Matrix4d transformation_;
    Eigen::Matrix6d information_;
};

class ReconstructionPipeline {
public:
    /// \brief Construct a new Reconstruction Pipeline object
    ///
    /// \param config Json object that contains the configuration of the
    /// pipeline.
    explicit ReconstructionPipeline(const Json::Value& config)
        : config_(config) {}

    virtual ~ReconstructionPipeline() {}

private:
    Json::Value config_;
    int n_fragments_;

public:
    /// \brief Make fragments from raw RGBD images.
    ///
    void MakeFragments() {
        utility::LogInfo("Making fragments from RGBD sequence.");
        MakeCleanFolder(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_fragment"].asString()));

        std::vector<std::string> color_files, depth_files;
        std::tie(color_files, depth_files) =
                ReadRGBDFiles(config_["path_dataset"].asString());

        n_fragments_ = (int)ceil((float)color_files.size() /
                                 config_["n_frames_per_fragment"].asFloat());

        if (config_["multi_threading"].asBool()) {
            std::vector<std::thread> thread_list;
            for (int i = 0; i < n_fragments_; i++) {
                thread_list.push_back(std::thread(
                        &ReconstructionPipeline::ProcessSingleFragment, this, i,
                        color_files, depth_files));
            }
            for (auto& thread : thread_list) {
                thread.join();
            }
        } else {
            for (int i = 0; i < n_fragments_; i++) {
                ProcessSingleFragment(i, color_files, depth_files);
            }
        }
    }

    /// \brief Register fragments and compute global odometry.
    ///
    void RegisterFragments() {
        utility::LogInfo("Registering fragments.");
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        MakeCleanFolder(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_scene"].asString()));

        pipelines::registration::PoseGraph scene_pose_graph;
        const bool ret = MakePoseGraphForScene(scene_pose_graph);
        if (!ret) {
            return;
        }

        // Optimize pose graph for scene.
        OptimizePoseGraph(
                config_["voxel_size"].asDouble() * 1.4,
                config_["preference_loop_closure_registration"].asDouble(),
                scene_pose_graph);
        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_posegraph_optimized"]
                                .asString()),
                scene_pose_graph);

        // Save global scene camera trajectory.
        SaveSceneTrajectory(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_posegraph_optimized"]
                                .asString()),
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_traj"].asString()));
    }

    /// \brief Refine fragments registration and re-compute global odometry.
    ///
    void RefineRegistration() {
        utility::LogInfo("Refining rough registration of fragments.");
        RefineFragments();
        SaveSceneTrajectory(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_refined_posegraph_optimized"]
                                .asString()),
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_traj"].asString()));
    }

    /// \brief Integrate RGBD images with global odometry.
    ///
    void IntegrateScene() {
        utility::LogInfo(
                "Integrate the whole RGBD sequence using estimated camera "
                "pose.");

        camera::PinholeCameraTrajectory camera_trajectory;
        io::ReadPinholeCameraTrajectory(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_traj"].asString()),
                camera_trajectory);

        const auto rgbd_files =
                ReadRGBDFiles(config_["path_dataset"].asString());
        IntegrateSceneRGBDTSDF(rgbd_files, camera_trajectory);
    }

    /// \brief Run SLAC optimization or fragments.
    ///
    void SLAC() {
        utility::LogInfo("Running SLAC optimization.");
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        const auto ply_files = ReadPlyFiles(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_fragment"].asString()));
        if (ply_files.size() == 0) {
            utility::LogError(
                    "No fragments found in {}, please make sure the "
                    "reconstruction_system has finished running on the "
                    "dataset ",
                    utility::filesystem::JoinPath(
                            config_["path_dataset"].asString(),
                            config_["folder_fragment"].asString()));
        }

        pipelines::registration::PoseGraph scene_pose_graph;
        io::ReadPoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_refined_posegraph_optimized"]
                                .asString()),
                scene_pose_graph);

        // SLAC optimizer parameters.
        t::pipelines::slac::SLACOptimizerParams params;
        params.max_iterations_ = config_["max_iterations"].asInt();
        params.voxel_size_ = config_["voxel_size"].asFloat();
        params.distance_threshold_ = config_["distance_threshold"].asFloat();
        params.fitness_threshold_ = config_["fitness_threshold"].asFloat();
        params.regularizer_weight_ = config_["regularizer_weight"].asFloat();
        params.device_ = core::Device(config_["device"].asString());
        params.slac_folder_ = utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_slac"].asString());

        t::pipelines::slac::SLACDebugOption debug_option(false, 0);
        pipelines::registration::PoseGraph update;
        if (config_["method"].asString() == "rigid") {
            update = t::pipelines::slac::RunRigidOptimizerForFragments(
                    ply_files, scene_pose_graph, params, debug_option);
        } else if (config_["method"].asString() == "slac") {
            t::pipelines::slac::ControlGrid control_grid;
            std::tie(update, control_grid) =
                    t::pipelines::slac::RunSLACOptimizerForFragments(
                            ply_files, scene_pose_graph, params, debug_option);
            const auto hashmap = control_grid.GetHashMap();
            const auto active_buf_indices =
                    hashmap->GetActiveIndices().To(core::Int64);

            const auto key_tensor =
                    hashmap->GetKeyTensor().IndexGet({active_buf_indices});
            key_tensor.Save(utility::filesystem::JoinPath(
                    params.GetSubfolderName(), "ctr_grid_keys.npy"));

            const auto value_tensor =
                    hashmap->GetValueTensor().IndexGet({active_buf_indices});
            value_tensor.Save(utility::filesystem::JoinPath(
                    params.GetSubfolderName(), "ctr_grid_values.npy"));
        } else {
            utility::LogError(
                    "Requested optimization method {}, is not implemented. "
                    "Implemented methods includes slac and rigid.",
                    config_["method"].asString());
        }

        // Write updated pose graph.
        io::WritePoseGraph(utility::filesystem::JoinPath(
                                   params.GetSubfolderName(),
                                   config_["template_optimized_posegraph_slac"]
                                           .asString()),
                           update);

        // Write trajectory for slac-integrate stage.
        SaveSceneTrajectory(utility::filesystem::JoinPath(
                                    params.GetSubfolderName(),
                                    config_["template_optimized_posegraph_slac"]
                                            .asString()),
                            params.GetSubfolderName() +
                                    "/optimized_trajectory_" +
                                    config_["method"].asString() + ".log");
    }

    /// \brief Integrate scene using SLAC results.
    ///
    void IntegrateSceneSLAC() {
        utility::LogInfo("Running SLAC integration.");
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        // Dataset path and slac subfolder path.
        // Slac default subfolder for 0.050 voxel size: `dataset/slac/0.050/`.
        const auto path_dataset = config_["path_dataset"].asString();
        const auto path_slac = utility::filesystem::JoinPath(
                path_dataset, config_["subfolder_slac"].asString());
        const auto path_fragment = utility::filesystem::JoinPath(
                path_dataset, config_["folder_fragment"].asString());

        std::vector<std::string> color_files, depth_files;
        std::tie(color_files, depth_files) = ReadRGBDFiles(path_dataset);
        if (color_files.size() != depth_files.size()) {
            utility::LogError(
                    "Number of color {} and depth {} files do not match.",
                    color_files.size(), depth_files.size());
        }

        pipelines::registration::PoseGraph scene_pose_graph;
        io::ReadPoseGraph(
                utility::filesystem::JoinPath(
                        path_slac, config_["template_optimized_posegraph_slac"]
                                           .asString()),
                scene_pose_graph);

        const core::Device device(config_["device"].asString());
        const Eigen::Matrix3d& intrinsic =
                GetCameraIntrinsic().intrinsic_matrix_;

        const core::Tensor intrinsic_t = core::Tensor::Init<double>(
                {{intrinsic(0, 0), 0, intrinsic(0, 2)},
                 {0, intrinsic(1, 1), intrinsic(1, 2)},
                 {0, 0, 1}},
                device);

        t::geometry::VoxelBlockGrid voxel_grid(
                {"tsdf", "weight", "color"},
                {core::Float32, core::Float32, core::Float32}, {{1}, {1}, {3}},
                config_["tsdf_cubic_size"].asDouble() / 512.0, 16,
                (int64_t)config_["block_count"].asInt(), device);

        // Load control grid.
        const auto ctr_grid_keys = core::Tensor::Load(
                utility::filesystem::JoinPath(path_slac, "ctr_grid_keys.npy"));
        const auto ctr_grid_values =
                core::Tensor::Load(utility::filesystem::JoinPath(
                        path_slac, "ctr_grid_values.npy"));

        t::pipelines::slac::ControlGrid ctr_grid(
                3.0 / 8, ctr_grid_keys.To(device), ctr_grid_values.To(device),
                device);

        int k = 0;
        const float depth_scale = config_["depth_scale"].asFloat();
        const float depth_max = config_["depth_max"].asFloat();

        for (size_t i = 0; i < scene_pose_graph.nodes_.size(); ++i) {
            pipelines::registration::PoseGraph fragment_pose_graph;
            io::ReadPoseGraph(utility::filesystem::JoinPath(
                                      path_fragment,
                                      "fragment_optimized_" +
                                              PadZeroToNumber(i, 3) + ".json"),
                              fragment_pose_graph);
            for (size_t j = 0; j < fragment_pose_graph.nodes_.size(); ++j) {
                const Eigen::Matrix4d& pose_local =
                        fragment_pose_graph.nodes_[j].pose_;

                core::Tensor extrinsic_local_t(pose_local.data(), {4, 4},
                                               core::Float64, device);
                extrinsic_local_t = extrinsic_local_t.T().Inverse();

                const Eigen::Matrix4d pose =
                        scene_pose_graph.nodes_[i].pose_ *
                        fragment_pose_graph.nodes_[j].pose_;
                core::Tensor extrinsic_t(pose.data(), {4, 4}, core::Float64,
                                         device);
                extrinsic_t = extrinsic_t.T().Inverse().Contiguous();

                t::geometry::Image depth, color;
                t::io::ReadImage(depth_files[k], depth);
                t::io::ReadImage(color_files[k], color);
                t::geometry::RGBDImage rgbd(color.To(device), depth.To(device));

                utility::LogInfo("Deforming and integrating Frame {}", k);
                const auto rgbd_projected = ctr_grid.Deform(
                        rgbd, intrinsic_t, extrinsic_t, depth_scale, depth_max);

                const auto frustum_block_coords =
                        voxel_grid.GetUniqueBlockCoordinates(
                                rgbd_projected.depth_, intrinsic_t, extrinsic_t,
                                depth_scale, depth_max);
                voxel_grid.Integrate(frustum_block_coords,
                                     rgbd_projected.depth_,
                                     rgbd_projected.color_, intrinsic_t,
                                     extrinsic_t, depth_scale, depth_max);
                k++;
            }
        }

        if (config_["save_output_as"].asString() == "pointcloud") {
            const auto pcd =
                    voxel_grid.ExtractPointCloud().To(core::Device("CPU:0"));
            t::io::WritePointCloud(
                    utility::filesystem::JoinPath(path_slac,
                                                  "output_slac_pointcloud.ply"),
                    pcd);
        } else {
            const auto mesh =
                    voxel_grid.ExtractTriangleMesh().To(core::Device("CPU:0"));
            const auto mesh_legacy = mesh.ToLegacy();
            io::WriteTriangleMesh(utility::filesystem::JoinPath(
                                          path_slac, "output_slac_mesh.ply"),
                                  mesh_legacy);
        }
    }

private:
    camera::PinholeCameraIntrinsic GetCameraIntrinsic() {
        camera::PinholeCameraIntrinsic intrinsic;
        if (config_.isMember("path_intrinsic") &&
            config_["path_intrinsic"] != "") {
            io::ReadIJsonConvertible(config_["path_intrinsic"].asString(),
                                     intrinsic);
        } else {
            intrinsic = camera::PinholeCameraIntrinsic(
                    camera::PinholeCameraIntrinsicParameters::
                            PrimeSenseDefault);
        }
        return intrinsic;
    }

    geometry::RGBDImage ReadRGBDImage(const std::string& color_file,
                                      const std::string& depth_file,
                                      bool convert_rgb_to_intensity) {
        geometry::Image rgb, depth;
        io::ReadImage(color_file, rgb);
        io::ReadImage(depth_file, depth);
        return *geometry::RGBDImage::CreateFromColorAndDepth(
                rgb, depth, config_["depth_scale"].asDouble(),
                config_["depth_max"].asDouble(), convert_rgb_to_intensity);
    }

    void SaveSceneTrajectory(const std::string& scene_pose_graph_file,
                             const std::string& trajectory_file) {
        const camera::PinholeCameraIntrinsic intrinsic = GetCameraIntrinsic();
        pipelines::registration::PoseGraph scene_pose_graph;
        io::ReadPoseGraph(scene_pose_graph_file, scene_pose_graph);

        camera::PinholeCameraTrajectory camera_trajectory;
        for (size_t i = 0; i < scene_pose_graph.nodes_.size(); i++) {
            pipelines::registration::PoseGraph fragment_pose_graph;
            io::ReadPoseGraph(
                    utility::filesystem::JoinPath(
                            config_["path_dataset"].asString(),
                            config_["template_fragment_posegraph_optimized"]
                                            .asString() +
                                    "fragment_optimized_" +
                                    PadZeroToNumber(i, 3) + ".json"),
                    fragment_pose_graph);
            for (size_t j = 0; j < fragment_pose_graph.nodes_.size(); j++) {
                const Eigen::Matrix4d odom =
                        scene_pose_graph.nodes_[i].pose_ *
                        fragment_pose_graph.nodes_[j].pose_;
                camera::PinholeCameraParameters camera_parameters;
                camera_parameters.intrinsic_ = intrinsic;
                camera_parameters.extrinsic_ = odom;
                camera_trajectory.parameters_.push_back(camera_parameters);
            }
        }

        io::WritePinholeCameraTrajectory(trajectory_file, camera_trajectory);
    }

    void ProcessSingleFragment(int fragment_id,
                               const std::vector<std::string>& color_files,
                               const std::vector<std::string>& depth_files) {
        const camera::PinholeCameraIntrinsic intrinsic = GetCameraIntrinsic();

        const int sid = fragment_id * config_["n_frames_per_fragment"].asInt();
        const int eid = std::min(sid + config_["n_frames_per_fragment"].asInt(),
                                 (int)color_files.size());

        MakePoseGraphForFragment(fragment_id, sid, eid, color_files,
                                 depth_files, intrinsic);
    }

    void MakePoseGraphForFragment(
            int fragment_id,
            int sid,
            int eid,
            const std::vector<std::string>& color_files,
            const std::vector<std::string>& depth_files,
            const camera::PinholeCameraIntrinsic& intrinsic) {
        pipelines::registration::PoseGraph pose_graph;
        Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
        pose_graph.nodes_.push_back(
                pipelines::registration::PoseGraphNode(trans_odometry));
        const int n_keyframes_per_n_frame =
                config_["n_keyframes_per_n_frame"].asInt();

        for (int s = sid; s < eid; ++s) {
            for (int t = s + 1; t < eid; ++t) {
                // Compute odometry.
                if (t == s + 1) {
                    utility::LogInfo(
                            "Fragment {:03d} / {:03d} :: RGBD odometry between "
                            "frame : "
                            "{} and {}",
                            fragment_id, n_fragments_ - 1, s, t);
                    const auto result = RegisterRGBDPair(
                            s, t, color_files, depth_files, intrinsic);
                    trans_odometry = std::get<1>(result) * trans_odometry;
                    pose_graph.nodes_.push_back(
                            pipelines::registration::PoseGraphNode(
                                    trans_odometry.inverse()));
                    pose_graph.edges_.push_back(
                            pipelines::registration::PoseGraphEdge(
                                    s - sid, t - sid, std::get<1>(result),
                                    std::get<2>(result), false));
                    // Keyframe loop closure.
                } else if (s % n_keyframes_per_n_frame == 0 &&
                           t % n_keyframes_per_n_frame == 0) {
                    utility::LogInfo(
                            "Fragment {:03d} / {:03d} :: RGBD loop closure "
                            "between "
                            "frame : "
                            "{} and {}",
                            fragment_id, n_fragments_ - 1, s, t);
                    const auto result = RegisterRGBDPair(
                            s, t, color_files, depth_files, intrinsic);
                    if (std::get<0>(result)) {
                        pose_graph.edges_.push_back(
                                pipelines::registration::PoseGraphEdge(
                                        s - sid, t - sid, std::get<1>(result),
                                        std::get<2>(result), true));
                    }
                }
            }
        }

        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_fragment_posegraph"].asString() +
                                "fragment_" + PadZeroToNumber(fragment_id, 3) +
                                ".json"),
                pose_graph);

        // Optimize pose graph.
        OptimizePoseGraph(
                config_["depth_diff_max"].asDouble(),
                config_["preference_loop_closure_odometry"].asDouble(),
                pose_graph);

        // Write optimized pose graph.
        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_fragment_posegraph_optimized"]
                                        .asString() +
                                "fragment_optimized_" +
                                PadZeroToNumber(fragment_id, 3) + ".json"),
                pose_graph);

        IntegrateFragmentRGBD(fragment_id, color_files, depth_files, intrinsic,
                              pose_graph);
    }

    bool MakePoseGraphForScene(
            pipelines::registration::PoseGraph& scene_pose_graph) {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        MakeCleanFolder(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_scene"].asString()));
        const auto ply_files = ReadPlyFiles(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_fragment"].asString()));
        if (ply_files.size() == 0) {
            utility::LogWarning("No ply files found.");
            return false;
        }
        n_fragments_ = ply_files.size();

        Eigen::Matrix4d odom = Eigen::Matrix4d::Identity();
        scene_pose_graph.nodes_.push_back(
                pipelines::registration::PoseGraphNode(odom));

        std::vector<MatchingResult> fragment_matching_results;
        for (int i = 0; i < n_fragments_; i++) {
            for (int j = i + 1; j < n_fragments_; j++) {
                fragment_matching_results.push_back(MatchingResult(i, j));
            }
        }

        const size_t num_pairs = fragment_matching_results.size();
        if (config_["multi_threading"].asBool()) {
#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
            for (int i = 0; i < (int)num_pairs; i++) {
                RegisterFragmentPair(ply_files, fragment_matching_results[i].s_,
                                     fragment_matching_results[i].t_,
                                     fragment_matching_results[i]);
            }
        } else {
            for (size_t i = 0; i < num_pairs; i++) {
                RegisterFragmentPair(ply_files, fragment_matching_results[i].s_,
                                     fragment_matching_results[i].t_,
                                     fragment_matching_results[i]);
            }
        }

        for (size_t i = 0; i < num_pairs; i++) {
            if (fragment_matching_results[i].success_) {
                const int& t = fragment_matching_results[i].t_;
                const int& s = fragment_matching_results[i].s_;
                const Eigen::Matrix4d& pose =
                        fragment_matching_results[i].transformation_;
                const Eigen::Matrix6d info =
                        fragment_matching_results[i].information_;
                if (s + 1 == t) {
                    odom = pose * odom;
                    const Eigen::Matrix4d& odom_inv = odom.inverse();
                    scene_pose_graph.nodes_.push_back(
                            pipelines::registration::PoseGraphNode(odom_inv));
                    scene_pose_graph.edges_.push_back(
                            pipelines::registration::PoseGraphEdge(
                                    s, t, pose, info, false));
                } else {
                    scene_pose_graph.edges_.push_back(
                            pipelines::registration::PoseGraphEdge(s, t, pose,
                                                                   info, true));
                }
            }
        }

        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_posegraph"].asString()),
                scene_pose_graph);
        return true;
    }

    void OptimizePoseGraph(double max_correspondence_distance,
                           double preference_loop_closure,
                           pipelines::registration::PoseGraph& pose_graph) {
        // Display messages from GlobalOptimization.
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
        pipelines::registration::GlobalOptimizationOption option(
                max_correspondence_distance, 0.25, preference_loop_closure, 0);

        pipelines::registration::GlobalOptimization(
                pose_graph,
                pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
                pipelines::registration::
                        GlobalOptimizationConvergenceCriteria(),
                option);
        utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    }

    void RefineFragments() {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        const auto ply_files = ReadPlyFiles(utility::filesystem::JoinPath(
                config_["path_dataset"].asString(),
                config_["folder_fragment"].asString()));

        pipelines::registration::PoseGraph scene_pose_graph;
        io::ReadPoseGraph(utility::filesystem::JoinPath(
                                  config_["path_dataset"].asString(),
                                  config_["template_global_posegraph_optimized"]
                                          .asString()),
                          scene_pose_graph);

        std::vector<MatchingResult> fragment_matching_results;

        for (auto& edge : scene_pose_graph.edges_) {
            const int s = edge.source_node_id_;
            const int t = edge.target_node_id_;
            MatchingResult mr(s, t);
            mr.transformation_ = edge.transformation_;
            fragment_matching_results.push_back(mr);
        }

        if (config_["multi_threading"].asBool()) {
#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
            for (int i = 0; i < (int)fragment_matching_results.size(); i++) {
                const int s = fragment_matching_results[i].s_;
                const int t = fragment_matching_results[i].t_;
                RefineFragmentPair(ply_files, s, t,
                                   fragment_matching_results[i]);
            }
        } else {
            for (size_t i = 0; i < fragment_matching_results.size(); i++) {
                const int s = fragment_matching_results[i].s_;
                const int t = fragment_matching_results[i].t_;
                RefineFragmentPair(ply_files, s, t,
                                   fragment_matching_results[i]);
            }
        }

        // Update scene pose graph.
        scene_pose_graph.edges_.clear();
        scene_pose_graph.nodes_.clear();
        Eigen::Matrix4d odom = Eigen::Matrix4d::Identity();
        scene_pose_graph.nodes_.push_back(
                pipelines::registration::PoseGraphNode(odom));
        for (auto& result : fragment_matching_results) {
            const int s = result.s_;
            const int t = result.t_;
            const Eigen::Matrix4d& pose = result.transformation_;
            const Eigen::Matrix6d& info = result.information_;

            if (s + 1 == t) {
                odom = pose * odom;
                scene_pose_graph.nodes_.push_back(
                        pipelines::registration::PoseGraphNode(odom.inverse()));
                scene_pose_graph.edges_.push_back(
                        pipelines::registration::PoseGraphEdge(s, t, pose, info,
                                                               false));
            } else {
                scene_pose_graph.edges_.push_back(
                        pipelines::registration::PoseGraphEdge(s, t, pose, info,
                                                               true));
            }
        }

        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_refined_posegraph"].asString()),
                scene_pose_graph);

        OptimizePoseGraph(
                config_["voxel_size"].asDouble() * 1.4,
                config_["preference_loop_closure_registration"].asDouble(),
                scene_pose_graph);

        io::WritePoseGraph(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_refined_posegraph_optimized"]
                                .asString()),
                scene_pose_graph);
    }

    void RefineFragmentPair(const std::vector<std::string>& pcd_files,
                            int s,
                            int t,
                            MatchingResult& matched_result) {
        geometry::PointCloud source_pcd, target_pcd;
        io::ReadPointCloud(pcd_files[s], source_pcd);
        io::ReadPointCloud(pcd_files[t], target_pcd);
        const double voxel_size = config_["voxel_size"].asDouble();

        // Preprocessing the fragments point clouds.
        geometry::PointCloud source_pcd_down, target_pcd_down;
        pipelines::registration::Feature source_features, target_features;
        std::tie(source_pcd_down, source_features) =
                PreProcessPointCloud(source_pcd, voxel_size);
        std::tie(target_pcd_down, target_features) =
                PreProcessPointCloud(target_pcd, voxel_size);

        const auto& init_trans = matched_result.transformation_;
        const auto result =
                MultiScaleICP(source_pcd_down, target_pcd_down,
                              {voxel_size, voxel_size / 2.0, voxel_size / 4.0},
                              {50, 30, 15}, init_trans);
        matched_result.transformation_ = std::get<0>(result);
        matched_result.information_ = std::get<1>(result);
    }

    void IntegrateFragmentRGBD(
            int fragment_id,
            const std::vector<std::string>& color_files,
            const std::vector<std::string>& depth_files,
            const camera::PinholeCameraIntrinsic& intrinsic,
            const pipelines::registration::PoseGraph& pose_graph) {
        geometry::PointCloud fragment;
        const size_t graph_num = pose_graph.nodes_.size();

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int i = 0; i < int(graph_num); ++i) {
            const int i_abs =
                    fragment_id * config_["n_frames_per_fragment"].asInt() + i;
            utility::LogInfo(
                    "Fragment {:03d} / {:03d} :: Integrate rgbd frame {:d} "
                    "({:d} "
                    "of "
                    "{:d}).",
                    fragment_id, n_fragments_ - 1, i_abs, i + 1, graph_num);
            const geometry::RGBDImage rgbd = ReadRGBDImage(
                    color_files[i_abs], depth_files[i_abs], false);
            auto pcd = geometry::PointCloud::CreateFromRGBDImage(
                    rgbd, intrinsic, Eigen::Matrix4d::Identity(), true);
            pcd->Transform(pose_graph.nodes_[i].pose_);
#pragma omp critical
            { fragment += *pcd; }
        }

        const geometry::PointCloud fragment_down = *fragment.VoxelDownSample(
                config_["tsdf_cubic_size"].asDouble() / 512.0);
        io::WritePointCloud(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_fragment_pointcloud"].asString() +
                                "fragment_" + PadZeroToNumber(fragment_id, 3) +
                                ".ply"),
                fragment_down, {false, true, false});
    }

    void IntegrateSceneRGBDTSDF(
            const std::tuple<std::vector<std::string>,
                             std::vector<std::string>>& rgbd_files,
            const camera::PinholeCameraTrajectory& camera_trajectory) {
        const camera::PinholeCameraIntrinsic intrinsic = GetCameraIntrinsic();

        pipelines::integration::ScalableTSDFVolume volume(
                config_["tsdf_cubic_size"].asDouble() / 512.0, 0.04,
                pipelines::integration::TSDFVolumeColorType::RGB8);

        const auto color_files = std::get<0>(rgbd_files);
        const auto depth_files = std::get<1>(rgbd_files);
        const size_t num = color_files.size();
        for (size_t i = 0; i < num; i++) {
            utility::LogInfo("Scene :: Integrate rgbd frame {} | {}", i, num);
            const geometry::RGBDImage rgbd =
                    ReadRGBDImage(color_files[i], depth_files[i], false);
            volume.Integrate(
                    rgbd, intrinsic,
                    camera_trajectory.parameters_[i].extrinsic_.inverse());
        }

        const auto mesh = volume.ExtractTriangleMesh();
        mesh->ComputeVertexNormals();

        if (config_["debug_mode"].asBool()) {
            visualization::DrawGeometries({mesh});
        }

        io::WriteTriangleMesh(
                utility::filesystem::JoinPath(
                        config_["path_dataset"].asString(),
                        config_["template_global_mesh"].asString()),
                *mesh, false, true);
    }

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> RegisterRGBDPair(
            int s,
            int t,
            const std::vector<std::string>& color_files,
            const std::vector<std::string>& depth_files,
            const camera::PinholeCameraIntrinsic& intrinsic) {
        const geometry::RGBDImage source_rgbd_image =
                ReadRGBDImage(color_files[s], depth_files[s], true);
        const geometry::RGBDImage target_rgbd_image =
                ReadRGBDImage(color_files[t], depth_files[t], true);

        if (abs(s - t) != 1) {
            Eigen::Matrix4d odo_init = PoseEstimation(
                    source_rgbd_image, target_rgbd_image, intrinsic);
            if (!odo_init.isIdentity(1e-8)) {
                return ComputeOdometry(source_rgbd_image, target_rgbd_image,
                                       odo_init, intrinsic);
            } else {
                return std::make_tuple(false, Eigen::Matrix4d::Identity(),
                                       Eigen::Matrix6d::Identity());
            }
        } else {
            return ComputeOdometry(source_rgbd_image, target_rgbd_image,
                                   Eigen::Matrix4d::Identity(), intrinsic);
        }
    }

    void RegisterFragmentPair(const std::vector<std::string>& pcd_files,
                              int s,
                              int t,
                              MatchingResult& matched_result) {
        geometry::PointCloud source_pcd, target_pcd;
        io::ReadPointCloud(pcd_files[s], source_pcd);
        io::ReadPointCloud(pcd_files[t], target_pcd);
        const double voxel_size = config_["voxel_size"].asDouble();

        // Preprocessing the fragments point clouds.
        geometry::PointCloud source_pcd_down, target_pcd_down;
        pipelines::registration::Feature source_features, target_features;
        std::tie(source_pcd_down, source_features) =
                PreProcessPointCloud(source_pcd, voxel_size);
        std::tie(target_pcd_down, target_features) =
                PreProcessPointCloud(target_pcd, voxel_size);

        Eigen::Matrix4d pose;
        Eigen::Matrix6d info;

        // Odometry estimation.
        if (s + 1 == t) {
            utility::LogInfo("Fragment odometry {} and {}", s, t);
            pipelines::registration::PoseGraph pose_graph_frag;
            io::ReadPoseGraph(
                    utility::filesystem::JoinPath(
                            config_["path_dataset"].asString(),
                            config_["template_fragment_posegraph_optimized"]
                                            .asString() +
                                    "fragment_optimized_" +
                                    PadZeroToNumber(s, 3) + ".json"),
                    pose_graph_frag);
            const int n_nodes = pose_graph_frag.nodes_.size();
            const Eigen::Matrix4d init_trans =
                    pose_graph_frag.nodes_[n_nodes - 1].pose_.inverse();
            const auto result = MultiScaleICP(source_pcd_down, target_pcd_down,
                                              {voxel_size}, {50}, init_trans);
            pose = std::get<0>(result);
            info = std::get<1>(result);
            matched_result.success_ = true;
            matched_result.transformation_ = pose;
            matched_result.information_ = info;
        } else {
            // Loop closure estimation.
            utility::LogInfo("Fragment loop closure {} and {}", s, t);
            const auto result = ComputeInitialRegistration(
                    source_pcd_down, target_pcd_down, source_features,
                    target_features, voxel_size * 1.4);
            const bool success = std::get<0>(result);
            if (!success) {
                utility::LogWarning(
                        "Global registration failed. Skip pair ({} | {}).", s,
                        t);
                matched_result.success_ = false;
                matched_result.transformation_ = Eigen::Matrix4d::Identity();
                matched_result.information_ = Eigen::Matrix6d::Identity();
            } else {
                pose = std::get<1>(result);
                info = std::get<2>(result);
                matched_result.success_ = true;
                matched_result.transformation_ = pose;
                matched_result.information_ = info;
            }
        }

        if (config_["debug_mode"].asBool()) {
            DrawRegistrationResult(source_pcd, target_pcd, pose);
        }
    }

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
    ComputeInitialRegistration(
            const geometry::PointCloud& src_pcd,
            const geometry::PointCloud& dst_pcd,
            const pipelines::registration::Feature& src_features,
            const pipelines::registration::Feature& dst_features,
            double distance_threshold) {
        const auto result =
                GlobalRegistration(src_pcd, dst_pcd, src_features, dst_features,
                                   distance_threshold);
        const Eigen::Matrix4d pose = std::get<1>(result);
        const Eigen::Matrix6d info =
                pipelines::registration::GetInformationMatrixFromPointClouds(
                        src_pcd, dst_pcd, distance_threshold, pose);
        if (info(5, 5) /
                    std::min(src_pcd.points_.size(), dst_pcd.points_.size()) <
            0.3) {
            return std::make_tuple(false, pose, Eigen::Matrix6d::Identity());
        }

        return std::make_tuple(true, pose, info);
    }

    std::tuple<geometry::PointCloud, pipelines::registration::Feature>
    PreProcessPointCloud(const geometry::PointCloud& pcd, double voxel_size) {
        auto pcd_down = pcd.VoxelDownSample(voxel_size);

        if (!pcd_down->HasNormals()) {
            pcd_down->EstimateNormals(
                    geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));
        }
        pcd_down->OrientNormalsTowardsCameraLocation();

        const auto fpfh = pipelines::registration::ComputeFPFHFeature(
                *pcd_down,
                geometry::KDTreeSearchParamHybrid(voxel_size * 5, 100));
        return std::make_tuple(*pcd_down, *fpfh);
    }

    Eigen::Matrix4d PoseEstimation(
            const geometry::RGBDImage& src,
            const geometry::RGBDImage& dst,
            const camera::PinholeCameraIntrinsic& intrinsic) {
        const auto src_pcd = geometry::PointCloud::CreateFromRGBDImage(
                src, intrinsic, Eigen::Matrix4d::Identity(), true);
        const auto dst_pcd = geometry::PointCloud::CreateFromRGBDImage(
                dst, intrinsic, Eigen::Matrix4d::Identity(), true);

        geometry::PointCloud src_pcd_down, dst_pcd_down;
        pipelines::registration::Feature src_features, dst_features;

        // Increase the voxel size to accelerate the point cloud FPFH features
        // extraction.
        std::tie(src_pcd_down, src_features) = PreProcessPointCloud(
                *src_pcd, config_["voxel_size"].asDouble() * 1.5);
        std::tie(dst_pcd_down, dst_features) = PreProcessPointCloud(
                *dst_pcd, config_["voxel_size"].asDouble() * 1.5);

        const double distance_threshold =
                config_["voxel_size"].asDouble() * 1.4;

        const auto registration =
                GlobalRegistration(src_pcd_down, dst_pcd_down, src_features,
                                   dst_features, distance_threshold);
        return std::get<1>(registration);
    }

    std::tuple<bool, Eigen::Matrix4d> GlobalRegistration(
            const geometry::PointCloud& src_pcd,
            const geometry::PointCloud& dst_pcd,
            const pipelines::registration::Feature& src_features,
            const pipelines::registration::Feature& dst_features,
            double distance_threshold) {
        pipelines::registration::RegistrationResult result;
        if (config_["global_registration"] == "fgr") {
            result = pipelines::registration::
                    FastGlobalRegistrationBasedOnFeatureMatching(
                            src_pcd, dst_pcd, src_features, dst_features,
                            pipelines::registration::
                                    FastGlobalRegistrationOption(
                                            distance_threshold));
        } else {
            std::vector<std::reference_wrapper<
                    const pipelines::registration::CorrespondenceChecker>>
                    checkers;
            auto edge_length_checker = pipelines::registration::
                    CorrespondenceCheckerBasedOnEdgeLength(0.9);
            checkers.push_back(edge_length_checker);
            auto distance_checer = pipelines::registration::
                    CorrespondenceCheckerBasedOnDistance(distance_threshold);
            checkers.push_back(distance_checer);
            result = pipelines::registration::
                    RegistrationRANSACBasedOnFeatureMatching(
                            src_pcd, dst_pcd, src_features, dst_features, false,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            4, checkers,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    1000000, 0.999));
        }
        if (result.transformation_.isIdentity(1e-8)) {
            return std::make_tuple(false, Eigen::Matrix4d::Identity());
        }
        return std::make_tuple(true, result.transformation_);
    }

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> ComputeOdometry(
            const geometry::RGBDImage& src,
            const geometry::RGBDImage& dst,
            const Eigen::Matrix4d& init_trans,
            const camera::PinholeCameraIntrinsic& intrinsic) {
        pipelines::odometry::OdometryOption option;
        option.depth_diff_max_ = config_["depth_diff_max"].asDouble();
        return pipelines::odometry::ComputeRGBDOdometry(
                src, dst, intrinsic, init_trans,
                pipelines::odometry::RGBDOdometryJacobianFromHybridTerm(),
                option);
    }

    std::tuple<Eigen::Matrix4d, Eigen::Matrix6d> MultiScaleICP(
            const geometry::PointCloud& src,
            const geometry::PointCloud& dst,
            const std::vector<double>& voxel_size,
            const std::vector<int>& max_iter,
            const Eigen::Matrix4d& init_trans = Eigen::Matrix4d::Identity()) {
        Eigen::Matrix4d current = init_trans;
        Eigen::Matrix6d info;
        const size_t num_scale = voxel_size.size();
        for (size_t i = 0; i < num_scale; i++) {
            const double max_dis = config_["voxel_szie"].asDouble() * 1.4;
            const auto src_down = src.VoxelDownSample(voxel_size[i]);
            const auto dst_down = dst.VoxelDownSample(voxel_size[i]);
            const pipelines::registration::ICPConvergenceCriteria criteria(
                    1e-6, 1e-6, max_iter[i]);
            pipelines::registration::RegistrationResult result;
            if (config_["icp_method"].asString() == "point_to_point") {
                result = pipelines::registration::RegistrationICP(
                        *src_down, *dst_down, max_dis, current,
                        pipelines::registration::
                                TransformationEstimationPointToPoint(),
                        criteria);
            } else if (config_["icp_method"].asString() == "point_to_plane") {
                result = pipelines::registration::RegistrationICP(
                        *src_down, *dst_down, max_dis, current,
                        pipelines::registration::
                                TransformationEstimationPointToPlane(),
                        criteria);
            } else if (config_["icp_method"].asString() == "color") {
                // Colored ICP is sensitive to threshold.
                // Fallback to preset distance threshold that works better.
                result = pipelines::registration::RegistrationColoredICP(
                        *src_down, *dst_down, voxel_size[i], current,
                        pipelines::registration::
                                TransformationEstimationForColoredICP(),
                        criteria);
            } else if (config_["icp_method"].asString() == "generalized") {
                result = pipelines::registration::RegistrationGeneralizedICP(
                        *src_down, *dst_down, max_dis, current,
                        pipelines::registration::
                                TransformationEstimationForGeneralizedICP(),
                        criteria);
            } else {
                utility::LogError("Unknown icp method.");
            }
            current = result.transformation_;
            if (i == num_scale - 1) {
                info = pipelines::registration::
                        GetInformationMatrixFromPointClouds(
                                src, dst, voxel_size[i] * 1.4, current);
            }
        }

        if (config_["debug_mode"].asBool()) {
            DrawRegistrationResult(src, dst, current, true);
        }

        return std::make_tuple(current, info);
    }
};

}  // namespace offline_reconstruction
}  // namespace apps
}  // namespace open3d
