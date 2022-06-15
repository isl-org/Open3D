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

#include <json/json.h>

#include <atomic>
#include <iomanip>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

namespace open3d {
namespace examples {
namespace legacy_reconstruction {

// ============== Helper functions for file system ==============
std::string PadZeroToNumber(int num, int size) {
    std::string s = std::to_string(num);
    while (s.size() < size) {
        s = "0" + s;
    }
    return s;
}

std::string ElapseTimeToHMS(double seconds) {
    int h = seconds / 3600;
    int m = (seconds - h * 3600) / 60;
    int s = seconds - h * 3600 - m * 60;

    std::string hs = std::to_string(h);
    std::string ms = std::to_string(m);
    std::string ss = std::to_string(s);

    if (h < 10) {
        hs = "0" + hs;
    }
    if (m < 10) {
        ms = "0" + ms;
    }
    if (s < 10) {
        ss = "0" + ss;
    }
    return hs + ":" + ms + ":" + ss;
}

std::string FloatToString(float f, int precision = 3) {
    std::stringstream oss;
    oss << std::fixed << std::setprecision(precision) << f;
    return oss.str();
}

std::string JoinPath(const std::string& path1, const std::string& path2) {
    std::string path = path1;
    if (path.back() != '/') {
        path += "/";
    }
    path += path2;
    return path;
}

std::string JoinPath(const std::vector<std::string>& paths) {
    std::string path = paths[0];
    for (size_t i = 1; i < paths.size(); i++) {
        path = JoinPath(path, paths[i]);
    }
    return path;
}

std::string AddIfExist(const std::string& path,
                       const std::vector<std::string>& folder_names) {
    for (auto& folder_name : folder_names) {
        const std::string folder_path = JoinPath(path, folder_name);
        if (utility::filesystem::DirectoryExists(folder_path)) {
            return folder_path;
        }
    }
    utility::LogError("None of the folders {} found in {}", folder_names, path);
}

bool CheckFolderStructure(const std::string& path_dataset) {
    if (utility::filesystem::FileExists(path_dataset) &&
        utility::filesystem::GetFileExtensionInLowerCase(path_dataset) ==
                "bag") {
        return true;
    }
    const std::string path_color =
            AddIfExist(path_dataset, {"color", "rgb", "image"});
    const std::string path_depth = JoinPath(path_dataset, "depth");
    if (!utility::filesystem::DirectoryExists(path_color) ||
        !utility::filesystem::DirectoryExists(path_depth)) {
        utility::LogWarning("Folder structure of {} is not correct",
                            path_dataset);
        return false;
    }
    return true;
}

void MakeCleanFolder(const std::string& path) {
    if (utility::filesystem::DirectoryExists(path)) {
        utility::filesystem::DeleteDirectory(path);
    }
    utility::filesystem::MakeDirectory(path);
}

std::tuple<std::string, std::string> GetRGBDFolders(
        const std::string& path_dataset) {
    return std::make_tuple(
            AddIfExist(path_dataset, {"image/", "rgb/", "color/"}),
            JoinPath(path_dataset, "depth/"));
}

std::tuple<std::vector<std::string>, std::vector<std::string>> ReadRGBDFiles(
        const std::string& path) {
    std::string path_rgb, path_depth;
    std::tie(path_rgb, path_depth) = GetRGBDFolders(path);

    std::vector<std::string> color_files, depth_files;
    utility::filesystem::ListFilesInDirectoryWithExtension(path_rgb, "png",
                                                           color_files);
    if (color_files.empty()) {
        utility::filesystem::ListFilesInDirectoryWithExtension(path_rgb, "jpg",
                                                               color_files);
    }
    utility::filesystem::ListFilesInDirectoryWithExtension(path_depth, "png",
                                                           depth_files);

    if (color_files.size() != depth_files.size()) {
        utility::LogError(
                "Number of color {} and depth {} images are not equal.",
                color_files.size(), depth_files.size());
    }
    return std::make_tuple(color_files, depth_files);
}

std::vector<std::string> ReadPlyFiles(const std::string& path) {
    std::vector<std::string> ply_files;
    utility::filesystem::ListFilesInDirectoryWithExtension(path, "ply",
                                                           ply_files);
    return ply_files;
}

std::tuple<std::string, std::string, float> ExtractRGBDFrames(
        const std::string& rgbd_video_file) {
    const std::string frames_folder =
            utility::filesystem::GetFileParentDirectory(rgbd_video_file);
    const std::string path_intrinsic = frames_folder + "intrinsic.json";
    if (!utility::filesystem::FileExists(path_intrinsic)) {
        utility::LogError("Intrinsic file not found: {}", path_intrinsic);
    } else {
        auto rgbd_video = t::io::RGBDVideoReader::Create(rgbd_video_file);
        rgbd_video->SaveFrames(frames_folder);
    }

    Json::Value intrinsic = utility::StringToJson(path_intrinsic);
    return std::make_tuple(frames_folder, path_intrinsic,
                           intrinsic["depth_scale"].asFloat());
}

// ============== Helper functions for json configuration ==============
void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     int default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     float default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void SetDefaultValue(Json::Value& config,
                     const std::string& key,
                     const std::string& default_value) {
    if (!config.isMember(key)) {
        config[key] = default_value;
    }
}

void InitConfig(Json::Value& config) {
    // Set default parameters if not specified.
    SetDefaultValue(config, "n_frames_per_fragment", 100);
    SetDefaultValue(config, "n_keyframes_per_n_frame", 5);
    SetDefaultValue(config, "depth_min", 0.3f);
    SetDefaultValue(config, "depth_max", 3.0f);
    SetDefaultValue(config, "voxel_size", 0.05f);
    SetDefaultValue(config, "depth_diff_max", 0.07f);
    SetDefaultValue(config, "depth_scale", 1000);
    SetDefaultValue(config, "preference_loop_closure_odometry", 0.1f);
    SetDefaultValue(config, "preference_loop_closure_registration", 5.0f);
    SetDefaultValue(config, "tsdf_cubic_size", 3.0f);
    SetDefaultValue(config, "icp_method", "color");
    SetDefaultValue(config, "global_registration", "ransac");
    SetDefaultValue(config, "multi_threading", true);

    // `slac` and `slac_integrate` related parameters. `voxel_size` and
    // `depth_min` parameters from previous section, are also used in `slac`
    // and `slac_integrate`.
    SetDefaultValue(config, "max_iterations", 5);
    SetDefaultValue(config, "sdf_trunc", 0.04f);
    SetDefaultValue(config, "block_count", 40000);
    SetDefaultValue(config, "distance_threshold", 0.07f);
    SetDefaultValue(config, "fitness_threshold", 0.3f);
    SetDefaultValue(config, "regularizer_weight", 1);
    SetDefaultValue(config, "method", "slac");
    SetDefaultValue(config, "device", "CPU:0");
    SetDefaultValue(config, "save_output_as", "pointcloud");
    SetDefaultValue(config, "folder_slac", "slac/");
    SetDefaultValue(config, "template_optimized_posegraph_slac",
                    "optimized_posegraph_slac.json");

    // Path related parameters.
    SetDefaultValue(config, "folder_fragment", "fragments/");
    SetDefaultValue(
            config, "subfolder_slac",
            "slac/" + FloatToString(config["voxel_size"].asFloat(), 3) + "/");
    SetDefaultValue(config, "template_fragment_posegraph", "fragments/");
    SetDefaultValue(config, "template_fragment_posegraph_optimized",
                    "fragments/");
    SetDefaultValue(config, "template_fragment_pointcloud", "fragments/");
    SetDefaultValue(config, "folder_scene", "scene/");
    SetDefaultValue(config, "template_global_posegraph",
                    "scene/global_registration.json");
    SetDefaultValue(config, "template_global_posegraph_optimized",
                    "scene/global_registration_optimized.json");
    SetDefaultValue(config, "template_refined_posegraph",
                    "scene/refined_registration.json");
    SetDefaultValue(config, "template_refined_posegraph_optimized",
                    "scene/refined_registration_optimized.json");
    SetDefaultValue(config, "template_global_mesh", "scene/integrated.ply");
    SetDefaultValue(config, "template_global_traj", "scene/trajectory.log");

    if (utility::filesystem::GetFileExtensionInLowerCase(
                config["path_dataset"].asString()) == "bag") {
        std::tie(config["path_dataset"], config["path_intrinsic"],
                 config["depth_scale"]) =
                ExtractRGBDFrames(config["path_dataset"].asString());
    }
}

void LoungeDataLoader(Json::Value& config) {
    utility::LogInfo("Loading Stanford Lounge RGB-D Dataset");

    data::LoungeRGBDImages rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 3.0f;
    config["voxel_size"] = 0.05f;
    config["depth_diff_max"] = 0.07f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 3.0f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

void BedroomDataLoader(Json::Value& config) {
    utility::LogInfo("Loading Redwood Bedroom RGB-D Dataset");

    data::BedroomRGBDImages rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 3.0f;
    config["voxel_size"] = 0.05f;
    config["depth_diff_max"] = 0.07f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 3.0f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

void JackJackroomDataLoader(Json::Value& config) {
    utility::LogInfo("Loading RealSense L515 Jack-Jack RGB-D Dataset");

    data::JackJackL515Bag rgbd;

    // Set dataset specific parameters.
    config["path_dataset"] = rgbd.GetExtractDir();
    config["path_intrinsic"] = "";
    config["depth_max"] = 0.85f;
    config["voxel_size"] = 0.025f;
    config["depth_diff_max"] = 0.03f;
    config["preference_loop_closure_odometry"] = 0.1f;
    config["preference_loop_closure_registration"] = 5.0f;
    config["tsdf_cubic_size"] = 0.75f;
    config["icp_method"] = "color";
    config["global_registration"] = "ransac";
    config["multi_threading"] = true;
}

Json::Value DefaultDatasetLoader(const std::string& name) {
    utility::LogInfo("Config file was not passed. Using deafult dataset: {}.",
                     name);
    Json::Value config;
    if (name == "lounge") {
        LoungeDataLoader(config);
    } else if (name == "bedroom") {
        BedroomDataLoader(config);
    } else if (name == "jack_jack") {
        JackJackroomDataLoader(config);
    } else {
        utility::LogError("Dataset {} is not supported.", name);
    }

    InitConfig(config);
    utility::LogInfo("Loaded data from {}", config["path_dataset"].asString());

    return config;
}

// ============== Helper functions for drawing results ==============
static const Eigen::Matrix4d flip_transformation = Eigen::Matrix4d({
        {1, 0, 0, 0},
        {0, -1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1},
});

void DrawRegistrationResult(const geometry::PointCloud& src,
                            const geometry::PointCloud& dst,
                            const Eigen::Matrix4f& transformation) {
    auto transformed_src = std::make_shared<geometry::PointCloud>(src);
    auto transformed_dst = std::make_shared<geometry::PointCloud>(dst);
    transformed_src->PaintUniformColor(Eigen::Vector3d(1, 0.706, 0));
    transformed_dst->PaintUniformColor(Eigen::Vector3d(0, 0.651, 0.929));
    transformed_src->Transform(flip_transformation * transformation);
    transformed_dst->Transform(flip_transformation);
    visualization::DrawGeometries({transformed_src, transformed_dst});
}

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

class OdometryTrajectory {
public:
    OdometryTrajectory() {}
    ~OdometryTrajectory() {}

public:
    bool WriteToJsonFile(const std::string& file_name);
    bool ReadFromJsonFile(const std::string& file_name);

public:
    std::vector<Eigen::Matrix4d> odomtry_list_;
};

class ReconstructionPipeline {
public:
    /**
     * @brief Construct a new Reconstruction Pipeline given config.
     *
     * @param config
     */
    explicit ReconstructionPipeline(const Json::Value& config)
        : config_(config) {}

    virtual ~ReconstructionPipeline() {}

private:
    Json::Value config_;
    int n_fragments_;

public:
    /**
     * @brief Make fragments from raw RGBD images.
     */
    void MakeFragments() {
        utility::LogInfo("Making fragments from RGBD sequence.");
        MakeCleanFolder(JoinPath(config_["path_dataset"].asString(),
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

    /**
     * @brief Register fragments and compute global odometry.
     */
    void RegisterFragments() {
        utility::LogInfo("Registering fragments.");
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        MakeCleanFolder(JoinPath(config_["path_dataset"].asString(),
                                 config_["folder_scene"].asString()));

        pipelines::registration::PoseGraph scene_pose_graph;
        MakePoseGraphForScene(scene_pose_graph);

        // Optimize pose graph for scene.
        OptimizePoseGraph(
                config_["voxel_size"].asDouble() * 1.4,
                config_["preference_loop_closure_registration"].asDouble(),
                scene_pose_graph);
        io::WritePoseGraph(
                JoinPath(config_["path_dataset"].asString(),
                         config_["template_global_posegraph_optimized"]
                                 .asString()),
                scene_pose_graph);
    }

    /**
     * @brief Refine fragments registration and re-compute global odometry.
     *
     */
    void RefineFragments() {}

    /**
     * @brief Integrate RGBD images with global odometry.
     */
    void IntegrateScene() {}

    /**
     * @brief Run SLAC optimization or fragments.
     */
    void SLAC() {}

    /**
     * @brief integrate scene using SLAC results.
     */
    void IntegrateSceneSLAC() {}

private:
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

    bool ReadFragmentData();

    void ReadJsonPipelineConfig(const std::string& file_name);

    void ProcessSingleFragment(int fragment_id,
                               const std::vector<std::string>& color_files,
                               const std::vector<std::string>& depth_files) {
        camera::PinholeCameraIntrinsic intrinsic;
        if (config_.isMember("path_intrinsic")) {
            io::ReadIJsonConvertible(config_["path_intrinsic"].asString(),
                                     intrinsic);
        } else {
            intrinsic = camera::PinholeCameraIntrinsic(
                    camera::PinholeCameraIntrinsicParameters::
                            PrimeSenseDefault);
        }

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
        utility::SetVerbosityLevel(utility::VerbosityLevel::Error);

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
                JoinPath(config_["path_dataset"].asString(),
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
                JoinPath(config_["path_dataset"].asString(),
                         config_["template_fragment_posegraph_optimized"]
                                         .asString() +
                                 "fragment_optimized_" +
                                 PadZeroToNumber(fragment_id, 3) + ".json"),
                pose_graph);

        IntegrateFragmentRGBD(fragment_id, color_files, depth_files, intrinsic,
                              pose_graph);
    }

    void MakePoseGraphForScene(
            pipelines::registration::PoseGraph& scene_pose_graph) {
        utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

        MakeCleanFolder(JoinPath(config_["path_dataset"].asString(),
                                 config_["folder_scene"].asString()));
        const auto ply_files =
                ReadPlyFiles(JoinPath(config_["path_dataset"].asString(),
                                      config_["folder_fragment"].asString()));

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
            std::vector<std::thread> thread_list;
            for (int i = 0; i < num_pairs; i++) {
                thread_list.push_back(std::thread(
                        &ReconstructionPipeline::RegisterFragmentPair, this,
                        fragment_matching_results[i].s_,
                        fragment_matching_results[i].t_,
                        std::ref(fragment_matching_results[i])));
            }
            for (auto& thread : thread_list) {
                thread.join();
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
                            open3d::pipelines::registration::PoseGraphNode(
                                    odom_inv));
                    scene_pose_graph.edges_.push_back(
                            open3d::pipelines::registration::PoseGraphEdge(
                                    s, t, pose, info, false));
                } else {
                    scene_pose_graph.edges_.push_back(
                            open3d::pipelines::registration::PoseGraphEdge(
                                    s, t, pose, info, true));
                }
            }
        }

        io::WritePoseGraph(
                JoinPath(config_["path_dataset"].asString(),
                         config_["template_global_posegraph"].asString()),
                scene_pose_graph);

        return scene_pose_graph;
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
        utility::SetVerbosityLevel(utility::VerbosityLevel::Error);
    }

    void RefineRegistration();

    void SLACOptimization();

    void IntegrateFragmentRGBD(
            int fragment_id,
            const std::vector<std::string>& color_files,
            const std::vector<std::string>& depth_files,
            const camera::PinholeCameraIntrinsic& intrinsic,
            const pipelines::registration::PoseGraph& pose_graph) {
        geometry::PointCloud fragment;
        const size_t graph_num = pose_graph.nodes_.size();

#pragma omp parallel for sheduling(static)
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
                JoinPath(config_["path_dataset"].asString(),
                         config_["template_fragment_pointcloud"].asString() +
                                 "fragment_" + PadZeroToNumber(fragment_id, 3) +
                                 ".ply"),
                fragment_down, {false, true, false});
    }

    void IntegrateSceneRGBDTSDF();

    void IntegrateSceneRGBD();

    void SaveFragmentResults();

    void SaveSceneResults();

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
                    JoinPath(config_["path_dataset"].asString(),
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
        } else {
            // Loop closure estimation.
            utility::LogInfo("Fragment loop closure {} and {}", s, t);
            const auto result = ComputeInitialRegistration(
                    source_pcd, target_pcd, source_features, target_features,
                    voxel_size * 1.4);
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
            }
        }

        if (config_["debug_mode"].asBool()) {
            DrawRegistrationResult(source_pcd, target_pcd, pose);
            utility::LogInfo("Initial transformation:");
            utility::LogInfo("{}", pose);
            utility::LogInfo("Information matrix:");
            utility::LogInfo("{}", info);
        }
    }

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d>
    ReconstructionPipeline::ComputeInitialRegistration(
            const geometry::PointCloud& src_pcd,
            const geometry::PointCloud& dst_pcd,
            const pipelines::registration::Feature& src_features,
            const pipelines::registration::Feature& dst_features,
            double distance_threshold) {
        Eigen::Matrix4d pose;
        bool ret;
        std::tie(ret, pose) =
                GlobalRegistration(src_pcd, dst_pcd, src_features, dst_features,
                                   distance_threshold);

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
        std::tie(src_pcd_down, src_features) = PreProcessPointCloud(
                *src_pcd, config_["voxel_size"].asDouble());
        std::tie(dst_pcd_down, dst_features) = PreProcessPointCloud(
                *dst_pcd, config_["voxel_size"].asDouble());

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
        return std::make_tuple(current, info);
    }
};

}  // namespace legacy_reconstruction
}  // namespace examples
}  // namespace open3d