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
    std::map<std::string, double> time_cost_table_;

    // Member variables for make fragments.
    std::vector<geometry::RGBDImage> rgbd_lists_;
    std::vector<geometry::Image> intensity_img_lists_;
    std::vector<pipelines::registration::Feature> fpfh_lists_;
    std::vector<pipelines::registration::PoseGraph> fragment_pose_graphs_;
    std::vector<geometry::PointCloud> fragment_point_clouds_;
    int n_fragments_;
    int n_keyframes_per_n_frame_;

    // Member variables for register fragments.
    std::vector<geometry::PointCloud> preprocessed_fragment_lists_;
    std::vector<pipelines::registration::Feature> fragment_features_;
    std::vector<MatchingResult> fragment_matching_results_;
    pipelines::registration::PoseGraph scene_pose_graph_;
    OdometryTrajectory scene_odometry_trajectory_;

public:
    /**
     * @brief Make fragments from raw RGBD images.
     */
    void MakeFragments() {}

    /**
     * @brief Register fragments and compute global odometry.
     */
    void RegisterFragments() {}

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
    void CheckConfig();

    bool ReadRGBDData();

    bool ReadFragmentData();

    void ReadJsonPipelineConfig(const std::string& file_name);

    void ComputeFPFH(const geometry::PointCloud& pcd,
                     pipelines::registration::Feature& feature);

    void BuildSingleFragment(int fragment_id);

    void BuildPoseGraphForFragment(int fragment_id, int sid, int eid);

    void BuildPoseGraphForScene();

    void OptimizePoseGraph(double max_correspondence_distance,
                           double preference_loop_closure,
                           pipelines::registration::PoseGraph& pose_graph);

    void RefineRegistration();

    void SLACOptimization();

    void IntegrateFragmentRGBD(int fragment_id);

    void IntegrateSceneRGBDTSDF();

    void IntegrateSceneRGBD();

    void SaveFragmentResults();

    void SaveSceneResults();

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> RegisterRGBDPair(int s,
                                                                        int t);

    void RefineFragmentPair(int s, int t, MatchingResult& matched_result);

    void RegisterFragmentPair(int s, int t, MatchingResult& matched_result);

    Eigen::Matrix4d PoseEstimation(int s, int t);

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> GlobalRegistration(
            int s, int t);

    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> ComputeOdometry(
            int s, int t, const Eigen::Matrix4d& init_trans);

    void PreProcessFragments(geometry::PointCloud& pcd, int i);

    std::tuple<Eigen::Matrix4d, Eigen::Matrix6d> MultiScaleICP(
            const geometry::PointCloud& src,
            const geometry::PointCloud& dst,
            const std::vector<float>& voxel_size,
            const std::vector<int>& max_iter,
            const Eigen::Matrix4d& init_trans = Eigen::Matrix4d::Identity());
};

}  // namespace legacy_reconstruction
}  // namespace examples
}  // namespace open3d