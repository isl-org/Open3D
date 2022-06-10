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

#include <atomic>
#include <sstream>
#include <thread>

#include "open3d/Open3D.h"

namespace open3d {
namespace examples {
namespace offline_reconstruction {

/// \class PipelineConfig
/// \brief Configuration for the offline reconstruction pipeline.
///
class PipelineConfig {
public:
    PipelineConfig() {
        data_path_ = "";
        depth_scale_ = 1000.0;
        depth_max_ = 3.0;
        depth_diff_max_ = 0.07;
        voxel_size_ = 0.01;
        tsdf_voxel_size_ = 0.005;
        tsdf_integration_ = false;
        enable_slac_ = false;
        make_fragment_param_ = {PipelineConfig::DescriptorType::FPFH, 40, 0.2};
        local_refine_method_ = LocalRefineMethod::ColoredICP;
        global_registration_method_ = GlobalRegistrationMethod::Ransac;
        optimization_param_ = {0.1, 5.0};
    }

    virtual ~PipelineConfig() {}

public:
    enum class DescriptorType { FPFH = 0 };

    struct MakeFragmentParam {
        DescriptorType descriptor_type;
        int n_frame_per_fragment;
        float keyframe_ratio;
    };

    enum class LocalRefineMethod {
        Point2PointICP = 0,
        Point2PlaneICP = 1,
        ColoredICP = 2,
        GeneralizedICP = 3
    };

    enum class GlobalRegistrationMethod { Ransac = 0, FGR = 1 };

    struct OptimizationParam {
        float preference_loop_closure_odometry;
        float preference_loop_closure_registration;
    };

    // Path to data stored folder.
    std::string data_path_;
    open3d::camera::PinholeCameraIntrinsic camera_intrinsic_;
    float depth_scale_;
    float depth_max_;
    float depth_diff_max_;
    float voxel_size_;
    float tsdf_voxel_size_;
    bool tsdf_integration_;
    bool enable_slac_;
    MakeFragmentParam make_fragment_param_;
    LocalRefineMethod local_refine_method_;
    GlobalRegistrationMethod global_registration_method_;
    OptimizationParam optimization_param_;
};

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
    explicit ReconstructionPipeline(const PipelineConfig& config);

    /**
     * @brief Construct a new Reconstruction Pipeline from file.
     *
     * @param config_file
     */
    ReconstructionPipeline(const std::string& config_file);

    virtual ~ReconstructionPipeline() {}

private:
    PipelineConfig config_;
    std::map<std::string, double> time_cost_table_;

    // Member variables for make fragments.
    std::vector<open3d::geometry::RGBDImage> rgbd_lists_;
    std::vector<open3d::geometry::Image> intensity_img_lists_;
    std::vector<pipelines::registration::Feature> fpfh_lists_;
    std::vector<open3d::pipelines::registration::PoseGraph>
            fragment_pose_graphs_;
    std::vector<open3d::geometry::PointCloud> fragment_point_clouds_;
    int n_fragments_;
    int n_keyframes_per_n_frame_;

    // Member variables for register fragments.
    std::vector<open3d::geometry::PointCloud> preprocessed_fragment_lists_;
    std::vector<open3d::pipelines::registration::Feature> fragment_features_;
    std::vector<MatchingResult> fragment_matching_results_;
    open3d::pipelines::registration::PoseGraph scene_pose_graph_;
    OdometryTrajectory scene_odometry_trajectory_;

public:
    /**
     * @brief Make fragments from raw RGBD images.
     * The output will be the fragment point clouds and fragment pose graph.
     *
     */
    void MakeFragments();

    /**
     * @brief Register fragments and compute global odometry.
     * The output will be the global odometry trajectory.
     *
     */
    void RegisterFragments();

    /**
     * @brief Integrate RGBD images with global odometry.
     *  The output will be the integrated triangle mesh of scene.
     */
    void IntegrateScene();

    /**
     * @brief Run the whole pipeline.
     *
     */
    void RunSystem();

    /**
     * @brief Get the Data Path
     *
     * @return std::string
     */
    std::string GetDataPath() const { return config_.data_path_; }

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

    void OptimizePoseGraph(
            double max_correspondence_distance,
            double preference_loop_closure,
            open3d::pipelines::registration::PoseGraph& pose_graph);

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

    void PreProcessFragments(open3d::geometry::PointCloud& pcd, int i);

    std::tuple<Eigen::Matrix4d, Eigen::Matrix6d> MultiScaleICP(
            const open3d::geometry::PointCloud& src,
            const open3d::geometry::PointCloud& dst,
            const std::vector<float>& voxel_size,
            const std::vector<int>& max_iter,
            const Eigen::Matrix4d& init_trans = Eigen::Matrix4d::Identity());
};

}  // namespace offline_reconstruction
}  // namespace examples
}  // namespace open3d