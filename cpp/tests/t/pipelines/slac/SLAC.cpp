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

#include <string>
#include <vector>

#include "core/CoreTest.h"
#include "open3d/camera/PinholeCameraTrajectory.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PoseGraphIO.h"
#include "open3d/pipelines/registration/PoseGraph.h"
#include "open3d/t/geometry/VoxelBlockGrid.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/slac/SLACOptimizer.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Timer.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class SLACPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(SLAC,
                         SLACPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

// PointCloud is similar if fitness is higher and rmse is lower than tolerance
// threshold.
static bool IsPointCloudSimilar(t::geometry::PointCloud source,
                                t::geometry::PointCloud target,
                                double voxel_size = 0.05,
                                float inlier_fitness_threshold = 0.99,
                                float inlier_rmse_threshold = 0.0001) {
    auto result = t::pipelines::registration::EvaluateRegistration(
            source, target, /*search_distance*/ voxel_size,
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")));
    if (result.fitness_ >= inlier_fitness_threshold &&
        result.inlier_rmse_ <= inlier_rmse_threshold) {
        return true;
    }
    return false;
}

TEST_P(SLACPermuteDevices, DISABLED_RunSLACOptimizerForFragments) {
    core::Device device = GetParam();

    std::string dataset_folder = utility::GetDataPathCommon(
            "reconstruction_system/livingroom1_clean_micro");
    std::string fragment_folder = dataset_folder + "/test_fragments";
    std::string scene_folder = dataset_folder + "/test_scene";
    std::string slac_folder = dataset_folder + "/output_slac";
    std::string test_slac_folder = dataset_folder + "/test_slac";

    std::vector<std::string> fragment_fnames;
    utility::filesystem::ListFilesInDirectoryWithExtension(
            fragment_folder, "ply", fragment_fnames);
    if (fragment_fnames.size() == 0) {
        utility::LogError(
                "No fragment found in {}, please make sure the test dataset "
                "has been downloaded in "
                "GetDataPathDownload(\"tests/reconstruction_system/\").",
                fragment_folder);
    }
    std::sort(fragment_fnames.begin(), fragment_fnames.end());

    std::string pose_graph_fname =
            scene_folder + "/refined_registration_optimized.json";

    auto pose_graph = open3d::io::CreatePoseGraphFromFile(pose_graph_fname);

    // Optimizer Parameters. [Hard coded for unit-tests].
    auto params = t::pipelines::slac::SLACOptimizerParams();
    params.slac_folder_ = slac_folder;
    params.voxel_size_ = 0.05;
    params.regularizer_weight_ = 1;
    params.distance_threshold_ = 0.07;
    params.fitness_threshold_ = 0.3;
    params.max_iterations_ = 5;
    params.device_ = device;

    // Debug options.
    auto debug_option = t::pipelines::slac::SLACDebugOption();
    debug_option.debug_ = false;

    t::pipelines::slac::ControlGrid control_grid;
    open3d::pipelines::registration::PoseGraph pose_graph_updated;

    // Testing SLAC.
    std::tie(pose_graph_updated, control_grid) =
            t::pipelines::slac::RunSLACOptimizerForFragments(
                    fragment_fnames, *pose_graph, params, debug_option);

    // TODO.
    // AssertSavedCorrespondences();

    // Write control grids.
    auto hashmap = control_grid.GetHashMap();
    core::Tensor active_buf_indices;
    hashmap->GetActiveIndices(active_buf_indices);
    active_buf_indices = active_buf_indices.To(core::Int64);

    hashmap->GetKeyTensor()
            .IndexGet({active_buf_indices})
            .Save(params.GetSubfolderName() + "/ctr_grid_keys.npy");
    // Check if same.
    core::Tensor output_control_grid_keys = core::Tensor::Load(
            params.GetSubfolderName() + "/ctr_grid_keys.npy");
    core::Tensor test_control_grid_keys = core::Tensor::Load(
            test_slac_folder + "/0.050" + "/ctr_grid_keys.npy");

    hashmap->GetValueTensor()
            .IndexGet({active_buf_indices})
            .Save(params.GetSubfolderName() + "/ctr_grid_values.npy");
    // Check if same.
    core::Tensor output_control_grid_values = core::Tensor::Load(
            params.GetSubfolderName() + "/ctr_grid_values.npy");
    core::Tensor test_control_grid_values = core::Tensor::Load(
            test_slac_folder + "/0.050" + "/ctr_grid_values.npy");

    // Write pose graph.
    io::WritePoseGraph(params.GetSubfolderName() + "/optimized_posegraph_" +
                               "slac" + ".json",
                       pose_graph_updated);

    // Skipping comparing PoseGraph. TODO.
    // auto output_pose_graph = io::CreatePoseGraphFromFile(
    //         params.GetSubfolderName() + "/optimized_posegraph_" + "slac" +
    //         ".json");
    // auto test_pose_graph = io::CreatePoseGraphFromFile(
    //         test_slac_folder + "/0.050" + "/optimized_posegraph_" + "slac" +
    //         ".json");

    // Write trajectory for SLACIntegrate
    open3d::camera::PinholeCameraTrajectory trajectory;
    for (size_t i = 0; i < pose_graph_updated.nodes_.size(); ++i) {
        auto fragment_pose_graph = io::CreatePoseGraphFromFile(fmt::format(
                "{}/fragment_optimized_{:03d}.json", fragment_folder, i));
        for (auto node : fragment_pose_graph->nodes_) {
            auto pose = pose_graph_updated.nodes_[i].pose_ * node.pose_;
            camera::PinholeCameraParameters param;
            param.extrinsic_ = pose.inverse().eval();
            trajectory.parameters_.push_back(param);
        }
    }
    // Skipping comparing CameraTrajectory. TODO.
    open3d::io::WritePinholeCameraTrajectory(params.GetSubfolderName() +
                                                     "/optimized_trajectory_" +
                                                     "slac" + ".log",
                                             trajectory);
}

TEST_P(SLACPermuteDevices, DISABLED_SLACIntegrate) {
    core::Device device = GetParam();

    std::string dataset_folder = utility::GetDataPathCommon(
            "reconstruction_system/livingroom1_clean_micro");
    std::string fragment_folder = dataset_folder + "/test_fragments";
    std::string color_folder = dataset_folder + "/image";
    std::string depth_folder = dataset_folder + "/depth";

    std::string scene_folder = dataset_folder + "/test_scene";
    std::string test_slac_subfolder = dataset_folder + "/test_slac/0.050";

    std::string test_pointcloud_filename =
            dataset_folder + "/test_output_pcd.ply";

    std::string output_slac_folder = dataset_folder + "/output_slac";

    // Optimized fragment pose graph.
    std::string posegraph_path =
            std::string(test_slac_subfolder + "/optimized_posegraph_slac.json");
    auto posegraph = io::CreatePoseGraphFromFile(posegraph_path);

    // Intrinsics. [None provided - default Primesense intrinsics will be used].
    std::string intrinsic_path = "";
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    // Voxelgrid options.
    int block_count = 40000;
    float voxel_size = 0.05;
    float depth_scale = 1000.f;
    float max_depth = 3.f;
    t::geometry::VoxelBlockGrid voxel_grid(
            {"tsdf", "weight", "color"},
            {core::Dtype::Float32, core::Dtype::Float32, core::Dtype::Float32},
            {{1}, {1}, {3}}, voxel_size, 16, block_count, device);

    // Load control grid
    core::Tensor ctr_grid_keys = core::Tensor::Load(test_slac_subfolder +
                                                    "/ctr_grid_keys."
                                                    "npy");
    core::Tensor ctr_grid_values = core::Tensor::Load(test_slac_subfolder +
                                                      "/ctr_grid_values."
                                                      "npy");
    t::pipelines::slac::ControlGrid ctr_grid(3.0 / 8, ctr_grid_keys.To(device),
                                             ctr_grid_values.To(device),
                                             device);

    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    int k = 0;
    for (size_t i = 0; i < posegraph->nodes_.size(); ++i) {
        utility::LogDebug("Fragment: {}", i);
        auto fragment_pose_graph = *io::CreatePoseGraphFromFile(fmt::format(
                "{}/fragment_optimized_{:03d}.json", fragment_folder, i));
        for (auto node : fragment_pose_graph.nodes_) {
            Eigen::Matrix4d pose_local = node.pose_;
            core::Tensor extrinsic_local_t =
                    core::eigen_converter::EigenMatrixToTensor(
                            pose_local.inverse().eval());

            Eigen::Matrix4d pose = posegraph->nodes_[i].pose_ * node.pose_;
            core::Tensor extrinsic_t =
                    core::eigen_converter::EigenMatrixToTensor(
                            pose.inverse().eval());

            auto depth =
                    t::io::CreateImageFromFile(depth_filenames[k])->To(device);
            auto color =
                    t::io::CreateImageFromFile(color_filenames[k])->To(device);
            t::geometry::RGBDImage rgbd(color, depth);

            utility::Timer timer;
            timer.Start();

            t::geometry::RGBDImage rgbd_projected =
                    ctr_grid.Deform(rgbd, intrinsic_t, extrinsic_local_t,
                                    depth_scale, max_depth);
            core::Tensor frustum_block_coords =
                    voxel_grid.GetUniqueBlockCoordinates(
                            rgbd.depth_, intrinsic_t, extrinsic_t, depth_scale,
                            max_depth);
            voxel_grid.Integrate(frustum_block_coords, rgbd_projected.depth_,
                                 rgbd_projected.color_, intrinsic_t,
                                 extrinsic_t, depth_scale, max_depth);
            timer.Stop();

            ++k;
            utility::LogDebug("{}: Deformation + Integration takes {}", k,
                              timer.GetDuration());
        }
    }

    auto pcd = voxel_grid.ExtractPointCloud();

    t::geometry::PointCloud test_pointcloud(device);
    t::io::ReadPointCloud(test_pointcloud_filename, test_pointcloud,
                          {"auto", false, false, true});

    test_pointcloud = test_pointcloud.To(device);

    IsPointCloudSimilar(pcd, test_pointcloud, voxel_size, 0.98, 0.00004);
}

}  // namespace tests
}  // namespace open3d
