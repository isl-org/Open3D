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

/// \file VoxelHashingRefact.cpp
/// In the refactored system, after input, all the data are wrapped in the Frame
/// container to be transported between modules, e.g. from RayCast to VO.
/// Each step should be responsible for the data consistency.

/// E.g. depth:
/// Raw input: uint16, scaled (x1000)
/// Integration input: uint16 or float, scaled (x1000)
/// RayCast output: float, scaled (x1000 (or not?))
/// VO input: float, unscaled and filled with nan
#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    VoxelHashing [color_folder] [depth_folder] [options]");
    utility::LogInfo("     Given RGBD images, reconstruct mesh or point cloud from color and depth images");
    utility::LogInfo("     [options]");
    utility::LogInfo("     --voxel_size [=0.0058 (m)]");
    utility::LogInfo("     --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("     --depth_scale [=1000.0]");
    utility::LogInfo("     --max_depth [=3.0]");
    utility::LogInfo("     --sdf_trunc [=0.04]");
    utility::LogInfo("     --device [CPU:0]");
    utility::LogInfo("     --mesh");
    utility::LogInfo("     --pointcloud");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char** argv) {
    using MaskCode = t::geometry::TSDFVoxelGrid::RayCastMaskCode;

    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 4) {
        PrintHelp();
        return 1;
    }

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    // Device
    std::string device_code = "CPU:0";
    if (utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    // Input RGBD files
    std::string color_folder = std::string(argv[1]);
    std::string depth_folder = std::string(argv[2]);

    std::vector<std::string> color_filenames, depth_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError(
                "[VoxelHashing] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }
    std::sort(color_filenames.begin(), color_filenames.end());
    std::sort(depth_filenames.begin(), depth_filenames.end());
    size_t n = color_filenames.size();
    size_t iterations = static_cast<size_t>(
            utility::GetProgramOptionAsInt(argc, argv, "--iterations", n));
    iterations = std::min(n, iterations);

    // GT trajectory for reference
    std::string gt_trajectory_path = std::string(argv[3]);
    auto gt_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(gt_trajectory_path);
    Eigen::Matrix4d src_pose_gt_eigen =
            gt_trajectory->parameters_[0].extrinsic_.inverse().eval();
    Tensor src_pose_gt =
            core::eigen_converter::EigenMatrixToTensor(src_pose_gt_eigen);
    Tensor T_frame_to_model = src_pose_gt;

    // Intrinsics
    std::string intrinsic_path = utility::GetProgramOptionAsString(
            argc, argv, "--intrinsic_path", "");
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    if (intrinsic_path.empty()) {
        utility::LogWarning("Using default Primesense intrinsics");
    } else if (!io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogError("Unable to convert json to intrinsics.");
    }
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, Dtype::Float32, device);

    // VoxelBlock configurations
    float voxel_size = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float sdf_trunc = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc", 0.04f));
    int block_resolution = utility::GetProgramOptionAsInt(
            argc, argv, "--block_resolution", 16);
    int block_count =
            utility::GetProgramOptionAsInt(argc, argv, "--block_count", 10000);

    // Odometry configurations
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float depth_max = static_cast<float>(
            utility::GetProgramOptionAsDouble(argc, argv, "--depth_max", 3.f));

    // Initialize model
    t::pipelines::voxelhashing::Model model(voxel_size, sdf_trunc,
                                            block_resolution, block_count,
                                            T_frame_to_model, device);

    // Initialize frame
    t::pipelines::voxelhashing::Frame input_frame(intrinsic_t, device);
    t::pipelines::voxelhashing::Frame raycast_frame(intrinsic_t, device);

    core::Tensor identity =
            core::Tensor::Eye(4, core::Dtype::Float32, core::Device("CPU:0"));

    // Iterate over frames
    for (size_t i = 0; i < iterations; ++i) {
        // Load image into frame
        t::geometry::Image input_depth =
                *t::io::CreateImageFromFile(depth_filenames[i]);
        t::geometry::Image input_color =
                *t::io::CreateImageFromFile(color_filenames[i]);
        input_frame.SetDataFromImage("depth", input_depth);
        input_frame.SetDataFromImage("color", input_color);

        if (i > 0) {
            utility::LogInfo("Frame-to-model for the frame {}", i);

            auto result = model.voxel_grid_.RayCast(
                    raycast_frame.GetIntrinsics(), T_frame_to_model.Inverse(),
                    input_depth.GetCols(), input_depth.GetRows(), 80, 0.1, 4.0,
                    std::min(i * 1.0f, 3.0f), MaskCode::DepthMap);
            raycast_frame.SetData("depth", result[MaskCode::DepthMap]);

            Tensor delta_frame_to_model =
                    t::pipelines::odometry::RGBDOdometryMultiScale(
                            t::geometry::RGBDImage(
                                    input_frame.GetDataAsImage("color"),
                                    input_frame.GetDataAsImage("depth")),
                            t::geometry::RGBDImage(
                                    t::geometry::Image(),
                                    raycast_frame.GetDataAsImage("depth")),
                            raycast_frame.GetIntrinsics(), identity,
                            depth_scale, depth_max, 0.07, {10, 0, 0},
                            t::pipelines::odometry::Method::PointToPlane);
            T_frame_to_model = T_frame_to_model.Matmul(delta_frame_to_model);
        }

        model.voxel_grid_.Integrate(input_frame.GetDataAsImage("depth"),
                                    input_frame.GetDataAsImage("color"),
                                    intrinsic_t, T_frame_to_model.Inverse(),
                                    depth_scale, depth_max);
    }

    if (utility::ProgramOptionExists(argc, argv, "--mesh")) {
        std::string filename = utility::GetProgramOptionAsString(
                argc, argv, "--mesh", "mesh_" + device.ToString() + ".ply");
        auto mesh = model.voxel_grid_.ExtractSurfaceMesh();
        auto mesh_legacy = std::make_shared<geometry::TriangleMesh>(
                mesh.ToLegacyTriangleMesh());
        open3d::io::WriteTriangleMesh(filename, *mesh_legacy);
    }

    if (utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        std::string filename = utility::GetProgramOptionAsString(
                argc, argv, "--pointcloud",
                "pcd_" + device.ToString() + ".ply");
        auto pcd = model.voxel_grid_.ExtractSurfacePoints();
        auto pcd_legacy = std::make_shared<open3d::geometry::PointCloud>(
                pcd.ToLegacyPointCloud());
        open3d::io::WritePointCloud(filename, *pcd_legacy);
    }
}
