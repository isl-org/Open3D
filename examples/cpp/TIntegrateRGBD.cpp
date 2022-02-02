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
#include "open3d/Open3D.h"

using namespace open3d;
using namespace open3d::core;

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TIntegrateRGBD [color_folder] [depth_folder] [trajectory] [options]");
    utility::LogInfo("      Given RGBD images, reconstruct mesh or point cloud from color and depth images");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --voxel_size [=0.0058 (m)]");
    utility::LogInfo("    --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("    --depth_scale [=1000.0]");
    utility::LogInfo("    --depth_max [=3.0]");
    utility::LogInfo("    --device [CPU:0]");
    utility::LogInfo("    --raycast");
    utility::LogInfo("    --mesh");
    utility::LogInfo("    --pointcloud");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    if (argc < 4 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    // Color and depth
    std::string color_folder = std::string(argv[1]);
    std::string depth_folder = std::string(argv[2]);

    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError(
                "[TIntegrateRGBD] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }

    // Trajectory
    std::string trajectory_path = std::string(argv[3]);
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

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
    Tensor intrinsic_t = Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    int block_count =
            utility::GetProgramOptionAsInt(argc, argv, "--block_count", 1000);

    float voxel_size = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float depth_max = static_cast<float>(
            utility::GetProgramOptionAsDouble(argc, argv, "--depth_max", 3.f));

    bool enable_raycast = utility::ProgramOptionExists(argc, argv, "--raycast");
    bool debug = utility::ProgramOptionExists(argc, argv, "--debug");

    // Device
    std::string device_code = "CPU:0";
    if (utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());
    t::geometry::VoxelBlockGrid voxel_grid(
            {"tsdf", "weight", "color"},
            {core::Dtype::Float32, core::Dtype::Float32, core::Dtype::Float32},
            {{1}, {1}, {3}}, voxel_size, 16, block_count, device);

    double time_total = 0;
    double time_int = 0;
    double time_raycasting = 0;
    for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
        utility::Timer timer;
        timer.Start();

        // Load image
        utility::Timer timer_io;
        timer_io.Start();
        t::geometry::Image depth =
                (*t::io::CreateImageFromFile(depth_filenames[i]));
        t::geometry::Image color =
                (*t::io::CreateImageFromFile(color_filenames[i]));
        timer_io.Stop();
        utility::LogInfo("IO takes {}", timer_io.GetDuration());

        timer_io.Start();
        depth = depth.To(device);
        color = color.To(device);
        timer_io.Stop();
        utility::LogInfo("Conversion takes {}", timer_io.GetDuration());

        Eigen::Matrix4d extrinsic = trajectory->parameters_[i].extrinsic_;
        Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic);

        utility::Timer int_timer;
        int_timer.Start();
        core::Tensor frustum_block_coords =
                voxel_grid.GetUniqueBlockCoordinates(depth, intrinsic_t,
                                                     extrinsic_t, depth_scale,
                                                     depth_max);
        voxel_grid.Integrate(frustum_block_coords, depth, color, intrinsic_t,
                             extrinsic_t);
        int_timer.Stop();
        utility::LogInfo("{}: Integration takes {}", i,
                         int_timer.GetDuration());
        time_int += int_timer.GetDuration();

        if (enable_raycast) {
            utility::Timer ray_timer;

            ray_timer.Start();
            auto result = voxel_grid.RayCast(
                    frustum_block_coords, intrinsic_t, extrinsic_t,
                    depth.GetCols(), depth.GetRows(), {"depth", "color"},
                    depth_scale, 0.1, depth_max, std::min(i * 1.0f, 3.0f));
            ray_timer.Stop();

            utility::LogInfo("{}: Raycast takes {}", i,
                             ray_timer.GetDuration());
            time_raycasting += ray_timer.GetDuration();

            if (debug) {
                t::geometry::Image depth_raycast(result["depth"]);
                visualization::DrawGeometries(
                        {std::make_shared<open3d::geometry::Image>(
                                depth_raycast
                                        .ColorizeDepth(depth_scale, 0.1,
                                                       depth_max)
                                        .ToLegacy())});
                t::geometry::Image color_raycast(result["color"]);
                visualization::DrawGeometries(
                        {std::make_shared<open3d::geometry::Image>(
                                color_raycast.ToLegacy())});
            }
        }

        timer.Stop();
        utility::LogInfo("{}: Per iteration takes {}", i, timer.GetDuration());
        time_total += timer.GetDuration();
    }

    size_t n = trajectory->parameters_.size();
    utility::LogInfo("per frame: {}, ray casting: {}, integration: {}",
                     time_total / n, time_raycasting / n, time_int / n);

    if (utility::ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh = voxel_grid.ExtractTriangleMesh(3.0f);
        auto mesh_legacy =
                std::make_shared<geometry::TriangleMesh>(mesh.ToLegacy());
        open3d::io::WriteTriangleMesh("mesh_" + device.ToString() + ".ply",
                                      *mesh_legacy);
    }

    if (utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd = voxel_grid.ExtractPointCloud(3.0f);
        auto pcd_legacy =
                std::make_shared<open3d::geometry::PointCloud>(pcd.ToLegacy());
        open3d::io::WritePointCloud("pcd_" + device.ToString() + ".ply",
                                    *pcd_legacy);
    }

    if (utility::ProgramOptionExists(argc, argv, "--tsdf")) {
        voxel_grid.Save("tsdf.npz");
    }

    return 0;
}
