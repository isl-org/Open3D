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
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 4) {
        PrintHelp();
        return 1;
    }

    using MaskCode = t::geometry::TSDFVoxelGrid::RayCastMaskCode;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // Color and depth
    std::string color_folder = std::string(argv[1]);
    std::string depth_folder = std::string(argv[2]);

    std::vector<std::string> color_filenames;
    utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    // Trajectory
    std::string gt_trajectory_path = std::string(argv[3]);
    auto gt_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(gt_trajectory_path);

    if (color_filenames.size() != depth_filenames.size()) {
        utility::LogError(
                "[VoxelHashing] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }

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

    // Device
    std::string device_code = "CPU:0";
    if (utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    utility::LogInfo("Using device: {}", device.ToString());

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, Dtype::Float32, device);

    int block_count =
            utility::GetProgramOptionAsInt(argc, argv, "--block_count", 1000);

    float voxel_size = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float max_depth = static_cast<float>(
            utility::GetProgramOptionAsDouble(argc, argv, "--max_depth", 3.f));
    float sdf_trunc = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc", 0.04f));

    t::pipelines::voxelhashing::Model model(
            voxel_size, sdf_trunc, block_count,
            core::Tensor::Eye(4, core::Dtype::Float32, core::Device("CPU:0")),
            device);

    size_t n = color_filenames.size();
    size_t iterations = static_cast<size_t>(
            utility::GetProgramOptionAsInt(argc, argv, "--iterations", n));
    iterations = std::min(n, iterations);

    Eigen::Matrix4d src_pose_gt_eigen =
            gt_trajectory->parameters_[0].extrinsic_.inverse().eval();
    Tensor src_pose_gt =
            core::eigen_converter::EigenMatrixToTensor(src_pose_gt_eigen);
    Tensor T_curr_to_model = src_pose_gt;

    core::Tensor diffs =
            core::Tensor::Empty({int64_t(iterations), 2}, core::Dtype::Float64,
                                core::Device("CPU:0"));

    size_t debug_idx = static_cast<size_t>(
            utility::GetProgramOptionAsInt(argc, argv, "--debug_idx", n));

    for (size_t i = 0; i < iterations; ++i) {
        // Load image
        t::geometry::Image src_depth =
                *t::io::CreateImageFromFile(depth_filenames[i]);
        t::geometry::Image src_color =
                *t::io::CreateImageFromFile(color_filenames[i]);

        Eigen::Matrix4d curr_pose_gt_eigen =
                gt_trajectory->parameters_[i].extrinsic_.inverse().eval();
        Tensor curr_pose_gt =
                core::eigen_converter::EigenMatrixToTensor(curr_pose_gt_eigen);

        if (i > 0) {
            t::geometry::RGBDImage src, dst;
            src.depth_ =
                    src_depth.To(device).To(core::Dtype::Float32, false, 1.0);
            src.color_ = src_color.To(device);

            utility::LogInfo("Frame-to-model for the frame {}", i);
            auto result = model.voxel_grid_.RayCast(
                    intrinsic_t, T_curr_to_model.Inverse(), src_depth.GetCols(),
                    src_depth.GetRows(), 100, 0.1, 3.0,
                    std::min(i * 1.0f, 3.0f),
                    MaskCode::DepthMap | MaskCode::ColorMap);
            dst.depth_ = t::geometry::Image(result[MaskCode::DepthMap]);
            dst.color_ = t::geometry::Image(result[MaskCode::ColorMap]);
            // visualization::DrawGeometries({std::make_shared<geometry::Image>(
            //         dst.depth_.ToLegacyImage())});

            // Debug: before odometry
            core::Tensor trans = core::Tensor::Eye(4, core::Dtype::Float64,
                                                   core::Device("CPU:0"));
            if (i > debug_idx) {
                auto source_pcd =
                        std::make_shared<open3d::geometry::PointCloud>(
                                t::geometry::PointCloud::CreateFromDepthImage(
                                        src.depth_, intrinsic_t, trans,
                                        depth_scale)
                                        .ToLegacyPointCloud());
                source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
                auto target_pcd =
                        std::make_shared<open3d::geometry::PointCloud>(
                                t::geometry::PointCloud::CreateFromDepthImage(
                                        dst.depth_, intrinsic_t, trans,
                                        depth_scale)
                                        .ToLegacyPointCloud());
                target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
                visualization::DrawGeometries({source_pcd, target_pcd});
            }

            // Odometry
            Tensor delta_curr_to_model =
                    t::pipelines::odometry::RGBDOdometryMultiScale(
                            src, dst, intrinsic_t, trans, depth_scale, 3.0,
                            0.07, {10, 0, 0},
                            t::pipelines::odometry::Method::PointToPlane);
            T_curr_to_model = T_curr_to_model.Matmul(delta_curr_to_model);

            // Debug: after odometry
            if (i > debug_idx) {
                auto source_pcd =
                        std::make_shared<open3d::geometry::PointCloud>(
                                t::geometry::PointCloud::CreateFromDepthImage(
                                        src.depth_, intrinsic_t,
                                        delta_curr_to_model.Inverse(),
                                        depth_scale)
                                        .ToLegacyPointCloud());
                source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
                auto target_pcd =
                        std::make_shared<open3d::geometry::PointCloud>(
                                t::geometry::PointCloud::CreateFromDepthImage(
                                        dst.depth_, intrinsic_t,
                                        core::Tensor::Eye(4,
                                                          core::Dtype::Float32,
                                                          device),
                                        depth_scale)
                                        .ToLegacyPointCloud());
                target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
                visualization::DrawGeometries({source_pcd, target_pcd});
            }
        }

        Tensor diff = curr_pose_gt.Inverse().Matmul(T_curr_to_model);
        double rot_err = std::acos(0.5 * (diff[0][0].Item<double>() +
                                          diff[1][1].Item<double>() +
                                          diff[2][2].Item<double>() - 1));
        double trans_err = std::sqrt(
                diff[0][3].Item<double>() * diff[0][3].Item<double>() +
                diff[1][3].Item<double>() * diff[1][3].Item<double>() +
                diff[2][3].Item<double>() * diff[2][3].Item<double>());
        diffs[i][0] = rot_err;
        diffs[i][1] = trans_err;
        utility::LogInfo("T_diff = {}", diff.ToString());
        utility::LogInfo("rot_err = {}, trans_err = {}", rot_err, trans_err);

        model.voxel_grid_.Integrate(src_depth.To(device), src_color.To(device),
                                    intrinsic_t, T_curr_to_model.Inverse(),
                                    depth_scale, max_depth);
    }

    std::string diffs_name =
            utility::GetProgramOptionAsString(argc, argv, "--output", "vh.npy");
    diffs.Save(diffs_name);

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
