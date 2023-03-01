// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TOdometryRGBD [src_depth] [dst_depth] [src_color] [dst_color]");
    utility::LogInfo("      Given two RGBD images, perform rgbd odometry and visualize results.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("    --depth_scale [=1000.0]");
    utility::LogInfo("    --depth_diff [=0.07]");
    utility::LogInfo("    --method [=PointToPlane | Intensity | Hybrid]");
    utility::LogInfo("    --device [CPU:0]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;
    using core::Tensor;
    using t::geometry::PointCloud;
    using t::geometry::RGBDImage;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 4 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string device_string =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CPU:0");
    core::Device device(device_string);

    // src and dst depth images
    std::string src_depth_path = std::string(argv[1]);
    std::string dst_depth_path = std::string(argv[2]);
    std::string src_color_path = std::string(argv[3]);
    std::string dst_color_path = std::string(argv[4]);

    // intrinsics and Tensor conversion
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

    // Parameters
    float depth_scale = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float depth_diff = static_cast<float>(utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_diff", 0.07f));
    std::string method = utility::GetProgramOptionAsString(
            argc, argv, "--method", "PointToPlane");

    // Read input
    auto src_depth = *t::io::CreateImageFromFile(src_depth_path);
    auto dst_depth = *t::io::CreateImageFromFile(dst_depth_path);
    auto src_color = *t::io::CreateImageFromFile(src_color_path);
    auto dst_color = *t::io::CreateImageFromFile(dst_color_path);

    RGBDImage src, dst;
    src.color_ = src_color.To(device);
    dst.color_ = dst_color.To(device);
    src.depth_ = src_depth.To(device);
    dst.depth_ = dst_depth.To(device);

    Tensor trans = Tensor::Eye(4, core::Dtype::Float64, device);

    // Visualize before odometry
    auto source_pcd = std::make_shared<open3d::geometry::PointCloud>(
            PointCloud::CreateFromDepthImage(src.depth_, intrinsic_t, trans,
                                             depth_scale)
                    .ToLegacy());
    source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
    auto target_pcd = std::make_shared<open3d::geometry::PointCloud>(
            PointCloud::CreateFromDepthImage(dst.depth_, intrinsic_t, trans,
                                             depth_scale)
                    .ToLegacy());
    target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
    visualization::DrawGeometries({source_pcd, target_pcd});

    // Decide method
    t::pipelines::odometry::Method odom_method;
    if (method == "PointToPlane") {
        odom_method = t::pipelines::odometry::Method::PointToPlane;
    } else if (method == "Hybrid") {
        odom_method = t::pipelines::odometry::Method::Hybrid;
    } else if (method == "Intensity") {
        odom_method = t::pipelines::odometry::Method::Intensity;
    } else {
        utility::LogError("Unsupported method {}", method);
    }

    // Apply odometry
    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_scale, 3.0,
            std::vector<t::pipelines::odometry::OdometryConvergenceCriteria>{
                    20, 10, 5},
            odom_method,
            t::pipelines::odometry::OdometryLossParams(depth_diff));

    // Visualize after odometry
    source_pcd = std::make_shared<open3d::geometry::PointCloud>(
            PointCloud::CreateFromRGBDImage(
                    RGBDImage(src.color_, src.depth_), intrinsic_t,
                    result.transformation_.Inverse(), depth_scale)
                    .ToLegacy());
    source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
    target_pcd = std::make_shared<open3d::geometry::PointCloud>(
            PointCloud::CreateFromRGBDImage(
                    RGBDImage(dst.color_, dst.depth_), intrinsic_t,
                    Tensor::Eye(4, core::Dtype::Float32, device), depth_scale)
                    .ToLegacy());
    target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
    visualization::DrawGeometries({source_pcd, target_pcd});
}
