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

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    TOdometryRGBD [src_depth] [dst_depth]");
    utility::LogInfo("     Given two depth images, perform rgbd odometry");
    utility::LogInfo("     [options]");
    utility::LogInfo("     --intrinsic_path [camera_intrinsic]");
    utility::LogInfo("     --depth_scale [=1000.0]");
    utility::LogInfo("     --depth_diff [=0.07]");
    utility::LogInfo("     --device [CPU:0]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char **argv) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 3) {
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
    core::Tensor intrinsic_t = core::Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, core::Dtype::Float32);

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

    t::geometry::RGBDImage src, dst;
    src.color_ = src_color.To(device);
    dst.color_ = dst_color.To(device);
    src.depth_ = src_depth.To(device);
    dst.depth_ = dst_depth.To(device);

    core::Tensor trans = core::Tensor::Eye(4, core::Dtype::Float64, device);

    // Visualize before odometry
    auto source_pcd = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud::CreateFromDepthImage(
                    src.depth_, intrinsic_t, trans, depth_scale)
                    .ToLegacyPointCloud());
    source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
    auto target_pcd = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud::CreateFromDepthImage(
                    dst.depth_, intrinsic_t, trans, depth_scale)
                    .ToLegacyPointCloud());
    target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
    visualization::DrawGeometries({source_pcd, target_pcd});

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

    auto result = t::pipelines::odometry::RGBDOdometryMultiScale(
            src, dst, intrinsic_t, trans, depth_scale, 3.0, {10, 5, 3},
            odom_method,
            t::pipelines::odometry::OdometryLossParams(depth_diff));

    // Visualize after odometry
    source_pcd = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud::CreateFromRGBDImage(
                    t::geometry::RGBDImage(src.color_, src.depth_), intrinsic_t,
                    result.transformation_.Inverse(), depth_scale)
                    .ToLegacyPointCloud());
    // source_pcd->PaintUniformColor(Eigen::Vector3d(1, 0, 0));
    target_pcd = std::make_shared<open3d::geometry::PointCloud>(
            t::geometry::PointCloud::CreateFromRGBDImage(
                    t::geometry::RGBDImage(dst.color_, dst.depth_), intrinsic_t,
                    core::Tensor::Eye(4, core::Dtype::Float32, device),
                    depth_scale)
                    .ToLegacyPointCloud());
    // target_pcd->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
    visualization::DrawGeometries({source_pcd, target_pcd});
}
