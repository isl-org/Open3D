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

#include "Open3D/Open3D.h"

using namespace open3d;

void PrintHelp() {
    PrintOpen3DVersion();
    utility::LogInfo("Usage :");
    utility::LogInfo("    > RGBDOdometry [color1] [depth1] [color2] [depth2]");
}

std::shared_ptr<geometry::RGBDImage> ReadRGBDImage(
        const char* color_filename,
        const char* depth_filename,
        const camera::PinholeCameraIntrinsic& intrinsic,
        bool visualize) {
    geometry::Image color, depth;
    io::ReadImage(color_filename, color);
    io::ReadImage(depth_filename, depth);
    utility::LogInfo("Reading RGBD image : ");
    utility::LogInfo("     Color : {:d} x {:d} x {:d} ({:d} bits per channel)",
                     color.width_, color.height_, color.num_of_channels_,
                     color.bytes_per_channel_ * 8);
    utility::LogInfo("     Depth : {:d} x {:d} x {:d} ({:d} bits per channel)",
                     depth.width_, depth.height_, depth.num_of_channels_,
                     depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 3.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<geometry::RGBDImage> rgbd_image =
            geometry::RGBDImage::CreateFromColorAndDepth(
                    color, depth, depth_scale, depth_trunc,
                    convert_rgb_to_intensity);
    if (visualize) {
        auto pcd = geometry::PointCloud::CreateFromRGBDImage(*rgbd_image,
                                                             intrinsic);
        visualization::DrawGeometries({pcd});
    }
    return rgbd_image;
}

int main(int argc, char* argv[]) {
    if (argc == 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        argc != 5) {
        PrintHelp();
        return 1;
    }
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    bool visualize = true;
    auto source = ReadRGBDImage(argv[1], argv[2], intrinsic, visualize);
    auto target = ReadRGBDImage(argv[3], argv[4], intrinsic, visualize);

    Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
            odometry::ComputeRGBDOdometry(
                    *source, *target, intrinsic, odo_init,
                    odometry::RGBDOdometryJacobianFromHybridTerm(),
                    odometry::OdometryOption());
    std::cout << "RGBD Odometry" << std::endl;
    std::cout << std::get<1>(rgbd_odo) << std::endl;
    return 0;
}
