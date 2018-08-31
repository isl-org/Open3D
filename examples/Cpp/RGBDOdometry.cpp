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

#include <IO/IO.h>
#include <Core/Core.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Utility/Helper.h>
#include <Visualization/Visualization.h>

using namespace open3d;

void PrintHelp()
{
    PrintOpen3DVersion();
    PrintInfo("Usage :\n");
    PrintInfo("    > TestRGBDOdometry [color1] [depth1] [color2] [depth2]\n");
}

std::shared_ptr<RGBDImage> ReadRGBDImage(
        const char* color_filename, const char* depth_filename,
        const PinholeCameraIntrinsic &intrinsic,
        bool visualize)
{
    Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);
    PrintDebug("Reading RGBD image : \n");
    PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
            color.width_, color.height_,
            color.num_of_channels_, color.bytes_per_channel_ * 8);
    PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
            depth.width_, depth.height_,
            depth.num_of_channels_, depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 3.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<RGBDImage> rgbd_image =
            CreateRGBDImageFromColorAndDepth(color, depth,
            depth_scale, depth_trunc, convert_rgb_to_intensity);
    if (visualize){
        auto pcd = CreatePointCloudFromRGBDImage(*rgbd_image, intrinsic);
        DrawGeometries({pcd});
    }
    return rgbd_image;
}

int main(int argc, char *argv[])
{
    if (argc == 1 || ProgramOptionExists(argc, argv, "--help") || argc != 5) {
        PrintHelp();
        return 1;
    }
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    PinholeCameraIntrinsic intrinsic = PinholeCameraIntrinsic(
            PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    bool visualize = true;
    auto source = ReadRGBDImage(argv[1], argv[2], intrinsic, visualize);
    auto target = ReadRGBDImage(argv[3], argv[4], intrinsic, visualize);

    Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
    std::tuple<bool, Eigen::Matrix4d, Eigen::Matrix6d> rgbd_odo =
            ComputeRGBDOdometry(*source, *target, intrinsic, odo_init,
            RGBDOdometryJacobianFromHybridTerm(),
            OdometryOption());
    std::cout << "RGBD Odometry" << std::endl;
    std::cout << std::get<1>(rgbd_odo) << std::endl;
}
