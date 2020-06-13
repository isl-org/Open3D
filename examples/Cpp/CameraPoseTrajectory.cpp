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

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "Open3D/Open3D.h"

int main(int argc, char *argv[]) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3) {
        utility::LogInfo("Usage :");
        utility::LogInfo(">    CameraPoseTrajectory trajectory_file pcds_dir");
        return 1;
    }
    const int NUM_OF_COLOR_PALETTE = 5;
    Eigen::Vector3d color_palette[NUM_OF_COLOR_PALETTE] = {
            Eigen::Vector3d(255, 180, 0) / 255.0,
            Eigen::Vector3d(0, 166, 237) / 255.0,
            Eigen::Vector3d(246, 81, 29) / 255.0,
            Eigen::Vector3d(127, 184, 0) / 255.0,
            Eigen::Vector3d(13, 44, 84) / 255.0,
    };

    camera::PinholeCameraTrajectory trajectory;
    io::ReadPinholeCameraTrajectory(argv[1], trajectory);
    std::vector<std::shared_ptr<const geometry::Geometry>> pcds;
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        std::string buffer =
                fmt::format("{}cloud_bin_{:d}.pcd", argv[2], (int)i);
        if (utility::filesystem::FileExists(buffer.c_str())) {
            auto pcd = io::CreatePointCloudFromFile(buffer.c_str());
            pcd->Transform(trajectory.parameters_[i].extrinsic_);
            pcd->colors_.clear();
            if ((int)i < NUM_OF_COLOR_PALETTE) {
                pcd->colors_.resize(pcd->points_.size(), color_palette[i]);
            } else {
                pcd->colors_.resize(pcd->points_.size(),
                                    (Eigen::Vector3d::Random() +
                                     Eigen::Vector3d::Constant(1.0)) *
                                            0.5);
            }
            pcds.push_back(pcd);
        }
    }
    visualization::DrawGeometriesWithCustomAnimation(pcds);

    return 0;
}
