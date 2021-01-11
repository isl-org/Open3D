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

#include "open3d/t/geometry/TSDFVoxelGrid.h"

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/Registration.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TSDFVoxelGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TSDFVoxelGrid,
                         TSDFVoxelGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TSDFVoxelGridPermuteDevices, Integrate) {
    core::Device device = GetParam();

    float voxel_size = 0.008;
    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Dtype::Float32},
                                           {"weight", core::Dtype::UInt16},
                                           {"color", core::Dtype::UInt16}},
                                          voxel_size, 0.04f, 16, 1000, device);

    // Intrinsics
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, core::Dtype::Float32);

    // Extrinsics
    std::string trajectory_path =
            std::string(TEST_DATA_DIR) + "/RGBD/odometry.log";
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
        // Load image
        std::shared_ptr<geometry::Image> depth_legacy = io::CreateImageFromFile(
                fmt::format("{}/RGBD/depth/{:05d}.png",
                            std::string(TEST_DATA_DIR), i));

        std::shared_ptr<geometry::Image> color_legacy = io::CreateImageFromFile(
                fmt::format("{}/RGBD/color/{:05d}.jpg",
                            std::string(TEST_DATA_DIR), i));

        t::geometry::Image depth =
                t::geometry::Image::FromLegacyImage(*depth_legacy, device);
        t::geometry::Image color =
                t::geometry::Image::FromLegacyImage(*color_legacy, device);

        Eigen::Matrix4f extrinsic =
                trajectory->parameters_[i].extrinsic_.cast<float>();
        core::Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic).To(
                        device);

        voxel_grid.Integrate(depth, color, intrinsic_t, extrinsic_t);
    }

    auto pcd = voxel_grid.ExtractSurfacePoints().ToLegacyPointCloud();
    auto pcd_gt = *io::CreatePointCloudFromFile(std::string(TEST_DATA_DIR) +
                                                "/RGBD/example_tsdf_pcd.ply");
    auto result = pipelines::registration::EvaluateRegistration(pcd, pcd_gt,
                                                                voxel_size);

    EXPECT_EQ(pcd.points_.size(), pcd_gt.points_.size());

    // Allow some numerical noise
    EXPECT_NEAR(result.fitness_, 1.0, 1e-5);
    EXPECT_NEAR(result.inlier_rmse_, 0, 1e-5);
}
}  // namespace tests
}  // namespace open3d
