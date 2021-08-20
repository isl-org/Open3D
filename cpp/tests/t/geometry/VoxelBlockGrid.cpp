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

#include "open3d/t/geometry/VoxelBlockGrid.h"

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

using namespace t::geometry;

class VoxelBlockGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(VoxelBlockGrid,
                         VoxelBlockGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(VoxelBlockGridPermuteDevices, Construct) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    for (auto backend : backends) {
        auto vbg = VoxelBlockGrid({"tsdf", "weight", "color"},
                                  {core::Float32, core::UInt16, core::UInt8},
                                  {{1}, {1}, {3}}, 3.0 / 512, 8, 10, device,
                                  backend);

        auto hashmap = vbg.GetHashMap();

        auto tsdf_tensor = hashmap.GetValueTensor(0);
        auto weight_tensor = hashmap.GetValueTensor(1);
        auto color_tensor = hashmap.GetValueTensor(2);

        utility::LogInfo("TSDF block shape = {}, dtype = {}",
                         tsdf_tensor.GetShape(),
                         tsdf_tensor.GetDtype().ToString());
        utility::LogInfo("Weight block shape = {}, dtype = {}",
                         weight_tensor.GetShape(),
                         weight_tensor.GetDtype().ToString());
        utility::LogInfo("Color block shape = {}, dtype = {}",
                         color_tensor.GetShape(),
                         color_tensor.GetDtype().ToString());
    }
}
TEST_P(VoxelBlockGridPermuteDevices, Integrate) {
    core::Device device = GetParam();

    std::vector<core::HashBackendType> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashBackendType::Slab);
        backends.push_back(core::HashBackendType::StdGPU);
    } else {
        backends.push_back(core::HashBackendType::TBB);
    }

    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    // Extrinsics
    std::string trajectory_path =
            std::string(TEST_DATA_DIR) + "/RGBD/odometry.log";
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    for (auto backend : backends) {
        auto vbg = VoxelBlockGrid({"tsdf", "weight", "color"},
                                  {core::Float32, core::Float32, core::UInt16},
                                  {{1}, {1}, {3}}, 3.0 / 512, 16, 10000, device,
                                  backend);

        for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
            // Load image
            t::geometry::Image depth =
                    t::io::CreateImageFromFile(
                            fmt::format("{}/RGBD/depth/{:05d}.png",
                                        std::string(TEST_DATA_DIR), i))
                            ->To(device);
            t::geometry::Image color =
                    t::io::CreateImageFromFile(
                            fmt::format("{}/RGBD/color/{:05d}.jpg",
                                        std::string(TEST_DATA_DIR), i))
                            ->To(device);

            Eigen::Matrix4d extrinsic = trajectory->parameters_[i].extrinsic_;
            core::Tensor extrinsic_t =
                    core::eigen_converter::EigenMatrixToTensor(extrinsic);

            vbg.Integrate(depth, color, intrinsic_t, extrinsic_t);
        }

        if (backend == core::HashBackendType::StdGPU) {
            vbg.GetHashMap().Save("vbg.npz");
        }

        auto pcd = std::make_shared<open3d::geometry::PointCloud>(
                vbg.ExtractSurfacePoints().ToLegacy());
        visualization::DrawGeometries({pcd});
    }
}
}  // namespace tests
}  // namespace open3d
