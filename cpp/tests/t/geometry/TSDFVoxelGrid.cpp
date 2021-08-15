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

#include "open3d/t/geometry/TSDFVoxelGrid.h"

#include "core/CoreTest.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/TSDFVoxelGridIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TSDFVoxelGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TSDFVoxelGrid,
                         TSDFVoxelGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

t::geometry::TSDFVoxelGrid IntegrateTestScene(
        const core::Device &device,
        const core::HashmapBackend &backend = core::HashmapBackend::Default) {
    float voxel_size = 0.008;
    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Float32},
                                           {"weight", core::UInt16},
                                           {"color", core::UInt16}},
                                          voxel_size, 0.04f, 16, 1000, device,
                                          backend);

    // Intrinsics
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    // Extrinsics
    std::string trajectory_path = GetDataPathCommon("RGBD/odometry.log");
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
        // Load image
        t::geometry::Image depth =
                t::io::CreateImageFromFile(GetDataPathCommon(fmt::format(
                                                   "RGBD/depth/{:05d}.png", i)))
                        ->To(device);
        t::geometry::Image color =
                t::io::CreateImageFromFile(GetDataPathCommon(fmt::format(
                                                   "RGBD/color/{:05d}.jpg", i)))
                        ->To(device);

        Eigen::Matrix4d extrinsic = trajectory->parameters_[i].extrinsic_;
        core::Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic);

        voxel_grid.Integrate(depth, color, intrinsic_t, extrinsic_t);
    }

    return voxel_grid;
}

TEST_P(TSDFVoxelGridPermuteDevices, Integrate) {
    core::Device device = GetParam();
    const float dist_threshold = 0.004;

    std::vector<core::HashmapBackend> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashmapBackend::Slab);
        backends.push_back(core::HashmapBackend::StdGPU);
    } else {
        backends.push_back(core::HashmapBackend::TBB);
    }

    for (auto backend : backends) {
        auto voxel_grid = IntegrateTestScene(device, backend);

        auto pcd = voxel_grid.ExtractSurfacePoints().ToLegacy();
        auto pcd_gt = *io::CreatePointCloudFromFile(
                GetDataPathCommon("RGBD/example_tsdf_pcd.ply"));
        auto result = pipelines::registration::EvaluateRegistration(
                pcd, pcd_gt, dist_threshold);

        EXPECT_EQ(pcd.points_.size(), pcd_gt.points_.size());

        // Allow some numerical noise
        EXPECT_NEAR(result.fitness_, 1.0, 1e-5);
        EXPECT_NEAR(result.inlier_rmse_, 0, 1e-5);
    }
}

TEST_P(TSDFVoxelGridPermuteDevices, IO) {
    core::Device device = GetParam();
    std::string file_name = "test_tsdf.json";
    std::string npz_file_name = "test_tsdf.npz";
    const float dist_threshold = 0.004;

    auto voxel_grid = IntegrateTestScene(device);
    auto pcd = voxel_grid.ExtractSurfacePoints().ToLegacy();

    t::io::WriteTSDFVoxelGrid(file_name, voxel_grid);
    EXPECT_TRUE(utility::filesystem::FileExists(file_name));
    EXPECT_TRUE(utility::filesystem::FileExists(npz_file_name));

    auto loaded_voxel_grid = *t::io::CreateTSDFVoxelGridFromFile(file_name);
    auto pcd_from_loaded_voxel_grid =
            loaded_voxel_grid.ExtractSurfacePoints().ToLegacy();

    auto result = pipelines::registration::EvaluateRegistration(
            pcd, pcd_from_loaded_voxel_grid, dist_threshold);

    // Allow some numerical noise
    EXPECT_NEAR(result.fitness_, 1.0, 1e-5);
    EXPECT_NEAR(result.inlier_rmse_, 0, 1e-5);

    utility::filesystem::RemoveFile(file_name);
    utility::filesystem::RemoveFile(npz_file_name);
}

TEST_P(TSDFVoxelGridPermuteDevices, DISABLED_Raycast) {
    core::Device device = GetParam();
    const float depth_scale = 1000.0f;
    const float depth_max = 3.0f;

    const int cols = 640;
    const int rows = 480;

    // Intrinsic
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    core::Tensor intrinsic_t = core::Tensor::Init<double>(
            {{focal_length.first, 0, principal_point.first},
             {0, focal_length.second, principal_point.second},
             {0, 0, 1}});

    // Extrinsic
    std::string trajectory_path = GetDataPathCommon("RGBD/odometry.log");
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);
    size_t n = trajectory->parameters_.size();
    Eigen::Matrix4d extrinsic = trajectory->parameters_[n - 1].extrinsic_;
    core::Tensor extrinsic_t =
            core::eigen_converter::EigenMatrixToTensor(extrinsic);

    std::vector<core::HashmapBackend> backends;
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        backends.push_back(core::HashmapBackend::Slab);
        backends.push_back(core::HashmapBackend::StdGPU);
    } else {
        backends.push_back(core::HashmapBackend::TBB);
    }

    for (auto backend : backends) {
        auto voxel_grid = IntegrateTestScene(device, backend);

        if (backend == core::HashmapBackend::Slab) {
            EXPECT_THROW(voxel_grid.RayCast(intrinsic_t, extrinsic_t, cols,
                                            rows, depth_scale, 0.1, depth_max,
                                            std::min((n - 1) * 1.0f, 3.0f)),
                         std::runtime_error);
        } else {
            using MaskCode = t::geometry::TSDFVoxelGrid::SurfaceMaskCode;
            auto result = voxel_grid.RayCast(
                    intrinsic_t, extrinsic_t, cols, rows, depth_scale, 0.1,
                    depth_max, std::min((n - 1) * 1.0f, 3.0f),
                    MaskCode::VertexMap | MaskCode::ColorMap |
                            MaskCode::NormalMap);
            core::Tensor vertex_map = result[MaskCode::VertexMap];
            core::Tensor color_map = result[MaskCode::ColorMap];
            core::Tensor normal_map = result[MaskCode::NormalMap];

            t::geometry::Image vertex(result[MaskCode::VertexMap]);
            visualization::DrawGeometries(
                    {std::make_shared<open3d::geometry::Image>(
                            vertex.ToLegacy())});

            // There are CPU/CUDA numerical differences around edges, so
            // we need to be tolerant.
            core::Tensor vertex_map_gt = core::Tensor::Load(
                    GetDataPathDownload() +
                    fmt::format("/RGBD/raycast_vtx_{:03d}.npy", n - 1));
            vertex_map.Save(fmt::format("raycast_vtx_{:03d}.npy", n - 1));
            int64_t discrepancy_count =
                    ((vertex_map.To(core::Device("CPU:0")) - vertex_map_gt)
                             .Abs()
                             .Ge(1e-5))
                            .To(core::Int64)
                            .Sum({0, 1, 2})
                            .Item<int64_t>();
            EXPECT_LE(discrepancy_count * 1.0f / vertex_map.NumElements(),
                      1e-3);
        }
    }
}
}  // namespace tests
}  // namespace open3d
