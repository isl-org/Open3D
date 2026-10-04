// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/integration/ScalableTSDFVolume.h"

#include <algorithm>

#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

template <typename Depth>
std::shared_ptr<geometry::RGBDImage> CreatePlaneRGBD(Depth depth_value,
                                                     double depth_scale,
                                                     double depth_trunc = 3.0) {
    geometry::Image color, depth;
    color.Prepare(32, 24, 3, 1);
    depth.Prepare(32, 24, 1, sizeof(Depth));
    std::fill(color.data_.begin(), color.data_.end(), 128);
    std::fill_n(depth.PointerAs<Depth>(), 32 * 24, depth_value);
    return geometry::RGBDImage::CreateFromColorAndDepth(
            color, depth, depth_scale, depth_trunc, false);
}

// Use the same physical scene for each encoding and sampling stride.
std::pair<std::shared_ptr<geometry::PointCloud>,
          std::shared_ptr<geometry::TriangleMesh>>
ReconstructPlane(const geometry::RGBDImage& rgbd,
                 int stride,
                 double voxel_length = 0.05) {
    camera::PinholeCameraIntrinsic intrinsic(32, 24, 28.0, 28.0, 15.5, 11.5);
    pipelines::integration::ScalableTSDFVolume volume(
            voxel_length, 2.0 * voxel_length,
            pipelines::integration::TSDFVolumeColorType::RGB8, 16, stride);
    volume.Integrate(rgbd, intrinsic, Eigen::Matrix4d::Identity());
    return {volume.ExtractPointCloud(), volume.ExtractTriangleMesh()};
}

void ExpectPlane(const geometry::PointCloud& pointcloud,
                 const geometry::TriangleMesh& mesh,
                 double depth,
                 double voxel_length) {
    ASSERT_FALSE(pointcloud.points_.empty());
    ASSERT_FALSE(mesh.vertices_.empty());
    ASSERT_FALSE(mesh.triangles_.empty());
    // TSDF zero crossings are interpolated between neighboring voxel centers.
    for (const auto* points : {&pointcloud.points_, &mesh.vertices_}) {
        for (const auto& point : *points) {
            EXPECT_TRUE(point.allFinite());
            EXPECT_NEAR(point.z(), depth, voxel_length);
        }
    }
}

}  // namespace

TEST(ScalableTSDFVolume, IntegrateMetricDepth) {
    const auto meters = CreatePlaneRGBD(2.0f, 1.0);
    const auto millimeters = CreatePlaneRGBD(uint16_t(2000), 1000.0);
    for (int stride : {1, 4}) {
        SCOPED_TRACE(stride);
        auto [meter_points, meter_mesh] = ReconstructPlane(*meters, stride);
        auto [millimeter_points, millimeter_mesh] =
                ReconstructPlane(*millimeters, stride);
        ExpectPlane(*meter_points, *meter_mesh, 2.0, 0.05);
        ExpectPlane(*millimeter_points, *millimeter_mesh, 2.0, 0.05);
        ASSERT_EQ(meter_points->points_.size(),
                  millimeter_points->points_.size());
        ExpectEQ(Sort(meter_points->points_), Sort(millimeter_points->points_));
        ASSERT_EQ(meter_mesh->vertices_.size(),
                  millimeter_mesh->vertices_.size());
        ExpectEQ(Sort(meter_mesh->vertices_), Sort(millimeter_mesh->vertices_));
        EXPECT_EQ(meter_mesh->triangles_.size(),
                  millimeter_mesh->triangles_.size());
    }
}

TEST(ScalableTSDFVolume, IntegrateDepthBeyondDefaultPointCloudTruncation) {
    // RGBD already applied the caller's truncation. Scale the scene to keep
    // the voxel count small while exercising depths beyond 1000 meters.
    const auto rgbd = CreatePlaneRGBD(2000.0f, 1.0, 5000.0);
    auto [points, mesh] = ReconstructPlane(*rgbd, 4, 50.0);
    ExpectPlane(*points, *mesh, 2000.0, 50.0);
}

TEST(ScalableTSDFVolume, DepthSamplingStride) {
    auto rgbd = CreatePlaneRGBD(2.0f, 1.0);
    for (int y = 0; y < 24; y += 4) {
        for (int x = 0; x < 32; x += 4) {
            *rgbd->depth_.PointerAt<float>(x, y) = 0.0f;
        }
    }
    auto [dense_points, dense_mesh] = ReconstructPlane(*rgbd, 1);
    ExpectPlane(*dense_points, *dense_mesh, 2.0, 0.05);
    auto [sampled_points, sampled_mesh] = ReconstructPlane(*rgbd, 4);
    EXPECT_TRUE(sampled_points->points_.empty());
    EXPECT_TRUE(sampled_mesh->vertices_.empty());
    EXPECT_TRUE(sampled_mesh->triangles_.empty());
}

TEST(ScalableTSDFVolume, DISABLED_VolumeUnit) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_Constructor) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_Destructor) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_MemberData) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_Reset) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_Integrate) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_ExtractPointCloud) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_ExtractTriangleMesh) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_ExtractVoxelPointCloud) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_LocateVolumeUnit) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_OpenVolumeUnit) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_GetNormalAt) { NotImplemented(); }

TEST(ScalableTSDFVolume, DISABLED_GetTSDFAt) { NotImplemented(); }

}  // namespace tests
}  // namespace open3d
