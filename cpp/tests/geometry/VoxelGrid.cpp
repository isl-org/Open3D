// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/VoxelGrid.h"

#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(VoxelGrid, Bounds) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(1, 0, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 2, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 3)));
    ExpectEQ(voxel_grid->GetMinBound(), Eigen::Vector3d(0, 0, 0));
    ExpectEQ(voxel_grid->GetMaxBound(), Eigen::Vector3d(10, 15, 20));
}

TEST(VoxelGrid, GetVoxel) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 0, 0)),
             Eigen::Vector3i(0, 0, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 1, 0)),
             Eigen::Vector3i(0, 0, 0));
    // Test near boundary voxel_size_ == 5
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 4.9, 0)),
             Eigen::Vector3i(0, 0, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 5, 0)),
             Eigen::Vector3i(0, 1, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 5.1, 0)),
             Eigen::Vector3i(0, 1, 0));
}

TEST(VoxelGrid, Visualization) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 0),
                                         Eigen::Vector3d(0.9, 0, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 1, 0),
                                         Eigen::Vector3d(0.9, 0.9, 0)));

    // Uncomment the line below for visualization test
    // visualization::DrawGeometries({voxel_grid});
}

TEST(VoxelGrid, Scale) {
    open3d::geometry::PointCloud pcd;
    open3d::data::PLYPointCloud pointcloud_ply;
    open3d::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);

    double voxel_size = 0.01;
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid = geometry::VoxelGrid::CreateFromPointCloud(
            pcd, voxel_size, geometry::VoxelGrid::VoxelPoolingMode::AVG);

    Eigen::Vector3d original_center = voxel_grid->GetCenter();

    // Get original current voxels centers
    std::vector<Eigen::Vector3d> original_centers;
    for (const auto &it : voxel_grid->voxels_) {
        original_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }

    // Scale the voxel grid by 2x
    double scale = 2.0;
    voxel_grid->Scale(scale, voxel_grid->GetCenter());
    EXPECT_NEAR(voxel_grid->voxel_size_, voxel_size * scale, 1e-6);
    // Get new current voxels centers
    std::vector<Eigen::Vector3d> new_centers;
    for (const auto &it : voxel_grid->voxels_) {
        new_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }
    // Check that the new centers are scaled by 2x
    EXPECT_EQ(original_centers.size(), new_centers.size());
    for (size_t i = 0; i < original_centers.size(); i++) {
        Eigen::Vector3d vec = new_centers[i] - voxel_grid->GetCenter();
        Eigen::Vector3d vec_orig = original_centers[i] - original_center;
        ExpectEQ(vec, (vec_orig * scale).eval());
    }
}

TEST(VoxelGrid, Translate) {
    open3d::geometry::PointCloud pcd;
    open3d::data::PLYPointCloud pointcloud_ply;
    open3d::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);

    double voxel_size = 0.01;
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid = geometry::VoxelGrid::CreateFromPointCloud(
            pcd, voxel_size, geometry::VoxelGrid::VoxelPoolingMode::AVG);

    Eigen::Vector3d original_center = voxel_grid->GetCenter();

    // Get original current voxels centers
    std::vector<Eigen::Vector3d> original_centers;
    for (const auto &it : voxel_grid->voxels_) {
        original_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }

    // Translate by (1, 2, 3)
    Eigen::Vector3d translation(1, 2, 3);
    voxel_grid->Translate(translation, true);
    EXPECT_NEAR(voxel_grid->voxel_size_, voxel_size, 1e-6);
    ExpectEQ(voxel_grid->GetCenter(), (original_center + translation).eval());
    // Get new current voxels centers
    std::vector<Eigen::Vector3d> new_centers;
    for (const auto &it : voxel_grid->voxels_) {
        new_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }
    // Check that the new centers are translated by (1, 2, 3)
    EXPECT_EQ(original_centers.size(), new_centers.size());
    for (size_t i = 0; i < original_centers.size(); i++) {
        ExpectEQ(new_centers[i], (original_centers[i] + translation).eval());
    }
}

TEST(VoxelGrid, Rotate) {
    open3d::geometry::PointCloud pcd;
    open3d::data::PLYPointCloud pointcloud_ply;
    open3d::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);

    double voxel_size = 0.01;
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid = geometry::VoxelGrid::CreateFromPointCloud(
            pcd, voxel_size, geometry::VoxelGrid::VoxelPoolingMode::AVG);

    Eigen::Vector3d original_center = voxel_grid->GetCenter();

    // Get original current voxels centers
    std::vector<Eigen::Vector3d> original_centers;
    for (const auto &it : voxel_grid->voxels_) {
        original_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }

    // Rotate by 90 degrees around Z axis
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
    voxel_grid->Rotate(R, original_center);
    EXPECT_NEAR(voxel_grid->voxel_size_, voxel_size, 1e-6);
    // Get new current voxels centers
    std::vector<Eigen::Vector3d> new_centers;
    for (const auto &it : voxel_grid->voxels_) {
        new_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }
    // Check that the new centers are rotated by R
    EXPECT_EQ(original_centers.size(), new_centers.size());
    for (size_t i = 0; i < original_centers.size(); i++) {
        ExpectEQ(new_centers[i],
                 (R * (original_centers[i] - original_center) + original_center)
                         .eval());
    }
}

TEST(VoxelGrid, Transform) {
    open3d::geometry::PointCloud pcd;
    open3d::data::PLYPointCloud pointcloud_ply;
    open3d::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);

    double voxel_size = 0.01;
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid = geometry::VoxelGrid::CreateFromPointCloud(
            pcd, voxel_size, geometry::VoxelGrid::VoxelPoolingMode::AVG);

    // Get original current voxels centers
    std::vector<Eigen::Vector3d> original_centers;
    for (const auto &it : voxel_grid->voxels_) {
        original_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }

    // Transform
    Eigen::Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;
    voxel_grid->Transform(transformation);
    EXPECT_NEAR(voxel_grid->voxel_size_, voxel_size, 1e-6);
    // Get new current voxels centers
    std::vector<Eigen::Vector3d> new_centers;
    for (const auto &it : voxel_grid->voxels_) {
        new_centers.push_back(
                voxel_grid->GetVoxelCenterCoordinate(it.second.grid_index_));
    }
    // Check that the new centers are transformed by transformation
    EXPECT_EQ(original_centers.size(), new_centers.size());
    for (size_t i = 0; i < original_centers.size(); i++) {
        Eigen::Vector4d vec;
        vec << original_centers[i](0), original_centers[i](1),
                original_centers[i](2), 1;
        vec = transformation * vec;
        ExpectEQ(new_centers[i], vec.head<3>().eval());
    }
}

}  // namespace tests
}  // namespace open3d
