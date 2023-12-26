// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/VoxelGrid.h"

#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/TriangleMesh.h"
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

}  // namespace tests
}  // namespace open3d
