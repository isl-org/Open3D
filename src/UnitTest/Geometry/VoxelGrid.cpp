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

#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Geometry/LineSet.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace unit_test;

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
