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

#include "Open3D/IO/ClassIO/VoxelGridIO.h"
#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"
#include "TestUtility/UnitTest.h"

using namespace open3d;
using namespace unit_test;

TEST(VoxelGridIO, PLYWriteRead) {
    // Create voxel_grid (two voxels)
    auto src_voxel_grid = std::make_shared<geometry::VoxelGrid>();
    src_voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    src_voxel_grid->voxel_size_ = 5;
    src_voxel_grid->voxels_ = {
            geometry::Voxel(Eigen::Vector3i(1, 2, 3),
                            Eigen::Vector3d(0.1, 0.2, 0.3)),
            geometry::Voxel(Eigen::Vector3i(4, 5, 6),
                            Eigen::Vector3d(0.4, 0.5, 0.6)),
    };

    // Write to file
    std::string file_name = std::string(TEST_DATA_DIR) + "/temp_voxel_grid.ply";
    EXPECT_TRUE(io::WriteVoxelGrid(file_name, *src_voxel_grid));

    // Read from file
    auto dst_voxel_grid = std::make_shared<geometry::VoxelGrid>();
    EXPECT_TRUE(io::ReadVoxelGrid(file_name, *dst_voxel_grid));
    EXPECT_EQ(std::remove(file_name.c_str()), 0);

    // Check values, account for unit8 conversion lost
    for (size_t i = 0; i < src_voxel_grid->voxels_.size(); ++i) {
        ExpectEQ(src_voxel_grid->voxels_[i].grid_index_,
                 dst_voxel_grid->voxels_[i].grid_index_);
        ExpectEQ(Eigen::Vector3d(src_voxel_grid->voxels_[i]
                                         .color_.cast<uint8_t>()
                                         .cast<double>()),
                 Eigen::Vector3d(dst_voxel_grid->voxels_[i]
                                         .color_.cast<uint8_t>()
                                         .cast<double>()));
    }

    // Uncomment the line below for visualization test
    // visualization::DrawGeometries({dst_voxel_grid});
}
