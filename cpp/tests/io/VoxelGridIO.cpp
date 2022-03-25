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

#include "open3d/io/VoxelGridIO.h"

#include "open3d/geometry/VoxelGrid.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(VoxelGridIO, PLYWriteRead) {
    // Create voxel_grid (two voxels)
    auto src_voxel_grid = std::make_shared<geometry::VoxelGrid>();
    src_voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    src_voxel_grid->voxel_size_ = 5;
    src_voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(1, 2, 3),
                                             Eigen::Vector3d(0.1, 0.2, 0.3)));
    src_voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(4, 5, 6),
                                             Eigen::Vector3d(0.4, 0.5, 0.6)));

    // Write to file
    std::string file_name = utility::filesystem::GetTempDirectoryPath() +
                            "/temp_voxel_grid.ply";
    EXPECT_TRUE(io::WriteVoxelGrid(file_name, *src_voxel_grid));

    // Read from file
    auto dst_voxel_grid = std::make_shared<geometry::VoxelGrid>();
    EXPECT_TRUE(io::ReadVoxelGrid(file_name, *dst_voxel_grid));

    // Check values, account for unit8 conversion lost
    EXPECT_EQ(src_voxel_grid->origin_, dst_voxel_grid->origin_);
    EXPECT_EQ(src_voxel_grid->voxel_size_, dst_voxel_grid->voxel_size_);
    EXPECT_EQ(src_voxel_grid->voxels_.size(), dst_voxel_grid->voxels_.size());
    for (auto &src_it : src_voxel_grid->voxels_) {
        const auto &src_voxel = src_it.second;
        const auto &src_i = src_voxel.grid_index_;
        const auto &src_c = src_voxel.color_;
        ExpectEQ(src_i, dst_voxel_grid->voxels_[src_i].grid_index_);
        auto src_rgb = utility::ColorToUint8(src_c);
        auto dst_rgb =
                utility::ColorToUint8(dst_voxel_grid->voxels_[src_i].color_);
        ExpectEQ(src_rgb, dst_rgb);
    }

    // Uncomment the line below for visualization test
    // visualization::DrawGeometries({dst_voxel_grid});
}

}  // namespace tests
}  // namespace open3d
