// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

using namespace open3d;

void PrintVoxelGridInformation(const geometry::VoxelGrid& voxel_grid) {
    utility::LogInfo("geometry::VoxelGrid with {:d} voxels",
                     voxel_grid.voxels_.size());
    utility::LogInfo("               origin: [{:f} {:f} {:f}]",
                     voxel_grid.origin_(0), voxel_grid.origin_(1),
                     voxel_grid.origin_(2));
    utility::LogInfo("               voxel_size: {:f}", voxel_grid.voxel_size_);
    return;
}

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Voxelization [pointcloud_filename] [voxel_filename_ply]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    auto voxel = geometry::VoxelGrid::CreateFromPointCloud(*pcd, 0.05);
    PrintVoxelGridInformation(*voxel);
    visualization::DrawGeometries({pcd, voxel});
    io::WriteVoxelGrid(argv[2], *voxel, true);

    auto voxel_read = io::CreateVoxelGridFromFile(argv[2]);
    PrintVoxelGridInformation(*voxel_read);
    visualization::DrawGeometries({pcd, voxel_read});
}
