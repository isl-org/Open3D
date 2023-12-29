// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > PCDFileFormat [filename] [ascii|binary|compressed]");
    utility::LogInfo("      The program will :");
    utility::LogInfo("      1. load the pointcloud in [filename].");
    utility::LogInfo("      2. visualize the point cloud.");
    utility::LogInfo("      3. if a save method is specified, write the point cloud into data.pcd.");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (!(argc == 2 || argc == 3) ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto cloud_ptr = io::CreatePointCloudFromFile(argv[1]);
    visualization::DrawGeometries({cloud_ptr}, "TestPCDFileFormat", 1920, 1080);

    if (argc >= 3) {
        std::string method(argv[2]);
        if (method == "ascii") {
            io::WritePointCloud("data.pcd", *cloud_ptr, {true});
        } else if (method == "binary") {
            io::WritePointCloud("data.pcd", *cloud_ptr, {false, false});
        } else if (method == "compressed") {
            io::WritePointCloud("data.pcd", *cloud_ptr, {false, true});
        }
    }

    return 0;
}
