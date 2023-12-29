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
    utility::LogInfo("    > MergeMesh source_directory target_file [option]");
    utility::LogInfo("      Merge mesh files under <source_directory>.");
    utility::LogInfo("");
    utility::LogInfo("Options (listed in the order of execution priority):");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).");
    utility::LogInfo("    --purge                   : Clear duplicated and unreferenced vertices and");
    utility::LogInfo("                                triangles.");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc <= 2 || utility::ProgramOptionExists(argc, argv, "--help")) {
        PrintHelp();
        return 0;
    }
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);

    std::string directory(argv[1]);
    std::vector<std::string> filenames;
    ListFilesInDirectory(directory, filenames);

    auto merged_mesh_ptr = std::make_shared<geometry::TriangleMesh>();
    for (const auto &filename : filenames) {
        auto mesh_ptr = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(filename, *mesh_ptr)) {
            *merged_mesh_ptr += *mesh_ptr;
        }
    }

    if (utility::ProgramOptionExists(argc, argv, "--purge")) {
        merged_mesh_ptr->RemoveDuplicatedVertices();
        merged_mesh_ptr->RemoveDuplicatedTriangles();
        merged_mesh_ptr->RemoveUnreferencedVertices();
        merged_mesh_ptr->RemoveDegenerateTriangles();
    }
    io::WriteTriangleMesh(argv[2], *merged_mesh_ptr);

    return 1;
}
