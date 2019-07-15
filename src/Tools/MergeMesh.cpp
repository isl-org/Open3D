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

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:\n");
    utility::LogInfo("    > MergeMesh source_directory target_file [option]\n");
    utility::LogInfo("      Merge mesh files under <source_directory>.\n");
    utility::LogInfo("\n");
    utility::LogInfo("Options (listed in the order of execution priority):\n");
    utility::LogInfo("    --help, -h                : Print help information.\n");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).\n");
    utility::LogInfo("    --purge                   : Clear duplicated and unreferenced vertices and\n");
    utility::LogInfo("                                triangles.\n");
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
