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
    utility::LogInfo("    > ManuallyCropGeometry [--pointcloud/mesh] geometry_file [options]\n");
    utility::LogInfo("      Manually crop geometry in speficied file.\n");
    utility::LogInfo("\n");
    utility::LogInfo("Options:\n");
    utility::LogInfo("    --pointcloud,             : Read geometry as point cloud.\n");
    utility::LogInfo("    --mesh,                   : Read geometry as mesh.\n");
    utility::LogInfo("    --help, -h                : Print help information.\n");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).\n");
    utility::LogInfo("    --voxel_size d            : Set downsample voxel size.\n");
    utility::LogInfo("    --without_dialog          : Disable dialogs. Default files will be used.\n");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace open3d;

    if (argc < 2 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);
    double voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", -1.0);
    bool with_dialog =
            !utility::ProgramOptionExists(argc, argv, "--without_dialog");

    visualization::VisualizerWithEditing vis(
            voxel_size, with_dialog,
            utility::filesystem::GetFileParentDirectory(argv[1]));
    vis.CreateVisualizerWindow("Crop Point Cloud", 1920, 1080, 100, 100);
    if (utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd_ptr = io::CreatePointCloudFromFile(argv[2]);
        if (pcd_ptr->IsEmpty()) {
            utility::LogWarning("Failed to read the point cloud.\n");
            return 0;
        }
        vis.AddGeometry(pcd_ptr);
        if (pcd_ptr->points_.size() > 5000000) {
            vis.GetRenderOption().point_size_ = 1.0;
        }
    } else if (utility::ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh_ptr = io::CreateMeshFromFile(argv[2]);
        if (mesh_ptr->IsEmpty()) {
            utility::LogWarning("Failed to read the mesh.\n");
            return 0;
        }
        vis.AddGeometry(mesh_ptr);
    }
    vis.Run();
    vis.DestroyVisualizerWindow();
    return 1;
}
