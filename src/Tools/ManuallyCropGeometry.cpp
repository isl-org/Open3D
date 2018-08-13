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

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

void PrintHelp()
{
    using namespace open3d;
    PrintOpen3DVersion();
    PrintInfo("Usage:\n");
    PrintInfo("    > ManuallyCropGeometry [--pointcloud/mesh] geometry_file [options]\n");
    PrintInfo("      Manually crop geometry in speficied file.\n");
    PrintInfo("\n");
    PrintInfo("Options:\n");
    PrintInfo("    --pointcloud,             : Read geometry as point cloud.\n");
    PrintInfo("    --mesh,                   : Read geometry as mesh.\n");
    PrintInfo("    --help, -h                : Print help information.\n");
    PrintInfo("    --verbose n               : Set verbose level (0-4).\n");
    PrintInfo("    --voxel_size d            : Set downsample voxel size.\n");
    PrintInfo("    --without_dialog          : Disable dialogs. Default files will be used.\n");
}

int main(int argc, char **argv)
{
    using namespace open3d;

    if (argc < 2 || ProgramOptionExists(argc, argv, "--help") ||
            ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    SetVerbosityLevel((VerbosityLevel)verbose);
    double voxel_size = GetProgramOptionAsDouble(argc, argv, "--voxel_size",
            -1.0);
    bool with_dialog = !ProgramOptionExists(argc, argv, "--without_dialog");

    VisualizerWithEditing vis(voxel_size, with_dialog,
            filesystem::GetFileParentDirectory(argv[1]));
    vis.CreateVisualizerWindow("Crop Point Cloud", 1920, 1080, 100, 100);
    if (ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd_ptr = CreatePointCloudFromFile(argv[2]);
        if (pcd_ptr->IsEmpty()) {
            PrintWarning("Failed to read the point cloud.\n");
            return 0;
        }
        vis.AddGeometry(pcd_ptr);
        if (pcd_ptr->points_.size() > 5000000) {
            vis.GetRenderOption().point_size_ = 1.0;
        }
    } else if (ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh_ptr = CreateMeshFromFile(argv[2]);
        if (mesh_ptr->IsEmpty()) {
            PrintWarning("Failed to read the mesh.\n");
            return 0;
        }
        vis.AddGeometry(mesh_ptr);
    }
    vis.Run();
    vis.DestroyVisualizerWindow();
    return 1;
}
