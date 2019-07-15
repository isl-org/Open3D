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

#include <iostream>
#include <memory>

#include "Open3D/Open3D.h"

void PrintHelp() {
    using namespace open3d;
    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:\n");
    utility::LogInfo("    > IntegrateRGBD [options]\n");
    utility::LogInfo("      Integrate RGBD stream and extract geometry.\n");
    utility::LogInfo("\n");
    utility::LogInfo("Basic options:\n");
    utility::LogInfo("    --help, -h                : Print help information.\n");
    utility::LogInfo("    --match file              : The match file of an RGBD stream. Must have.\n");
    utility::LogInfo("    --log file                : The log trajectory file. Must have.\n");
    utility::LogInfo("    --save_pointcloud         : Save a point cloud created by marching cubes.\n");
    utility::LogInfo("    --save_mesh               : Save a mesh created by marching cubes.\n");
    utility::LogInfo("    --save_voxel              : Save a point cloud of the TSDF voxel.\n");
    utility::LogInfo("    --every_k_frames k        : Save/reset every k frames. Default: 0 (none).\n");
    utility::LogInfo("    --length l                : Length of the volume, in meters. Default: 4.0.\n");
    utility::LogInfo("    --resolution r            : Resolution of the voxel grid. Default: 512.\n");
    utility::LogInfo("    --sdf_trunc_percentage t  : TSDF truncation percentage, of the volume length. Default: 0.01.\n");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    if (argc <= 1 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }

    std::string match_filename =
            utility::GetProgramOptionAsString(argc, argv, "--match");
    std::string log_filename =
            utility::GetProgramOptionAsString(argc, argv, "--log");
    bool save_pointcloud =
            utility::ProgramOptionExists(argc, argv, "--save_pointcloud");
    bool save_mesh = utility::ProgramOptionExists(argc, argv, "--save_mesh");
    bool save_voxel = utility::ProgramOptionExists(argc, argv, "--save_voxel");
    int every_k_frames =
            utility::GetProgramOptionAsInt(argc, argv, "--every_k_frames", 0);
    double length =
            utility::GetProgramOptionAsDouble(argc, argv, "--length", 4.0);
    int resolution =
            utility::GetProgramOptionAsInt(argc, argv, "--resolution", 512);
    double sdf_trunc_percentage = utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc_percentage", 0.01);
    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    utility::SetVerbosityLevel((utility::VerbosityLevel)verbose);

    auto camera_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(log_filename);
    std::string dir_name =
            utility::filesystem::GetFileParentDirectory(match_filename).c_str();
    FILE *file = fopen(match_filename.c_str(), "r");
    if (file == NULL) {
        utility::LogError("Unable to open file {}\n", match_filename);
        fclose(file);
        return 0;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];
    int index = 0;
    int save_index = 0;
    integration::ScalableTSDFVolume volume(
            length / (double)resolution, length * sdf_trunc_percentage,
            integration::TSDFVolumeColorType::RGB8);
    utility::FPSTimer timer("Process RGBD stream",
                            (int)camera_trajectory->parameters_.size());
    geometry::Image depth, color;
    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::vector<std::string> st;
        utility::SplitString(st, buffer, "\t\r\n ");
        if (st.size() >= 2) {
            utility::LogInfo("Processing frame {:d} ...\n", index);
            io::ReadImage(dir_name + st[0], depth);
            io::ReadImage(dir_name + st[1], color);
            auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
                    color, depth, 1000.0, 4.0, false);
            if (index == 0 ||
                (every_k_frames > 0 && index % every_k_frames == 0)) {
                volume.Reset();
            }
            volume.Integrate(*rgbd,
                             camera_trajectory->parameters_[index].intrinsic_,
                             camera_trajectory->parameters_[index].extrinsic_);
            index++;
            if (index == (int)camera_trajectory->parameters_.size() ||
                (every_k_frames > 0 && index % every_k_frames == 0)) {
                utility::LogInfo("Saving fragment {:d} ...\n", save_index);
                std::string save_index_str = std::to_string(save_index);
                if (save_pointcloud) {
                    utility::LogInfo("Saving pointcloud {:d} ...\n",
                                     save_index);
                    auto pcd = volume.ExtractPointCloud();
                    io::WritePointCloud("pointcloud_" + save_index_str + ".ply",
                                        *pcd);
                }
                if (save_mesh) {
                    utility::LogInfo("Saving mesh {:d} ...\n", save_index);
                    auto mesh = volume.ExtractTriangleMesh();
                    io::WriteTriangleMesh("mesh_" + save_index_str + ".ply",
                                          *mesh);
                }
                if (save_voxel) {
                    utility::LogInfo("Saving voxel {:d} ...\n", save_index);
                    auto voxel = volume.ExtractVoxelPointCloud();
                    io::WritePointCloud("voxel_" + save_index_str + ".ply",
                                        *voxel);
                }
                save_index++;
            }
            timer.Signal();
        }
    }
    fclose(file);
    return 0;
}
