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

#include <Core/Core.h>
#include <IO/IO.h>

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > IntegrateRGBD [options]\n");
	printf("      Integrate RGBD stream and extract geometry.\n");
	printf("\n");
	printf("Basic options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --match file              : The match file of an RGBD stream. Must have.\n");
	printf("    --log file                : The log trajectory file. Must have.\n");
	printf("    --save_pointcloud         : Save a point cloud created by marching cubes.\n");
	printf("    --save_mesh               : Save a mesh created by marching cubes.\n");
	printf("    --save_voxel              : Save a point cloud of the TSDF voxel.\n");
	printf("    --every_k_frames k        : Save/reset every k frames. Default: 0 (none).\n");
	printf("    --length l                : Length of the volume, in meters. Default: 4.0.\n");
	printf("    --resolution r            : Resolution of the voxel grid. Default: 512.\n");
	printf("    --sdf_trunc_percentage t  : TSDF truncation percentage, of the volume length. Default: 0.01.\n");
	printf("    --verbose n               : Set verbose level (0-4). Default: 2.\n");
}

int main(int argc, char *argv[])
{
	using namespace three;

	if (argc <= 1 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}

	std::string match_filename = GetProgramOptionAsString(argc, argv,
			"--match");
	std::string log_filename = GetProgramOptionAsString(argc, argv, "--log");
	bool save_pointcloud = ProgramOptionExists(argc, argv, "--save_pointcloud");
	bool save_mesh = ProgramOptionExists(argc, argv, "--save_mesh");
	bool save_voxel = ProgramOptionExists(argc, argv, "--save_voxel");
	int every_k_frames = GetProgramOptionAsInt(argc, argv, "--every_k_frames",
			0);
	double length = GetProgramOptionAsDouble(argc, argv, "--length", 4.0);
	int resolution = GetProgramOptionAsInt(argc, argv, "--resolution", 512);
	double sdf_trunc_percentage = GetProgramOptionAsDouble(argc, argv,
			"--sdf_trunc_percentage", 0.01);
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);

	auto camera_trajectory = CreatePinholeCameraTrajectoryFromFile(
			log_filename);
	std::string dir_name = filesystem::GetFileParentDirectory(
			match_filename).c_str();
	FILE *file = fopen(match_filename.c_str(), "r");
	if (file == NULL) {
		PrintError("Unable to open file %s\n", match_filename.c_str());
		fclose(file);
		return 0;
	}
	char buffer[DEFAULT_IO_BUFFER_SIZE];
	int index = 0;
	int save_index = 0;
	//UniformTSDFVolume volume(length, resolution, length * sdf_trunc_percentage,
	//		true);
	ScalableTSDFVolume volume(length / (double)resolution,
			length * sdf_trunc_percentage, true);
	FPSTimer timer("Process RGBD stream",
			(int)camera_trajectory->extrinsic_.size());
	Image depth, color;
	while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
		std::vector<std::string> st;
		SplitString(st, buffer, "\t\r\n ");
		if (st.size() >= 2) {
			PrintDebug("Processing frame %d ...\n", index);
			ReadImage(dir_name + st[0], depth);
			ReadImage(dir_name + st[1], color);
			auto rgbd = CreateRGBDImageFromColorAndDepth(color, depth, 1000.0,
					4.0, false);
			if (index == 0 ||
					(every_k_frames > 0 && index % every_k_frames == 0)) {
				volume.Reset();
			}
			volume.Integrate(*rgbd, camera_trajectory->intrinsic_,
					camera_trajectory->extrinsic_[index]);
			index++;
			if (index == (int)camera_trajectory->extrinsic_.size() ||
					(every_k_frames > 0 && index % every_k_frames == 0)) {
				PrintDebug("Saving fragment %d ...\n", save_index);
				std::string save_index_str = std::to_string(save_index);
				if (save_pointcloud) {
					PrintDebug("Saving pointcloud %d ...\n", save_index);
					auto pcd = volume.ExtractPointCloud();
					WritePointCloud("pointcloud_" + save_index_str + ".ply",
							*pcd);
				}
				if (save_mesh) {
					PrintDebug("Saving mesh %d ...\n", save_index);
					auto mesh = volume.ExtractTriangleMesh();
					WriteTriangleMesh("mesh_" + save_index_str + ".ply",
							*mesh);
				}
				if (save_voxel) {
					PrintDebug("Saving voxel %d ...\n", save_index);
					auto voxel = volume.ExtractVoxelPointCloud();
					WritePointCloud("voxel_" + save_index_str + ".ply",
							*voxel);
				}
				save_index++;
			}
			timer.Signal();
		}
	}
	fclose(file);
	return 1;
}
