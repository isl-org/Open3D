// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ConvertPointCloud source_file target_file [options]\n");
	printf("      Read point cloud from source file and convert it to target file.\n");
	printf("\n");
	printf("Options:\n");
	printf("    --voxel_sample voxel_size : Downsample the point cloud with a voxel.\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4).\n");
}

int main(int argc, char **argv)
{
	using namespace three;
	using namespace three::filesystem;

	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);
	if (argc < 3 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	
	double voxel_size = GetProgramOptionAsDouble(argc, argv, "--voxel_sample",
			0.0);
	auto pointcloud_ptr = CreatePointCloudFromFile(argv[1]);
	if (voxel_size > 0.0) {
		auto downsample_ptr = std::make_shared<PointCloud>();
		VoxelDownSample(*pointcloud_ptr, voxel_size, *downsample_ptr);
		pointcloud_ptr = downsample_ptr;
	}
	WritePointCloud(argv[2], *pointcloud_ptr, false, true);
	return 1;
}
