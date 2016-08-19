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

#include <limits>

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ConvertPointCloud source_file target_file [options]\n");
	printf("      Read point cloud from source file and convert it to target file.\n");
	printf("\n");
	printf("Options:\n");
	printf("    --voxel_sample voxel_size : Downsample the point cloud with a voxel.\n");
	printf("    --clip_x_min x0           : Clip points with x coordinate less than x0.\n");
	printf("    --clip_x_max x1           : Clip points with x coordinate larger than x1.\n");
	printf("    --clip_y_min y0           : Clip points with y coordinate less than y0.\n");
	printf("    --clip_y_max y1           : Clip points with y coordinate larger than y1.\n");
	printf("    --clip_z_min z0           : Clip points with z coordinate less than z0.\n");
	printf("    --clip_z_max z1           : Clip points with z coordinate larger than z1.\n");
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
	
	auto pointcloud_ptr = CreatePointCloudFromFile(argv[1]);
	
	// clip
	if (ProgramOptionExistsAny(argc, argv, {"--clip_x_min", "--clip_x_max",
			"--clip_y_min", "--clip_y_max", "--clip_z_min", "--clip_z_max"})) {
		Eigen::Vector3d min_bound, max_bound;
		min_bound(0) = GetProgramOptionAsDouble(argc, argv, "--clip_x_min",
				std::numeric_limits<double>::lowest());
		min_bound(1) = GetProgramOptionAsDouble(argc, argv, "--clip_y_min",
				std::numeric_limits<double>::lowest());
		min_bound(2) = GetProgramOptionAsDouble(argc, argv, "--clip_z_min",
				std::numeric_limits<double>::lowest());
		max_bound(0) = GetProgramOptionAsDouble(argc, argv, "--clip_x_max",
				std::numeric_limits<double>::max());
		max_bound(1) = GetProgramOptionAsDouble(argc, argv, "--clip_y_max",
				std::numeric_limits<double>::max());
		max_bound(2) = GetProgramOptionAsDouble(argc, argv, "--clip_z_max",
				std::numeric_limits<double>::max());
		auto clip_ptr = std::make_shared<PointCloud>();
		ClipPointCloud(*pointcloud_ptr, min_bound, max_bound, *clip_ptr);
		pointcloud_ptr = clip_ptr;
	}
	
	// voxel_downsample
	double voxel_size = GetProgramOptionAsDouble(argc, argv, "--voxel_sample",
			0.0);
	if (voxel_size > 0.0) {
		auto downsample_ptr = std::make_shared<PointCloud>();
		VoxelDownSample(*pointcloud_ptr, voxel_size, *downsample_ptr);
		pointcloud_ptr = downsample_ptr;
	}
	WritePointCloud(argv[2], *pointcloud_ptr, false, true);
	return 1;
}
