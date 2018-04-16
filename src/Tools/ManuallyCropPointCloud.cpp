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
	printf("Usage:\n");
	printf("    > ManuallyCropPointCloud pointcloud_file [options]\n");
	printf("      Manually crop point clouds in pointcloud_file.\n");
	printf("\n");
	printf("Options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4).\n");
	printf("    --voxel_size d            : Set downsample voxel size.\n");
	printf("    --without_dialog          : Disable dialogs. Default files will be used.\n");
}

int main(int argc, char **argv)
{
	using namespace three;

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

	auto pcd_ptr = CreatePointCloudFromFile(argv[1]);
	if (pcd_ptr->IsEmpty()) {
		PrintWarning("Failed to read the point cloud.\n");
		return 0;
	}
	VisualizerWithEditing vis(voxel_size, with_dialog,
			filesystem::GetFileParentDirectory(argv[1]));
	vis.CreateWindow("Crop Point Cloud", 1920, 1080, 100, 100);
	vis.AddGeometry(pcd_ptr);
	if (pcd_ptr->points_.size() > 5000000) {
		vis.GetRenderOption().point_size_ = 1.0;
	}
	vis.Run();
	vis.DestroyWindow();
	return 1;
}
