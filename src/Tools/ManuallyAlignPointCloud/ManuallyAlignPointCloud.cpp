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

#include <thread>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include "VisualizerForAlignment.h"

void PrintHelp()
{
	printf("Usage:\n");
	printf("    > ManuallyAlignPointCloud source_file target_file [options]\n");
	printf("      Manually align point clouds in source_file and target_file.\n");
	printf("\n");
	printf("Options:\n");
	printf("    --help, -h                : Print help information.\n");
	printf("    --verbose n               : Set verbose level (0-4).\n");
	printf("    --voxel_size d            : Set downsample voxel size.\n");
	printf("    --max_corres_distance d   : Set max correspondence distance.\n");
	printf("    --without_scaling         : Disable scaling in transformations.\n");
	printf("    --without_dialog          : Disable dialogs. Default files will be used.\n");
}

int main(int argc, char **argv)
{
	using namespace three;

	if (argc < 3 || ProgramOptionExists(argc, argv, "--help") ||
			ProgramOptionExists(argc, argv, "-h")) {
		PrintHelp();
		return 0;
	}
	
	int verbose = GetProgramOptionAsInt(argc, argv, "--verbose", 2);
	SetVerbosityLevel((VerbosityLevel)verbose);
	double voxel_size = GetProgramOptionAsDouble(argc, argv, "--voxel_size",
			-1.0);
	double max_corres_distance = GetProgramOptionAsDouble(argc, argv,
			"--max_corres_distance", -1.0);
	bool with_scaling = !ProgramOptionExists(argc, argv, "--without_scaling");
	bool with_dialog = !ProgramOptionExists(argc, argv, "--without_dialog");
	std::string default_polygon_filename =
			filesystem::GetFileNameWithoutExtension(argv[2]) + ".json";

	auto source_ptr = CreatePointCloudFromFile(argv[1]);
	auto target_ptr = CreatePointCloudFromFile(argv[2]);
	if (source_ptr->IsEmpty() || target_ptr->IsEmpty()) {
		PrintWarning("Failed to read one of the point clouds.\n");
		return 0;
	}
	VisualizerWithEditing vis_source, vis_target;
	VisualizerForAlignment vis_main(vis_source, vis_target, voxel_size,
			max_corres_distance, with_scaling, with_dialog,
			default_polygon_filename);
	
	vis_source.CreateWindow("Source Point Cloud", 1280, 720, 10, 100);
	vis_source.AddGeometry(source_ptr);
	if (source_ptr->points_.size() > 5000000) {
		vis_source.GetRenderOption().point_size_ = 1.0;
	}
	vis_source.BuildUtilities();
	vis_target.CreateWindow("Target Point Cloud", 1280, 720, 10, 880);
	vis_target.AddGeometry(target_ptr);
	if (target_ptr->points_.size() > 5000000) {
		vis_target.GetRenderOption().point_size_ = 1.0;
	}
	vis_target.BuildUtilities();
	vis_main.CreateWindow("Alignment", 1280, 1440, 1300, 100);
	vis_main.AddSourceAndTarget(source_ptr, target_ptr);
	vis_main.BuildUtilities();

	while (vis_source.PollEvents() && vis_target.PollEvents() &&
			vis_main.PollEvents()) {
	}
	
	vis_source.DestroyWindow();
	vis_target.DestroyWindow();
	vis_main.DestroyWindow();
	return 1;
}
