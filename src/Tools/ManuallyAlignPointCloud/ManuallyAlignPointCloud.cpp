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
	printf("    --without_scaling         : Disable scaling in transformations.\n");
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
	bool with_scaling = !ProgramOptionExists(argc, argv, "--without_scaling");

	auto source_ptr = CreatePointCloudFromFile(argv[1]);
	auto target_ptr = CreatePointCloudFromFile(argv[2]);
	auto source_trans_ptr = std::make_shared<PointCloud>();
	*source_trans_ptr = *source_ptr;
	if (source_ptr->IsEmpty() || target_ptr->IsEmpty()) {
		PrintWarning("Failed to read one of the point clouds.\n");
		return 0;
	}
	VisualizerWithEditing vis_source, vis_target;
	VisualizerWithKeyCallback vis_main;
	
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
	vis_main.GetRenderOption().point_size_ = 1.0;
	vis_main.AddGeometry(target_ptr);
	vis_main.AddGeometry(source_trans_ptr);
	vis_main.RegisterKeyCallback(GLFW_KEY_A,
			[&vis_source, &vis_target, source_ptr, target_ptr, source_trans_ptr,
			with_scaling](Visualizer &vis) {
				const auto &source_idx = vis_source.GetPickedPoints();
				const auto &target_idx = vis_target.GetPickedPoints();
				if (source_idx.empty() || target_idx.empty() ||
						source_idx.size() != target_idx.size()) {
					PrintWarning("# of picked points mismatch: %d in source, %d in target.\n",
							(int)source_idx.size(), (int)target_idx.size());
					return false;
				}
				TransformationEstimationPointToPoint p2p(with_scaling);
				TransformationEstimation::CorrespondenceSet corres;
				for (size_t i = 0; i < source_idx.size(); i++) {
					corres.push_back(std::make_pair((int)source_idx[i],
							(int)target_idx[i]));
				}
				PrintInfo("Error is %.4f before alignment.\n",
						p2p.ComputeError(*source_ptr, *target_ptr, corres));
				auto trans = p2p.ComputeTransformation(*source_ptr, *target_ptr,
						corres);
				PrintInfo("Transformation is:\n");
				PrintInfo("\t%.6f %.6f %.6f %.6f\n", trans(0, 0), trans(0, 1),
						trans(0, 2), trans(0, 3));
				PrintInfo("\t%.6f %.6f %.6f %.6f\n", trans(1, 0), trans(1, 1),
						trans(1, 2), trans(1, 3));
				PrintInfo("\t%.6f %.6f %.6f %.6f\n", trans(2, 0), trans(2, 1),
						trans(2, 2), trans(2, 3));
				PrintInfo("\t%.6f %.6f %.6f %.6f\n", trans(3, 0), trans(3, 1),
						trans(3, 2), trans(3, 3));
				*source_trans_ptr = *source_ptr;
				source_trans_ptr->Transform(trans);
				PrintInfo("Error is %.4f after alignment.\n",
						p2p.ComputeError(*source_trans_ptr, *target_ptr,
						corres));
				vis.ResetViewPoint(true);
				return true;
			});
	while (vis_source.PollEvents() && vis_target.PollEvents() &&
			vis_main.PollEvents()) {
	}
	vis_source.DestroyWindow();
	vis_target.DestroyWindow();
	vis_main.DestroyWindow();
	return 1;
}
