// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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
#include <IO/ClassIO/PoseGraphIO.h>
#include <Core/Registration/PoseGraph.h>

#include "Helper.h"

void PrintTest(three::PoseGraph& input) {
	std::cout << "Node" << std::endl;	
	for (size_t i = 0; i < input.nodes_.size(); i++) {
		std::cout << input.nodes_[i].pose_ << std::endl;
	}
	std::cout << "Edge" << std::endl;
	for (size_t i = 0; i < input.edges_.size(); i++) {
		std::cout << input.edges_[i].target_node_id_ << " " <<
				input.edges_[i].source_node_id_ << std::endl;
		std::cout << input.edges_[i].transformation_ << std::endl;
		std::cout << input.edges_[i].information_ << std::endl;
	}
}

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
	// randomly generate PoseGraph
	PoseGraph test;
	test.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	test.edges_.push_back(PoseGraphEdge(1, 2,Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random()));
	test.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	test.edges_.push_back(PoseGraphEdge(0, 2, Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random()));
	WritePoseGraph("test_pose_graph.json", test);
	PrintTest(test);

	// reads PoseGrapy and display
	PoseGraph test_new;
	ReadPoseGraph("test_pose_graph.json", test_new);
	PrintTest(test_new);

	return 1;
}
