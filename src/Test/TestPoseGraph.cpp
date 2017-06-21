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

#include <cstdio>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Core/Registration/PoseGraph.h>

int main(int argc, char **argv)
{
	using namespace three;

	SetVerbosityLevel(three::VERBOSE_ALWAYS);
	
	if (argc != 1) {
		PrintInfo("Usage:\n");
		PrintInfo("    > TestPoseGraph\n");
		PrintInfo("    The program will :\n");
		PrintInfo("    1) Generate random PoseGraph\n");
		PrintInfo("    2) Save random PoseGraph as test_pose_graph.json\n");
		PrintInfo("    3) Reads PoseGraph from test_pose_graph.json\n");
		PrintInfo("    4) Save loaded PoseGraph as test_pose_graph_copy.json\n");
		return 0;
	}

	PoseGraph new_pose_graph;
	new_pose_graph.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	new_pose_graph.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	new_pose_graph.edges_.push_back(PoseGraphEdge(0, 1, 
			Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false));
	new_pose_graph.edges_.push_back(PoseGraphEdge(0, 2,
			Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false));
	WritePoseGraph("test_pose_graph.json", new_pose_graph);
	PoseGraph pose_graph;
	ReadPoseGraph("test_pose_graph.json", pose_graph);
	WritePoseGraph("test_pose_graph_copy.json", pose_graph);

	return 0;
}
