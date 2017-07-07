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

#include <Eigen/Dense> // for debugging
#include <Core/Core.h>
#include <IO/IO.h>
#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

using namespace three;

std::shared_ptr<PoseGraph> CustomLoadFromINFO(std::string filename) {
	std::shared_ptr<PoseGraph> output = std::make_shared<PoseGraph>();
	int id1, id2, frame;
	Eigen::Matrix6d info;
	Eigen::Matrix6d info_new;
	FILE * f = fopen(filename.c_str(), "r");
	if (f != NULL) {
		char buffer[1024];
		while (fgets(buffer, 1024, f) != NULL) {
			if (strlen(buffer) > 0 && buffer[0] != '#') {
				sscanf(buffer, "%d %d %d", &id1, &id2, &frame);
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(0, 0), &info(0, 1), &info(0, 2), &info(0, 3), &info(0, 4), &info(0, 5));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(1, 0), &info(1, 1), &info(1, 2), &info(1, 3), &info(1, 4), &info(1, 5));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(2, 0), &info(2, 1), &info(2, 2), &info(2, 3), &info(2, 4), &info(2, 5));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(3, 0), &info(3, 1), &info(3, 2), &info(3, 3), &info(3, 4), &info(3, 5));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(4, 0), &info(4, 1), &info(4, 2), &info(4, 3), &info(4, 4), &info(4, 5));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf %lf %lf", &info(5, 0), &info(5, 1), &info(5, 2), &info(5, 3), &info(5, 4), &info(5, 5));
				PoseGraphEdge new_edge;
				new_edge.source_node_id_ = id2;
				new_edge.target_node_id_ = id1;				
				info_new.block<3, 3>(3, 3) = info.block<3, 3>(0, 0);
				info_new.block<3, 3>(3, 0) = info.block<3, 3>(0, 3);
				info_new.block<3, 3>(0, 3) = info.block<3, 3>(3, 0);
				info_new.block<3, 3>(0, 0) = info.block<3, 3>(3, 3);
				new_edge.information_ = info_new;
				output->edges_.push_back(new_edge);
			}
		}
		fclose(f);
	}
	PrintDebug("%s output->edges_.size() : %d\n", filename.c_str(), output->edges_.size());
	return output;
}

std::shared_ptr<PoseGraph> CustomLoadFromLOG(std::string filename) {
	std::shared_ptr<PoseGraph> output = std::make_shared<PoseGraph>();
	int id1, id2, frame;	
	Eigen::Matrix4d trans;
	FILE * f = fopen(filename.c_str(), "r");
	if (f != NULL) {
		char buffer[1024];
		while (fgets(buffer, 1024, f) != NULL) {
			if (strlen(buffer) > 0 && buffer[0] != '#') {
				sscanf(buffer, "%d %d %d", &id1, &id2, &frame);
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2), &trans(0, 3));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2), &trans(1, 3));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2), &trans(2, 3));
				fgets(buffer, 1024, f);
				sscanf(buffer, "%lf %lf %lf %lf", &trans(3, 0), &trans(3, 1), &trans(3, 2), &trans(3, 3));
				PoseGraphEdge new_edge;
				new_edge.source_node_id_ = id2;
				new_edge.target_node_id_ = id1;
				new_edge.transformation_ = trans;
				output->edges_.push_back(new_edge);
			}
		}
		fclose(f);
	}
	return output;
}

std::shared_ptr<PoseGraph> MergeGraph(
		const PoseGraph &odo_log, const PoseGraph &odo_info,
		const PoseGraph &loop_log, const PoseGraph &loop_info )
{
	std::shared_ptr<PoseGraph> output = std::make_shared<PoseGraph>();
	for (int i = 0; i < odo_log.edges_.size() + 1; i++)
	{
		PoseGraphNode new_odo_node;
		if (i == 0)
			new_odo_node.pose_ = Eigen::Matrix4d::Identity();
		else
			new_odo_node.pose_ = output->nodes_[i - 1].pose_ * 
				odo_log.edges_[i - 1].transformation_;
		output->nodes_.push_back(new_odo_node);		
	}
	for (int i = 0; i < odo_log.edges_.size(); i++)
	{
		PoseGraphEdge new_odo_edge;
		new_odo_edge.source_node_id_ = odo_log.edges_[i].source_node_id_;
		new_odo_edge.target_node_id_ = odo_log.edges_[i].target_node_id_;
		new_odo_edge.transformation_ = odo_log.edges_[i].transformation_;
		new_odo_edge.information_ = odo_info.edges_[i].information_;
		output->edges_.push_back(new_odo_edge);
	}	
	for (int i = 0; i < loop_log.edges_.size(); i++)
	{
		PoseGraphEdge new_loop_edge;
		new_loop_edge.source_node_id_ = loop_log.edges_[i].source_node_id_;
		new_loop_edge.target_node_id_ = loop_log.edges_[i].target_node_id_;
		new_loop_edge.transformation_ = loop_log.edges_[i].transformation_;
		new_loop_edge.information_ = loop_info.edges_[i].information_;
		output->edges_.push_back(new_loop_edge);
	}
	// check broken edge
	std::vector<bool> edge_check;
	int n_edges = output->nodes_.size();
	edge_check.resize(n_edges);
	for (int i = 0; i < n_edges; i++)
		edge_check[i] = false;
	for (int i = 0; i < n_edges; i++) {
		edge_check[output->edges_[i].source_node_id_] = true;
		edge_check[output->edges_[i].target_node_id_] = true;
	}		
	for (int i = 0; i < n_edges; i++)
		if (!edge_check[i])
			PrintDebug("Error: edge for node %d is missing\n", i);
	return output;
}

std::shared_ptr<PoseGraph> LoadOldPoseGraph(
		const std::string &odo_log, const std::string &odo_info, 
		const std::string &loop_log, const std::string &loop_info)
{
	auto posegraph_odo_log = CustomLoadFromLOG(odo_log);
	auto posegraph_odo_info = CustomLoadFromINFO(odo_info);
	auto posegraph_loop_log = CustomLoadFromLOG(loop_log);
	auto posegraph_loop_info = CustomLoadFromINFO(loop_info);
	auto posegraph_merged = MergeGraph(
			*posegraph_odo_log, *posegraph_odo_info,
			*posegraph_loop_log, *posegraph_loop_info);
	return posegraph_merged;
}

int main(int argc, char **argv)
{
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

	////////////////////////
	//// test posegraph read and write
	//PoseGraph new_pose_graph;
	//new_pose_graph.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	//new_pose_graph.nodes_.push_back(PoseGraphNode(Eigen::Matrix4d::Random()));
	//new_pose_graph.edges_.push_back(PoseGraphEdge(0, 1, 
	//		Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false));
	//new_pose_graph.edges_.push_back(PoseGraphEdge(0, 2,
	//		Eigen::Matrix4d::Random(), Eigen::Matrix6d::Random(), false));
	//WritePoseGraph("test_pose_graph.json", new_pose_graph);
	//PoseGraph pose_graph;
	//ReadPoseGraph("test_pose_graph.json", pose_graph);
	//WritePoseGraph("test_pose_graph_copy.json", pose_graph);
	////////////////////////

	////////////////////
	// test posegraph optimization
	//auto old_pose_graph = LoadOldPoseGraph(
	//		"C:/git/Open3D/src/Test/TestData/GraphOptimization/frag_000_odo.log",
	//		"C:/git/Open3D/src/Test/TestData/GraphOptimization/frag_000_odo.info",
	//		"C:/git/Open3D/src/Test/TestData/GraphOptimization/frag_000_loop.log",
	//		"C:/git/Open3D/src/Test/TestData/GraphOptimization/frag_000_loop.info");
	auto old_pose_graph = LoadOldPoseGraph(
			"C:/git/Open3D/src/Test/TestData/GraphOptimization/odometry.log",
			"C:/git/Open3D/src/Test/TestData/GraphOptimization/odometry.info",
			"C:/git/Open3D/src/Test/TestData/GraphOptimization/result.txt",
			"C:/git/Open3D/src/Test/TestData/GraphOptimization/result.info");
	WritePoseGraph("test_pose_graph_old.json", *old_pose_graph);
	auto pose_graph = CreatePoseGraphFromFile(
		"C:/git/Open3D/build/bin/Test/Release/test_pose_graph_old.json");
	auto pose_graph_optimized = GlobalOptimization(*pose_graph);
	////////////////////



	//// 6d to 4x4
	//Eigen::Matrix4d M;
	//M << 0.9999888060, 0.0013563634, 0.0045330144, -0.0015367448,
	//	-0.0013477337, 0.9999972749, -0.0019062503, 0.0015274951,
	//	-0.0045355876, 0.0019001196, 0.9999879089, -0.0007733108,
	//	0.0000000000, -0.0000000000, -0.0000000000, 1.0000000000;
	//std::cout << M << std::endl;
	//Eigen::Vector6d m = TransformMatrix4dToVector6d(M);
	//std::cout << m << std::endl;
	//Eigen::Matrix4d M2 = TransformVector6dToMatrix4d(m);
	//std::cout << M2 << std::endl;

	//int edge_id = 99;
	//int want_to_see = 1;
	//for (int id = edge_id; id < edge_id + want_to_see; id++) {
	//	Eigen::Matrix6d TestInfo = pose_graph->edges_[id].information_;
	//	std::cout << pose_graph->edges_[id].source_node_id_ << ", " <<
	//		pose_graph->edges_[id].target_node_id_ << std::endl;
	//	std::cout << TestInfo << std::endl;
	//}
	//int src_id = 0;
	//int tgt_id = 2;
	//Eigen::Matrix4d M = pose_graph->nodes_[tgt_id].pose_.inverse() * 
	//		pose_graph->nodes_[src_id].pose_;
	//Eigen::Vector6d m = TransformMatrix4dToVector6d(M);
	//Eigen::Matrix6d TestInfo = pose_graph->edges_[edge_id].information_;
	//double residual = m.transpose() * TestInfo * m;
	//std::cout << "src : " << src_id << ", tgt : " << tgt_id <<
	//	", residual is : " << residual << std::endl;

	///// why the errors are so high?
	//int edge_id = 140;
	//Eigen::Matrix4d M = pose_graph->edges_[edge_id].transformation_.inverse();
	//Eigen::Vector6d m = TransformMatrix4dToVector6d(M);
	//Eigen::Matrix6d TestInfo = pose_graph->edges_[edge_id].information_;
	//double residual = m.transpose() * TestInfo * m;
	//std::cout << "residual is : " << residual << std::endl;

	

	return 0;
}
