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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Registration/PoseGraph.h>
#include <IO/ClassIO/PoseGraphIO.h>

using namespace three;

void pybind_globaloptimization(py::module &m)
{
	py::class_<PoseGraphNode> pose_graph_node(m, "PoseGraphNode");
	py::detail::bind_default_constructor<PoseGraphNode>(pose_graph_node);
	py::detail::bind_copy_functions<PoseGraphNode>(pose_graph_node);
	pose_graph_node
		.def_readwrite("pose", &PoseGraphNode::pose_)		
		.def("__init__", [](PoseGraphNode &c,
				Eigen::Matrix4d pose = Eigen::Matrix4d::Identity()) {
				new (&c)PoseGraphNode(pose); }, "pose"_a)
		.def("__repr__", [](const PoseGraphNode &rr) {
			return std::string("PoseGraphNode, access pose to get its current pose.\n");
	});
					
	py::class_<PoseGraphEdge> pose_graph_edge(m, "PoseGraphEdge");
	py::detail::bind_default_constructor<PoseGraphEdge>(pose_graph_edge);
	py::detail::bind_copy_functions<PoseGraphEdge>(pose_graph_edge);
	pose_graph_edge
		.def_readwrite("target_node_id", &PoseGraphEdge::target_node_id_)
		.def_readwrite("source_node_id", &PoseGraphEdge::source_node_id_)
		.def_readwrite("transformation", &PoseGraphEdge::transformation_)
		.def_readwrite("information", &PoseGraphEdge::information_)
		.def_readwrite("uncertain", &PoseGraphEdge::uncertain_)
		.def_readwrite("confidence", &PoseGraphEdge::confidence_)
		.def("__init__", [](PoseGraphEdge &c,
				int target_node_id, int source_node_id,
				Eigen::Matrix4d transformation, Eigen::Matrix6d information,
				bool uncertain,
				double confidence) {
				new (&c)PoseGraphEdge(target_node_id, source_node_id, 
				transformation, information, uncertain, confidence); },
				"target_node_id"_a = -1, "source_node_id"_a = -1,
				"transformation"_a = Eigen::Matrix4d::Identity(), 
				"information"_a = Eigen::Matrix6d::Identity(), 
				"uncertain"_a = false,
				"confidence"_a = 1.0)
		.def("__repr__", [](const PoseGraphEdge &rr) {
			return std::string("PoseGraphEdge from nodes %d to %d, access transformation to get relative transformation\n", 
					rr.source_node_id_, rr.target_node_id_);
	});

	py::class_<PoseGraph> pose_graph(m, "PoseGraph");
	py::detail::bind_default_constructor<PoseGraph>(pose_graph);
	py::detail::bind_copy_functions<PoseGraph>(pose_graph);
	pose_graph
		.def_readwrite("nodes", &PoseGraph::nodes_)
		.def_readwrite("edges", &PoseGraph::edges_)
		.def("__repr__", [](const PoseGraph &rr) {
		return std::string("PoseGraph with ") +
			std::to_string(rr.nodes_.size()) +
			std::string(" nodes and ") +
			std::to_string(rr.edges_.size()) +
			std::string(" edges.\n");
	});
}

void pybind_globaloptimization_methods(py::module &m)
{
	m.def("ReadPoseGraph", [](const std::string &filename) {
			PoseGraph pose_graph;
			ReadPoseGraph(filename, pose_graph);
			return pose_graph;
			}, "Function to read PoseGraph from file", "filename"_a);
	m.def("WritePoseGraph", [](const std::string &filename, 
			const PoseGraph pose_graph) {
			WritePoseGraph(filename, pose_graph);
			}, "Function to write PoseGraph to file", "filename"_a, "pose_graph"_a);
}
