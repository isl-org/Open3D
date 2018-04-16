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

#include "py3d_core.h"
#include "py3d_core_trampoline.h"

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>
#include <Core/Registration/GlobalOptimizationMethod.h>
#include <Core/Registration/GlobalOptimizationConvergenceCriteria.h>
#include <IO/ClassIO/PoseGraphIO.h>

using namespace three;

template <class GlobalOptimizationMethodBase = GlobalOptimizationMethod>
class PyGlobalOptimizationMethod : public GlobalOptimizationMethodBase
{
public:
	using GlobalOptimizationMethodBase::GlobalOptimizationMethodBase;
	void OptimizePoseGraph(
		PoseGraph &pose_graph,
		const GlobalOptimizationConvergenceCriteria &criteria,
		const GlobalOptimizationOption &option) const override {
		PYBIND11_OVERLOAD_PURE(
				void,
				GlobalOptimizationMethodBase,
				pose_graph, criteria, option);
	}
};

void pybind_globaloptimization(py::module &m)
{
	py::class_<PoseGraphNode, std::shared_ptr<PoseGraphNode>>
			pose_graph_node(m, "PoseGraphNode");
	py::detail::bind_default_constructor<PoseGraphNode>(pose_graph_node);
	py::detail::bind_copy_functions<PoseGraphNode>(pose_graph_node);
	pose_graph_node
		.def_readwrite("pose", &PoseGraphNode::pose_)
		.def(py::init([](Eigen::Matrix4d pose = Eigen::Matrix4d::Identity()) {
			return new PoseGraphNode(pose); }), "pose"_a)
		.def("__repr__", [](const PoseGraphNode &rr) {
			return std::string("PoseGraphNode, access pose to get its current pose.");
	});
	py::bind_vector<std::vector<PoseGraphNode>>(m, "PoseGraphNodeVector");

	py::class_<PoseGraphEdge, std::shared_ptr<PoseGraphEdge>>
			pose_graph_edge(m, "PoseGraphEdge");
	py::detail::bind_default_constructor<PoseGraphEdge>(pose_graph_edge);
	py::detail::bind_copy_functions<PoseGraphEdge>(pose_graph_edge);
	pose_graph_edge
		.def_readwrite("source_node_id", &PoseGraphEdge::source_node_id_)
		.def_readwrite("target_node_id", &PoseGraphEdge::target_node_id_)
		.def_readwrite("transformation", &PoseGraphEdge::transformation_)
		.def_readwrite("information", &PoseGraphEdge::information_)
		.def_readwrite("uncertain", &PoseGraphEdge::uncertain_)
		.def_readwrite("confidence", &PoseGraphEdge::confidence_)
		.def(py::init([](int source_node_id, int target_node_id,
				Eigen::Matrix4d transformation, Eigen::Matrix6d information,
				bool uncertain, double confidence) {
			return new PoseGraphEdge(source_node_id, target_node_id,
					transformation, information, uncertain, confidence);
		}), "source_node_id"_a = -1, "target_node_id"_a = -1,
				"transformation"_a = Eigen::Matrix4d::Identity(),
				"information"_a = Eigen::Matrix6d::Identity(),
				"uncertain"_a = false,
				"confidence"_a = 1.0)
		.def("__repr__", [](const PoseGraphEdge &rr) {
			return std::string("PoseGraphEdge from nodes ") +
					std::to_string(rr.source_node_id_) + std::string(" to ") +
					std::to_string(rr.target_node_id_) +
					std::string(", access transformation to get relative transformation");
	});
	py::bind_vector<std::vector<PoseGraphEdge>>(m, "PoseGraphEdgeVector");

	py::class_<PoseGraph, std::shared_ptr<PoseGraph>> pose_graph(m, "PoseGraph");
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
			std::string(" edges.");
	});

	py::class_<GlobalOptimizationMethod,
			PyGlobalOptimizationMethod<GlobalOptimizationMethod>>
			global_optimization_method(m, "GlobalOptimizationMethod");
	global_optimization_method
			.def("OptimizePoseGraph",
			&GlobalOptimizationMethod::OptimizePoseGraph);

	py::class_<GlobalOptimizationLevenbergMarquardt,
			PyGlobalOptimizationMethod<GlobalOptimizationLevenbergMarquardt>,
			GlobalOptimizationMethod> global_optimization_method_lm(m,
			"GlobalOptimizationLevenbergMarquardt");
	py::detail::bind_default_constructor<GlobalOptimizationLevenbergMarquardt>
			(global_optimization_method_lm);
	py::detail::bind_copy_functions<GlobalOptimizationLevenbergMarquardt>(
			global_optimization_method_lm);
	global_optimization_method_lm
			.def("__repr__", [](const GlobalOptimizationLevenbergMarquardt &te) {
			return std::string("GlobalOptimizationLevenbergMarquardt");
	});

	py::class_<GlobalOptimizationGaussNewton,
			PyGlobalOptimizationMethod<GlobalOptimizationGaussNewton>,
			GlobalOptimizationMethod> global_optimization_method_gn(m,
				"GlobalOptimizationGaussNewton");
	py::detail::bind_default_constructor<GlobalOptimizationGaussNewton>
			(global_optimization_method_gn);
	py::detail::bind_copy_functions<GlobalOptimizationGaussNewton>(
			global_optimization_method_gn);
	global_optimization_method_gn
			.def("__repr__", [](const GlobalOptimizationGaussNewton &te) {
			return std::string("GlobalOptimizationGaussNewton");
	});

	py::class_<GlobalOptimizationConvergenceCriteria> criteria
			(m, "GlobalOptimizationConvergenceCriteria");
	py::detail::bind_default_constructor
			<GlobalOptimizationConvergenceCriteria>(criteria);
	py::detail::bind_copy_functions
			<GlobalOptimizationConvergenceCriteria>(criteria);
	criteria
		.def_readwrite("max_iteration",
				&GlobalOptimizationConvergenceCriteria::max_iteration_)
		.def_readwrite("min_relative_increment",
				&GlobalOptimizationConvergenceCriteria::min_relative_increment_)
		.def_readwrite("min_relative_residual_increment",
				&GlobalOptimizationConvergenceCriteria::
				min_relative_residual_increment_)
		.def_readwrite("min_right_term",
				&GlobalOptimizationConvergenceCriteria::min_right_term_)
		.def_readwrite("min_residual",
				&GlobalOptimizationConvergenceCriteria::min_residual_)
		.def_readwrite("max_iteration_lm",
				&GlobalOptimizationConvergenceCriteria::max_iteration_lm_)
		.def_readwrite("upper_scale_factor",
				&GlobalOptimizationConvergenceCriteria::upper_scale_factor_)
		.def_readwrite("lower_scale_factor",
				&GlobalOptimizationConvergenceCriteria::lower_scale_factor_)
		.def("__repr__", [](const GlobalOptimizationConvergenceCriteria &cr) {
		return std::string("GlobalOptimizationConvergenceCriteria") +
			std::string("\n> max_iteration : ") +
			std::to_string(cr.max_iteration_) +
			std::string("\n> min_relative_increment : ") +
			std::to_string(cr.min_relative_increment_) +
			std::string("\n> min_relative_residual_increment : ") +
			std::to_string(cr.min_relative_residual_increment_) +
			std::string("\n> min_right_term : ") +
			std::to_string(cr.min_right_term_) +
			std::string("\n> min_residual : ") +
			std::to_string(cr.min_residual_) +
			std::string("\n> max_iteration_lm : ") +
			std::to_string(cr.max_iteration_lm_) +
			std::string("\n> upper_scale_factor : ") +
			std::to_string(cr.upper_scale_factor_) +
			std::string("\n> lower_scale_factor : ") +
			std::to_string(cr.lower_scale_factor_);
	});

	py::class_<GlobalOptimizationOption> option
			(m, "GlobalOptimizationOption");
	py::detail::bind_default_constructor
			<GlobalOptimizationOption>(option);
	py::detail::bind_copy_functions
			<GlobalOptimizationOption>(option);
	option
		.def_readwrite("max_correspondence_distance",
				&GlobalOptimizationOption::
				max_correspondence_distance_)
		.def_readwrite("edge_prune_threshold",
				&GlobalOptimizationOption::edge_prune_threshold_)
		.def_readwrite("reference_node",
				&GlobalOptimizationOption::reference_node_)
		.def(py::init([](double max_correspondence_distance,
				double edge_prune_threshold, int reference_node) {
			return new GlobalOptimizationOption(max_correspondence_distance,
				edge_prune_threshold, reference_node);
		}), "max_correspondence_distance"_a = 0.03,
				"edge_prune_threshold"_a = 0.25, "reference_node"_a = -1)
		.def("__repr__", [](const GlobalOptimizationOption &goo) {
			return std::string("GlobalOptimizationOption") +
				std::string("\n> max_correspondence_distance : ") +
				std::to_string(goo.max_correspondence_distance_) +
				std::string("\n> edge_prune_threshold : ") +
				std::to_string(goo.edge_prune_threshold_) +
				std::string("\n> reference_node : ") +
				std::to_string(goo.reference_node_);
	});
}

void pybind_globaloptimization_methods(py::module &m)
{
	m.def("read_pose_graph", [](const std::string &filename) {
			PoseGraph pose_graph;
			ReadPoseGraph(filename, pose_graph);
			return pose_graph;
			}, "Function to read PoseGraph from file", "filename"_a);
	m.def("write_pose_graph", [](const std::string &filename,
			const PoseGraph pose_graph) {
			WritePoseGraph(filename, pose_graph);
			}, "Function to write PoseGraph to file",
					"filename"_a, "pose_graph"_a);
	m.def("global_optimization", [](PoseGraph &pose_graph,
			const GlobalOptimizationMethod &method,
			const GlobalOptimizationConvergenceCriteria &criteria,
			const GlobalOptimizationOption &option) {
			GlobalOptimization(
					pose_graph, method, criteria, option);
			}, "Function to optimize PoseGraph", "pose_graph"_a, "method"_a,
					"criteria"_a, "option"_a);
}
