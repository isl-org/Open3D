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

#include "Python/registration/registration.h"

#include <Open3D/Registration/PoseGraph.h>
#include <Open3D/Registration/GlobalOptimization.h>
#include <Open3D/Registration/GlobalOptimizationMethod.h>
#include <Open3D/Registration/GlobalOptimizationConvergenceCriteria.h>

using namespace open3d;

template <class GlobalOptimizationMethodBase =
                  registration::GlobalOptimizationMethod>
class PyGlobalOptimizationMethod : public GlobalOptimizationMethodBase {
public:
    using GlobalOptimizationMethodBase::GlobalOptimizationMethodBase;
    void OptimizePoseGraph(
            registration::PoseGraph &pose_graph,
            const registration::GlobalOptimizationConvergenceCriteria &criteria,
            const registration::GlobalOptimizationOption &option)
            const override {
        PYBIND11_OVERLOAD_PURE(void, GlobalOptimizationMethodBase, pose_graph,
                               criteria, option);
    }
};

void pybind_global_optimization(py::module &m) {
    py::class_<registration::PoseGraphNode,
               std::shared_ptr<registration::PoseGraphNode>>
            pose_graph_node(m, "PoseGraphNode", "PoseGraphNode");
    py::detail::bind_default_constructor<registration::PoseGraphNode>(
            pose_graph_node);
    py::detail::bind_copy_functions<registration::PoseGraphNode>(
            pose_graph_node);
    pose_graph_node.def_readwrite("pose", &registration::PoseGraphNode::pose_)
            .def(py::init([](Eigen::Matrix4d pose =
                                     Eigen::Matrix4d::Identity()) {
                     return new registration::PoseGraphNode(pose);
                 }),
                 "pose"_a)
            .def("__repr__", [](const registration::PoseGraphNode &rr) {
                return std::string(
                        "registration::PoseGraphNode, access pose to get its "
                        "current pose.");
            });
    py::bind_vector<std::vector<registration::PoseGraphNode>>(
            m, "PoseGraphNodeVector");

    py::class_<registration::PoseGraphEdge,
               std::shared_ptr<registration::PoseGraphEdge>>
            pose_graph_edge(m, "PoseGraphEdge", "PoseGraphEdge");
    py::detail::bind_default_constructor<registration::PoseGraphEdge>(
            pose_graph_edge);
    py::detail::bind_copy_functions<registration::PoseGraphEdge>(
            pose_graph_edge);
    pose_graph_edge
            .def_readwrite("source_node_id",
                           &registration::PoseGraphEdge::source_node_id_)
            .def_readwrite("target_node_id",
                           &registration::PoseGraphEdge::target_node_id_)
            .def_readwrite("transformation",
                           &registration::PoseGraphEdge::transformation_)
            .def_readwrite("information",
                           &registration::PoseGraphEdge::information_)
            .def_readwrite("uncertain",
                           &registration::PoseGraphEdge::uncertain_)
            .def_readwrite("confidence",
                           &registration::PoseGraphEdge::confidence_)
            .def(py::init([](int source_node_id, int target_node_id,
                             Eigen::Matrix4d transformation,
                             Eigen::Matrix6d information, bool uncertain,
                             double confidence) {
                     return new registration::PoseGraphEdge(
                             source_node_id, target_node_id, transformation,
                             information, uncertain, confidence);
                 }),
                 "source_node_id"_a = -1, "target_node_id"_a = -1,
                 "transformation"_a = Eigen::Matrix4d::Identity(),
                 "information"_a = Eigen::Matrix6d::Identity(),
                 "uncertain"_a = false, "confidence"_a = 1.0)
            .def("__repr__", [](const registration::PoseGraphEdge &rr) {
                return std::string("registration::PoseGraphEdge from nodes ") +
                       std::to_string(rr.source_node_id_) +
                       std::string(" to ") +
                       std::to_string(rr.target_node_id_) +
                       std::string(
                               ", access transformation to get relative "
                               "transformation");
            });
    py::bind_vector<std::vector<registration::PoseGraphEdge>>(
            m, "PoseGraphEdgeVector");

    py::class_<registration::PoseGraph,
               std::shared_ptr<registration::PoseGraph>>
            pose_graph(m, "PoseGraph");
    py::detail::bind_default_constructor<registration::PoseGraph>(pose_graph);
    py::detail::bind_copy_functions<registration::PoseGraph>(pose_graph);
    pose_graph.def_readwrite("nodes", &registration::PoseGraph::nodes_)
            .def_readwrite("edges", &registration::PoseGraph::edges_)
            .def("__repr__", [](const registration::PoseGraph &rr) {
                return std::string("registration::PoseGraph with ") +
                       std::to_string(rr.nodes_.size()) +
                       std::string(" nodes and ") +
                       std::to_string(rr.edges_.size()) +
                       std::string(" edges.");
            });

    py::class_<
            registration::GlobalOptimizationMethod,
            PyGlobalOptimizationMethod<registration::GlobalOptimizationMethod>>
            global_optimization_method(m, "GlobalOptimizationMethod",
                                       "GlobalOptimizationMethod");
    global_optimization_method.def(
            "OptimizePoseGraph",
            &registration::GlobalOptimizationMethod::OptimizePoseGraph);

    py::class_<registration::GlobalOptimizationLevenbergMarquardt,
               PyGlobalOptimizationMethod<
                       registration::GlobalOptimizationLevenbergMarquardt>,
               registration::GlobalOptimizationMethod>
            global_optimization_method_lm(
                    m, "GlobalOptimizationLevenbergMarquardt",
                    "GlobalOptimizationLevenbergMarquardt");
    py::detail::bind_default_constructor<
            registration::GlobalOptimizationLevenbergMarquardt>(
            global_optimization_method_lm);
    py::detail::bind_copy_functions<
            registration::GlobalOptimizationLevenbergMarquardt>(
            global_optimization_method_lm);
    global_optimization_method_lm.def(
            "__repr__",
            [](const registration::GlobalOptimizationLevenbergMarquardt &te) {
                return std::string("GlobalOptimizationLevenbergMarquardt");
            });

    py::class_<registration::GlobalOptimizationGaussNewton,
               PyGlobalOptimizationMethod<
                       registration::GlobalOptimizationGaussNewton>,
               registration::GlobalOptimizationMethod>
            global_optimization_method_gn(m, "GlobalOptimizationGaussNewton",
                                          "GlobalOptimizationGaussNewton");
    py::detail::bind_default_constructor<
            registration::GlobalOptimizationGaussNewton>(
            global_optimization_method_gn);
    py::detail::bind_copy_functions<
            registration::GlobalOptimizationGaussNewton>(
            global_optimization_method_gn);
    global_optimization_method_gn.def(
            "__repr__",
            [](const registration::GlobalOptimizationGaussNewton &te) {
                return std::string("GlobalOptimizationGaussNewton");
            });

    py::class_<registration::GlobalOptimizationConvergenceCriteria> criteria(
            m, "GlobalOptimizationConvergenceCriteria",
            "GlobalOptimizationConvergenceCriteria");
    py::detail::bind_default_constructor<
            registration::GlobalOptimizationConvergenceCriteria>(criteria);
    py::detail::bind_copy_functions<
            registration::GlobalOptimizationConvergenceCriteria>(criteria);
    criteria.def_readwrite(
                    "max_iteration",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            max_iteration_)
            .def_readwrite(
                    "min_relative_increment",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            min_relative_increment_)
            .def_readwrite(
                    "min_relative_residual_increment",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            min_relative_residual_increment_)
            .def_readwrite(
                    "min_right_term",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            min_right_term_)
            .def_readwrite(
                    "min_residual",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            min_residual_)
            .def_readwrite(
                    "max_iteration_lm",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            max_iteration_lm_)
            .def_readwrite(
                    "upper_scale_factor",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            upper_scale_factor_)
            .def_readwrite(
                    "lower_scale_factor",
                    &registration::GlobalOptimizationConvergenceCriteria::
                            lower_scale_factor_)
            .def("__repr__",
                 [](const registration::GlobalOptimizationConvergenceCriteria
                            &cr) {
                     return std::string(
                                    "GlobalOptimizationConvergenceCriteria") +
                            std::string("\n> max_iteration : ") +
                            std::to_string(cr.max_iteration_) +
                            std::string("\n> min_relative_increment : ") +
                            std::to_string(cr.min_relative_increment_) +
                            std::string(
                                    "\n> min_relative_residual_increment : ") +
                            std::to_string(
                                    cr.min_relative_residual_increment_) +
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

    py::class_<registration::GlobalOptimizationOption> option(
            m, "GlobalOptimizationOption", "GlobalOptimizationOption");
    py::detail::bind_default_constructor<
            registration::GlobalOptimizationOption>(option);
    py::detail::bind_copy_functions<registration::GlobalOptimizationOption>(
            option);
    option.def_readwrite("max_correspondence_distance",
                         &registration::GlobalOptimizationOption::
                                 max_correspondence_distance_)
            .def_readwrite("edge_prune_threshold",
                           &registration::GlobalOptimizationOption::
                                   edge_prune_threshold_)
            .def_readwrite("preference_loop_closure",
                           &registration::GlobalOptimizationOption::
                                   preference_loop_closure_)
            .def_readwrite(
                    "reference_node",
                    &registration::GlobalOptimizationOption::reference_node_)
            .def(py::init([](double max_correspondence_distance,
                             double edge_prune_threshold,
                             double preference_loop_closure,
                             int reference_node) {
                     return new registration::GlobalOptimizationOption(
                             max_correspondence_distance, edge_prune_threshold,
                             preference_loop_closure, reference_node);
                 }),
                 "max_correspondence_distance"_a = 0.03,
                 "edge_prune_threshold"_a = 0.25,
                 "preference_loop_closure"_a = 1.0, "reference_node"_a = -1)
            .def("__repr__",
                 [](const registration::GlobalOptimizationOption &goo) {
                     return std::string("GlobalOptimizationOption") +
                            std::string("\n> max_correspondence_distance : ") +
                            std::to_string(goo.max_correspondence_distance_) +
                            std::string("\n> edge_prune_threshold : ") +
                            std::to_string(goo.edge_prune_threshold_) +
                            std::string("\n> preference_loop_closure : ") +
                            std::to_string(goo.preference_loop_closure_) +
                            std::string("\n> reference_node : ") +
                            std::to_string(goo.reference_node_);
                 });
}

void pybind_global_optimization_methods(py::module &m) {
    m.def("global_optimization",
          [](registration::PoseGraph &pose_graph,
             const registration::GlobalOptimizationMethod &method,
             const registration::GlobalOptimizationConvergenceCriteria
                     &criteria,
             const registration::GlobalOptimizationOption &option) {
              registration::GlobalOptimization(pose_graph, method, criteria,
                                               option);
          },
          "Function to optimize registration::PoseGraph", "pose_graph"_a,
          "method"_a, "criteria"_a, "option"_a);
}
