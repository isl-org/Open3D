// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/GlobalOptimization.h"
#include "open3d/pipelines/registration/GlobalOptimizationConvergenceCriteria.h"
#include "open3d/pipelines/registration/GlobalOptimizationMethod.h"
#include "open3d/pipelines/registration/PoseGraph.h"
#include "pybind/docstring.h"
#include "pybind/pipelines/registration/registration.h"

namespace open3d {
namespace pipelines {
namespace registration {

template <class GlobalOptimizationMethodBase = GlobalOptimizationMethod>
class PyGlobalOptimizationMethod : public GlobalOptimizationMethodBase {
public:
    using GlobalOptimizationMethodBase::GlobalOptimizationMethodBase;
    void OptimizePoseGraph(
            PoseGraph &pose_graph,
            const GlobalOptimizationConvergenceCriteria &criteria,
            const GlobalOptimizationOption &option) const override {
        PYBIND11_OVERLOAD_PURE(void, GlobalOptimizationMethodBase, pose_graph,
                               criteria, option);
    }
};

void pybind_global_optimization(py::module &m) {
    // open3d.registration.PoseGraphNode
    py::class_<PoseGraphNode, std::shared_ptr<PoseGraphNode>> pose_graph_node(
            m, "PoseGraphNode", "Node of ``PoseGraph``.");
    py::detail::bind_default_constructor<PoseGraphNode>(pose_graph_node);
    py::detail::bind_copy_functions<PoseGraphNode>(pose_graph_node);
    pose_graph_node.def_readwrite("pose", &PoseGraphNode::pose_)
            .def(py::init([](Eigen::Matrix4d pose =
                                     Eigen::Matrix4d::Identity()) {
                     return new PoseGraphNode(pose);
                 }),
                 "pose"_a)
            .def("__repr__", [](const PoseGraphNode &rr) {
                return std::string(
                        "PoseGraphNode, access "
                        "pose to get its "
                        "current pose.");
            });

    // open3d.registration.PoseGraphNodeVector
    auto pose_graph_node_vector = py::bind_vector<std::vector<PoseGraphNode>>(
            m, "PoseGraphNodeVector");
    pose_graph_node_vector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Vector of PoseGraphNode";
            }),
            py::none(), py::none(), "");

    // open3d.registration.PoseGraphEdge
    py::class_<PoseGraphEdge, std::shared_ptr<PoseGraphEdge>> pose_graph_edge(
            m, "PoseGraphEdge", "Edge of ``PoseGraph``.");
    py::detail::bind_default_constructor<PoseGraphEdge>(pose_graph_edge);
    py::detail::bind_copy_functions<PoseGraphEdge>(pose_graph_edge);
    pose_graph_edge
            .def_readwrite("source_node_id", &PoseGraphEdge::source_node_id_,
                           "int: Source ``PoseGraphNode`` id.")
            .def_readwrite("target_node_id", &PoseGraphEdge::target_node_id_,
                           "int: Target ``PoseGraphNode`` id.")
            .def_readwrite(
                    "transformation", &PoseGraphEdge::transformation_,
                    "``4 x 4`` float64 numpy array: Transformation matrix.")
            .def_readwrite("information", &PoseGraphEdge::information_,
                           "``6 x 6`` float64 numpy array: Information matrix.")
            .def_readwrite("uncertain", &PoseGraphEdge::uncertain_,
                           "bool: Whether the edge is uncertain. Odometry edge "
                           "has uncertain == false, loop closure edges has "
                           "uncertain == true")
            .def_readwrite(
                    "confidence", &PoseGraphEdge::confidence_,
                    "float from 0 to 1: Confidence value of the edge. if "
                    "uncertain is true, it has confidence bounded in [0,1].   "
                    "1 means reliable, and 0 means "
                    "unreliable edge. This correspondence to "
                    "line process value in [Choi et al 2015] See "
                    "core/registration/globaloptimization.h for more details.")
            .def(py::init([](int source_node_id, int target_node_id,
                             Eigen::Matrix4d transformation,
                             Eigen::Matrix6d information, bool uncertain,
                             double confidence) {
                     return new PoseGraphEdge(source_node_id, target_node_id,
                                              transformation, information,
                                              uncertain, confidence);
                 }),
                 "source_node_id"_a = -1, "target_node_id"_a = -1,
                 "transformation"_a = Eigen::Matrix4d::Identity(),
                 "information"_a = Eigen::Matrix6d::Identity(),
                 "uncertain"_a = false, "confidence"_a = 1.0)
            .def("__repr__", [](const PoseGraphEdge &rr) {
                return std::string(
                               "PoseGraphEdge "
                               "from nodes ") +
                       std::to_string(rr.source_node_id_) +
                       std::string(" to ") +
                       std::to_string(rr.target_node_id_) +
                       std::string(
                               ", access transformation to get relative "
                               "transformation");
            });

    // open3d.registration.PoseGraphEdgeVector
    auto pose_graph_edge_vector = py::bind_vector<std::vector<PoseGraphEdge>>(
            m, "PoseGraphEdgeVector");
    pose_graph_edge_vector.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Vector of PoseGraphEdge";
            }),
            py::none(), py::none(), "");

    // open3d.registration.PoseGraph
    py::class_<PoseGraph, std::shared_ptr<PoseGraph>> pose_graph(
            m, "PoseGraph", "Data structure defining the pose graph.");
    py::detail::bind_default_constructor<PoseGraph>(pose_graph);
    py::detail::bind_copy_functions<PoseGraph>(pose_graph);
    pose_graph
            .def_readwrite(
                    "nodes", &PoseGraph::nodes_,
                    "``List(PoseGraphNode)``: List of ``PoseGraphNode``.")
            .def_readwrite(
                    "edges", &PoseGraph::edges_,
                    "``List(PoseGraphEdge)``: List of ``PoseGraphEdge``.")
            .def("__repr__", [](const PoseGraph &rr) {
                return std::string("PoseGraph with ") +
                       std::to_string(rr.nodes_.size()) +
                       std::string(" nodes and ") +
                       std::to_string(rr.edges_.size()) +
                       std::string(" edges.");
            });

    // open3d.registration.GlobalOptimizationMethod
    py::class_<GlobalOptimizationMethod,
               PyGlobalOptimizationMethod<GlobalOptimizationMethod>>
            global_optimization_method(
                    m, "GlobalOptimizationMethod",
                    "Base class for global optimization method.");
    global_optimization_method.def("OptimizePoseGraph",
                                   &GlobalOptimizationMethod::OptimizePoseGraph,
                                   "pose_graph"_a, "criteria"_a, "option"_a,
                                   "Run pose graph optimization.");
    docstring::ClassMethodDocInject(
            m, "GlobalOptimizationMethod", "OptimizePoseGraph",
            {{"pose_graph", "The pose graph to be optimized (in-place)."},
             {"criteria", "Convergence criteria."},
             {"option", "Global optimization options."}});

    py::class_<GlobalOptimizationLevenbergMarquardt,
               PyGlobalOptimizationMethod<GlobalOptimizationLevenbergMarquardt>,
               GlobalOptimizationMethod>
            global_optimization_method_lm(
                    m, "GlobalOptimizationLevenbergMarquardt",
                    "Global optimization with Levenberg-Marquardt algorithm. "
                    "Recommended over the Gauss-Newton method since the LM has "
                    "better convergence characteristics.");
    py::detail::bind_default_constructor<GlobalOptimizationLevenbergMarquardt>(
            global_optimization_method_lm);
    py::detail::bind_copy_functions<GlobalOptimizationLevenbergMarquardt>(
            global_optimization_method_lm);
    global_optimization_method_lm.def(
            "__repr__", [](const GlobalOptimizationLevenbergMarquardt &te) {
                return std::string("GlobalOptimizationLevenbergMarquardt");
            });

    py::class_<GlobalOptimizationGaussNewton,
               PyGlobalOptimizationMethod<GlobalOptimizationGaussNewton>,
               GlobalOptimizationMethod>
            global_optimization_method_gn(
                    m, "GlobalOptimizationGaussNewton",
                    "Global optimization with Gauss-Newton algorithm.");
    py::detail::bind_default_constructor<GlobalOptimizationGaussNewton>(
            global_optimization_method_gn);
    py::detail::bind_copy_functions<GlobalOptimizationGaussNewton>(
            global_optimization_method_gn);
    global_optimization_method_gn.def(
            "__repr__", [](const GlobalOptimizationGaussNewton &te) {
                return std::string("GlobalOptimizationGaussNewton");
            });

    py::class_<GlobalOptimizationConvergenceCriteria> criteria(
            m, "GlobalOptimizationConvergenceCriteria",
            "Convergence criteria of GlobalOptimization.");
    py::detail::bind_default_constructor<GlobalOptimizationConvergenceCriteria>(
            criteria);
    py::detail::bind_copy_functions<GlobalOptimizationConvergenceCriteria>(
            criteria);
    criteria.def_readwrite(
                    "max_iteration",
                    &GlobalOptimizationConvergenceCriteria::max_iteration_,
                    "int: Maximum iteration number for iterative optimization "
                    "module.")
            .def_readwrite("min_relative_increment",
                           &GlobalOptimizationConvergenceCriteria::
                                   min_relative_increment_,
                           "float: Minimum relative increments.")
            .def_readwrite("min_relative_residual_increment",
                           &GlobalOptimizationConvergenceCriteria::
                                   min_relative_residual_increment_,
                           "float: Minimum relative residual increments.")
            .def_readwrite(
                    "min_right_term",
                    &GlobalOptimizationConvergenceCriteria::min_right_term_,
                    "float: Minimum right term value.")
            .def_readwrite(
                    "min_residual",
                    &GlobalOptimizationConvergenceCriteria::min_residual_,
                    "float: Minimum residual value.")
            .def_readwrite(
                    "max_iteration_lm",
                    &GlobalOptimizationConvergenceCriteria::max_iteration_lm_,
                    "int: Maximum iteration number for Levenberg Marquardt "
                    "method. max_iteration_lm is used for additional "
                    "Levenberg-Marquardt inner loop that automatically changes "
                    "steepest gradient gain.")
            .def_readwrite(
                    "upper_scale_factor",
                    &GlobalOptimizationConvergenceCriteria::upper_scale_factor_,
                    "float: Upper scale factor value. Scaling factors are used "
                    "for levenberg marquardt algorithm these are scaling "
                    "factors that increase/decrease lambda used in H_LM = H + "
                    "lambda * I")
            .def_readwrite(
                    "lower_scale_factor",
                    &GlobalOptimizationConvergenceCriteria::lower_scale_factor_,
                    "float: Lower scale factor value.")
            .def("__repr__", [](const GlobalOptimizationConvergenceCriteria
                                        &cr) {
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

    py::class_<GlobalOptimizationOption> option(
            m, "GlobalOptimizationOption", "Option for GlobalOptimization.");
    py::detail::bind_default_constructor<GlobalOptimizationOption>(option);
    py::detail::bind_copy_functions<GlobalOptimizationOption>(option);
    option.def_readwrite(
                  "max_correspondence_distance",
                  &GlobalOptimizationOption::max_correspondence_distance_,
                  "float: Identifies which distance value is used for "
                  "finding neighboring points when making information "
                  "matrix. According to [Choi et al 2015], this "
                  "distance is used for determining $mu, a line process "
                  "weight.")
            .def_readwrite("edge_prune_threshold",
                           &GlobalOptimizationOption::edge_prune_threshold_,
                           "float: According to [Choi et al 2015], "
                           "line_process weight < edge_prune_threshold (0.25) "
                           "is pruned.")
            .def_readwrite("preference_loop_closure",
                           &GlobalOptimizationOption::preference_loop_closure_,
                           "float: Balancing parameter to decide which one is "
                           "more reliable: odometry vs loop-closure. [0,1] -> "
                           "try to unchange odometry edges, [1) -> try to "
                           "utilize loop-closure. Recommendation: 0.1 for RGBD "
                           "Odometry, 2.0 for fragment registration.")
            .def_readwrite("reference_node",
                           &GlobalOptimizationOption::reference_node_,
                           "int: The pose of this node is unchanged after "
                           "optimization.")
            .def(py::init([](double max_correspondence_distance,
                             double edge_prune_threshold,
                             double preference_loop_closure,
                             int reference_node) {
                     return new GlobalOptimizationOption(
                             max_correspondence_distance, edge_prune_threshold,
                             preference_loop_closure, reference_node);
                 }),
                 "max_correspondence_distance"_a = 0.03,
                 "edge_prune_threshold"_a = 0.25,
                 "preference_loop_closure"_a = 1.0, "reference_node"_a = -1)
            .def("__repr__", [](const GlobalOptimizationOption &goo) {
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
    m.def(
            "global_optimization",
            [](PoseGraph &pose_graph, const GlobalOptimizationMethod &method,
               const GlobalOptimizationConvergenceCriteria &criteria,
               const GlobalOptimizationOption &option) {
                GlobalOptimization(pose_graph, method, criteria, option);
            },
            "Function to optimize PoseGraph", "pose_graph"_a, "method"_a,
            "criteria"_a, "option"_a);
    docstring::FunctionDocInject(
            m, "global_optimization",
            {{"pose_graph", "The pose_graph to be optimized (in-place)."},
             {"method",
              "Global optimization method. Either "
              "``GlobalOptimizationGaussNewton()`` or "
              "``GlobalOptimizationLevenbergMarquardt("
              ")``."},
             {"criteria", "Global optimization convergence criteria."},
             {"option", "Global optimization option."}});
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
