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

#include "pybind/t/pipelines/slac/slac.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/slac/SLACOptimizer.h"
#include "open3d/utility/Console.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

void pybind_slac_classes(py::module &m) {
    py::class_<SLACOptimizerParams> slac_optimizer_params(
            m, "slac_optimizer_params", "SLAC optimzation parameters.");
    py::detail::bind_copy_functions<SLACOptimizerParams>(slac_optimizer_params);
    slac_optimizer_params
            .def(py::init<const int, const float, const float, const float,
                          const float, const core::Device, const std::string>(),
                 "max_iterations"_a = 5, "voxel_size"_a = 0.05,
                 "distance_threshold"_a = 0.07, "fitness_threshold"_a = 0.3,
                 "regularizor_weight"_a = 1, "device"_a = core::Device("CPU:0"),
                 "slac_folder"_a = "")
            .def_readwrite("max_iterations",
                           &SLACOptimizerParams::max_iterations_,
                           "Number of iterations.")
            .def_readwrite("voxel_size", &SLACOptimizerParams::voxel_size_,
                           "Voxel size to downsample input point cloud.")
            .def_readwrite("distance_threshold",
                           &SLACOptimizerParams::distance_threshold_,
                           " Distance threshold to filter inconsistent "
                           "correspondences.")
            .def_readwrite("fitness_threshold",
                           &SLACOptimizerParams::fitness_threshold_,
                           "Fitness threshold to filter inconsistent pairs.")
            .def_readwrite("regularizor_weight",
                           &SLACOptimizerParams::regularizor_weight_,
                           "Weight of the regularizor.")
            .def_readwrite("device", &SLACOptimizerParams::device_,
                           "Device to use.")
            .def_readwrite("slac_folder", &SLACOptimizerParams::slac_folder_,
                           "Relative directory to store SLAC results in the "
                           "dataset folder.")
            .def("__repr__", [](const SLACOptimizerParams &params) {
                return fmt::format(
                        "SLACOptimizerParams[max_iterations={:d}, "
                        "voxel_size={:e}, distance_threshold={:e}, "
                        "fitness_threshold={:e}, regularizor_weight={:e}, "
                        "device={}, slac_folder={}].",
                        params.max_iterations_, params.max_iterations_,
                        params.voxel_size_, params.distance_threshold_,
                        params.fitness_threshold_, params.regularizor_weight_,
                        params.device_.ToString(), params.slac_folder_);
            });

    py::class_<SLACDebugOption> slac_debug_option(m, "slac_debug_option",
                                                  "SLAC debug options.");
    py::detail::bind_copy_functions<SLACDebugOption>(slac_debug_option);
    slac_debug_option
            .def(py::init<const bool, const int>(), "debug"_a = false,
                 "debug_start_node_idx"_a = 0)
            .def(py::init<const int>(), "debug_start_node_idx"_a = 0)
            .def_readwrite("debug", &SLACDebugOption::debug_, "Enable debug.")
            .def_readwrite("debug_start_node_idx",
                           &SLACDebugOption::debug_start_node_idx_,
                           "The node id to start debugging with. Smaller nodes "
                           "will be skipped for visualization.")
            .def("__repr__", [](const SLACDebugOption &debug_option) {
                return fmt::format(
                        "SLACDebugOption[debug={}, "
                        "debug_start_node_idx={:d}].",
                        debug_option.debug_,
                        debug_option.debug_start_node_idx_);
            });
}

// Registration functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"fnames_processed",
                 "List of filenames (str) for pre-processed pointcloud "
                 "fragments."},
                {"fragment_filenames",
                 "List of filenames (str) for pointcloud fragments."},
                {"fragment_pose_graph", "PoseGraph for pointcloud fragments"},
                {"params",
                 "slac_optimizer_params containing the configurations."},
                {"debug_option",
                 "slac_debug_option containing the debug options."}};

void pybind_slac_methods(py::module &m) {
    m.def("save_correspondences_for_pointclouds",
          &SaveCorrespondencesForPointClouds,
          "Read pose graph containing loop closures and odometry to compute "
          "correspondences. Uses aggressive pruning -- reject any suspicious "
          "pair.",
          "fnames_processed"_a, "fragment_pose_graph"_a,
          "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "save_correspondences_for_pointclouds",
                                 map_shared_argument_docstrings);

    m.def("run_slac_optimize_for_fragments", &RunSLACOptimizerForFragments,
          "Simultaneous Localization and Calibration: Self-Calibration of "
          "Consumer Depth Cameras, CVPR 2014 Qian-Yi Zhou and Vladlen Koltun "
          "Estimate a shared control grid for all fragments for scene "
          "reconstruction, implemented in "
          "https://github.com/qianyizh/ElasticReconstruction. ",
          "fragment_filenames"_a, "fragment_pose_graph"_a,
          "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "run_slac_optimize_for_fragments",
                                 map_shared_argument_docstrings);

    m.def("run_rigid_optimize_for_fragments", &RunRigidOptimizerForFragments,
          "RunRigidOptimizerForFragments.", "fragment_filenames"_a,
          "fragment_pose_graph"_a, "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "run_rigid_optimize_for_fragments",
                                 map_shared_argument_docstrings);
}

void pybind_registration(py::module &m) {
    py::module m_submodule = m.def_submodule(
            "slac",
            "Tensor-based Simultaneous Localisation and Calibration pipeline.");
    pybind_slac_classes(m_submodule);
    pybind_slac_methods(m_submodule);
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
