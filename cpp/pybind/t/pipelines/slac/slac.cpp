// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/pipelines/slac/slac.h"

#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/pipelines/slac/ControlGrid.h"
#include "open3d/t/pipelines/slac/SLACOptimizer.h"
#include "open3d/utility/Logging.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

void pybind_slac_classes(py::module &m) {
    py::class_<SLACOptimizerParams> slac_optimizer_params(
            m, "slac_optimizer_params",
            "SLAC parameters to tune in optimization.");
    py::detail::bind_copy_functions<SLACOptimizerParams>(slac_optimizer_params);
    slac_optimizer_params
            .def(py::init<const int, const float, const float, const float,
                          const float, const core::Device, const std::string>(),
                 "max_iterations"_a = 5, "voxel_size"_a = 0.05,
                 "distance_threshold"_a = 0.07, "fitness_threshold"_a = 0.3,
                 "regularizer_weight"_a = 1, "device"_a = core::Device("CPU:0"),
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
            .def_readwrite("regularizer_weight",
                           &SLACOptimizerParams::regularizer_weight_,
                           "Weight of the regularizer.")
            .def_readwrite("device", &SLACOptimizerParams::device_,
                           "Device to use.")
            .def_readwrite("slac_folder", &SLACOptimizerParams::slac_folder_,
                           "Relative directory to store SLAC results in the "
                           "dataset folder.")
            .def(
                    "get_subfolder_name",
                    [](const SLACOptimizerParams &slac_optimizer_params) {
                        return slac_optimizer_params.GetSubfolderName();
                    },
                    "Relative directory to store SLAC results in the dataset "
                    "folder.")
            .def("__repr__", [](const SLACOptimizerParams &params) {
                return fmt::format(
                        "SLACOptimizerParams[max_iterations={:d}, "
                        "voxel_size={:e}, distance_threshold={:e}, "
                        "fitness_threshold={:e}, regularizer_weight={:e}, "
                        "device={}, slac_folder={}].",
                        params.max_iterations_, params.voxel_size_,
                        params.distance_threshold_, params.fitness_threshold_,
                        params.regularizer_weight_, params.device_.ToString(),
                        params.slac_folder_);
            });

    py::class_<SLACDebugOption> slac_debug_option(m, "slac_debug_option",
                                                  "SLAC debug options.");
    py::detail::bind_copy_functions<SLACDebugOption>(slac_debug_option);
    slac_debug_option
            .def(py::init<const bool, const int>(), "debug"_a = false,
                 "debug_start_node_idx"_a = 0)
            .def(py::init<const int>(), "debug_start_node_idx"_a)
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

    py::class_<ControlGrid> control_grid(
            m, "control_grid",
            " ControlGrid is a spatially hashed voxel grid used for non-rigid "
            "point cloud registration and TSDF integration. Each grid stores a "
            "map from the initial grid location to the deformed location. You "
            "can imagine a control grid as a jelly that is warped upon "
            "perturbation with its overall shape preserved. "
            "Reference: "
            "https://github.com/qianyizh/ElasticReconstruction/blob/master/"
            "FragmentOptimizer/OptApp.cpp "
            "http://vladlen.info/papers/elastic-fragments.pdf. ");
    py::detail::bind_copy_functions<ControlGrid>(control_grid);
    control_grid.def(py::init<>())
            .def(py::init<float, int64_t, const core::Device>(), "grid_size"_a,
                 "grid_count"_a = 1000, "device"_a = core::Device("CPU:0"))
            .def(py::init<float, core::Tensor, core::Tensor,
                          const core::Device>(),
                 "grid_size"_a, "keys"_a, "values"_a,
                 "device"_a = core::Device("CPU:0"))
            .def(
                    "touch",
                    [](ControlGrid &control_grid, geometry::PointCloud &pcd) {
                        control_grid.Touch(pcd);
                    },
                    "Allocate control grids in the shared camera space.",
                    "pointcloud"_a)
            .def("compactify", &ControlGrid::Compactify,
                 "Force rehashing, so that all entries are remapped to [0, "
                 "size) and form a contiguous index map.")
            .def("get_neighbor_grid_map", &ControlGrid::GetNeighborGridMap,
                 "Get the neighbor indices per grid to construct the "
                 "regularizer. "
                 "Returns a 6-way neighbor grid map for all the active "
                 "entries of shape (N, ). "
                 "\n - buf_indices Active indices in the buffer of shape (N, ) "
                 "\n - buf_indices_nb Neighbor indices (including "
                 "non-allocated "
                 "entries) for the active entries of shape (N, 6). "
                 "\n - masks_nb Corresponding neighbor masks of shape (N, "
                 "6). ")
            .def(
                    "parameterize",
                    [](ControlGrid &control_grid,
                       const geometry::PointCloud &pcd) {
                        return control_grid.Parameterize(pcd);
                    },
                    "Parameterize an input point cloud by embedding each point "
                    "in the grid "
                    "with 8 corners via indexing and interpolation. "
                    "Returns: A PointCloud with parameterization attributes: "
                    "\n- neighbors: Index of 8 neighbor control grid points of "
                    "shape (8, ) in Int64. "
                    "\n- ratios: Interpolation ratios of 8 neighbor control "
                    "grid points of shape (8, ) in Float32.",
                    "pointcloud"_a)
            .def(
                    "deform",
                    [](ControlGrid &control_grid,
                       const geometry::PointCloud &pcd) {
                        return control_grid.Deform(pcd);
                    },
                    "Non-rigidly deform a point cloud using the control grid.",
                    "pointcloud"_a)
            .def(
                    "deform",
                    [](ControlGrid &control_grid, const geometry::Image &depth,
                       const core::Tensor &intrinsics,
                       const core::Tensor &extrinsics, float depth_scale,
                       float depth_max) {
                        return control_grid.Deform(depth, intrinsics,
                                                   extrinsics, depth_scale,
                                                   depth_max);
                    },
                    "Non-rigidly deform a depth image by "
                    "\n- unprojecting the depth image to a point cloud "
                    "\n- deform the point cloud; "
                    "\n- project the deformed point cloud back to the image. ",
                    "depth"_a, "intrinsics"_a, "extrinsics"_a, "depth_scale"_a,
                    "depth_max"_a)
            .def(
                    "deform",
                    [](ControlGrid &control_grid,
                       const geometry::RGBDImage &rgbd,
                       const core::Tensor &intrinsics,
                       const core::Tensor &extrinsics, float depth_scale,
                       float depth_max) {
                        return control_grid.Deform(rgbd, intrinsics, extrinsics,
                                                   depth_scale, depth_max);
                    },
                    "Non-rigidly deform a RGBD image by "
                    "\n- unprojecting the RGBD image to a point cloud "
                    "\n- deform the point cloud; "
                    "\n- project the deformed point cloud back to the image. ",
                    "rgbd"_a, "intrinsics"_a, "extrinsics"_a, "depth_scale"_a,
                    "depth_max"_a)
            .def("get_init_positions", &ControlGrid::GetInitPositions,
                 "Get control grid original positions directly from tensor "
                 "keys.")
            .def("get_curr_positions", &ControlGrid::GetCurrPositions,
                 "Get control grid shifted positions from tensor values "
                 "(optimized in-place)")
            .def(
                    "get_hashmap",
                    [](ControlGrid &control_grid) {
                        return *control_grid.GetHashMap();
                    },
                    "Get the control grid hashmap.")
            .def("size", &ControlGrid::Size)
            .def("get_device", &ControlGrid::GetDevice)
            .def("get_anchor_idx", &ControlGrid::GetAnchorIdx)
            .def("__repr__", [](ControlGrid &control_grid) {
                return fmt::format(
                        "ControlGrid[size={:d}, "
                        "anchor_idx={:d}].",
                        control_grid.Size(), control_grid.GetAnchorIdx());
            });
}

// SLAC functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"fnames_processed",
                 "List of filenames (str) for pre-processed pointcloud "
                 "fragments."},
                {"fragment_filenames",
                 "List of filenames (str) for pointcloud fragments."},
                {"fragment_pose_graph", "PoseGraph for pointcloud fragments"},
                {"params",
                 "slac_optimizer_params Parameters to tune in optimization."},
                {"debug_option", "debug options."}};

void pybind_slac_methods(py::module &m) {
    m.def("save_correspondences_for_pointclouds",
          &SaveCorrespondencesForPointClouds,
          py::call_guard<py::gil_scoped_release>(),
          "Read pose graph containing loop closures and odometry to compute "
          "correspondences. Uses aggressive pruning -- reject any suspicious "
          "pair.",
          "fnames_processed"_a, "fragment_pose_graph"_a,
          "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "save_correspondences_for_pointclouds",
                                 map_shared_argument_docstrings);

    m.def("run_slac_optimizer_for_fragments", &RunSLACOptimizerForFragments,
          "Simultaneous Localization and Calibration: Self-Calibration of "
          "Consumer Depth Cameras, CVPR 2014 Qian-Yi Zhou and Vladlen Koltun "
          "Estimate a shared control grid for all fragments for scene "
          "reconstruction, implemented in "
          "https://github.com/qianyizh/ElasticReconstruction. ",
          "fragment_filenames"_a, "fragment_pose_graph"_a,
          "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "run_slac_optimizer_for_fragments",
                                 map_shared_argument_docstrings);

    m.def("run_rigid_optimizer_for_fragments", &RunRigidOptimizerForFragments,
          "Extended ICP to simultaneously align multiple point clouds with "
          "dense pairwise point-to-plane distances.",
          "fragment_filenames"_a, "fragment_pose_graph"_a,
          "params"_a = SLACOptimizerParams(),
          "debug_option"_a = SLACDebugOption());
    docstring::FunctionDocInject(m, "run_rigid_optimizer_for_fragments",
                                 map_shared_argument_docstrings);
}

void pybind_slac(py::module &m) {
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
