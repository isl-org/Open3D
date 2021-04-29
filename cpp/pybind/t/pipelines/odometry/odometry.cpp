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

#include "pybind/t/pipelines/odometry/odometry.h"

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

void pybind_odometry_classes(py::module &m) {
    py::enum_<Method>(m, "Method", "Tensor odometry esitmation method.")
            .value("PointToPlane", Method::PointToPlane)
            .value("Intensity", Method::Intensity)
            .value("Hybrid", Method::Hybrid)
            .export_values();

    py::class_<OdometryConvergenceCriteria> odometry_convergence_criteria(
            m, "OdometryConvergenceCriteria",
            "Class that defines the convergence criteria of odometry. "
            "Odometry algorithm stops if the relative change of fitness and "
            "rmse "
            "hit ``relative_fitness`` and ``relative_rmse`` individually. ");
    py::detail::bind_copy_functions<OdometryConvergenceCriteria>(
            odometry_convergence_criteria);
    odometry_convergence_criteria
            .def(py::init<double, double>(), "relative_fitness"_a = 1e-6,
                 "relative_rmse"_a = 1e-6)
            .def_readwrite(
                    "relative_fitness",
                    &OdometryConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def_readwrite(
                    "relative_rmse",
                    &OdometryConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inliner RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def("__repr__", [](const OdometryConvergenceCriteria &c) {
                return fmt::format(
                        "ICPConvergenceCriteria class "
                        "with relative_fitness={:e}, relative_rmse={:e}, ",
                        c.relative_fitness_, c.relative_rmse_);
            });

    // open3d.t.pipelines.odometry.OdometryResult
    py::class_<OdometryResult> odometry_result(
            m, "OdometryResult", "Class that contains the odometry results.");
    py::detail::bind_default_constructor<OdometryResult>(odometry_result);
    py::detail::bind_copy_functions<OdometryResult>(odometry_result);
    odometry_result
            .def_readwrite("transformation", &OdometryResult::transformation_,
                           "``4 x 4`` float64 tensor on CPU: The estimated "
                           "transformation matrix.")
            .def_readwrite("inlier_rmse", &OdometryResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &OdometryResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in target). Higher is better.")
            .def("__repr__", [](const OdometryResult &odom_result) {
                return fmt::format(
                        "OdometryResult with "
                        "fitness={:e}"
                        ", inlier_rmse={:e}"
                        "\nAccess transformation to get result.",
                        odom_result.fitness_, odom_result.inlier_rmse_);
            });

    // open3d.t.pipelines.odometry.OdometryLossParams
    py::class_<OdometryLossParams> odometry_loss_params(
            m, "OdometryLossParams",
            "Class that contains the odometry loss parameters.");
    py::detail::bind_default_constructor<OdometryLossParams>(
            odometry_loss_params);
    py::detail::bind_copy_functions<OdometryLossParams>(odometry_loss_params);
    odometry_loss_params
            .def_readwrite("depth_outlier_trunc",
                           &OdometryLossParams::depth_outlier_trunc_, "float.")
            .def_readwrite("depth_huber_delta",
                           &OdometryLossParams::depth_huber_delta_, "float.")
            .def_readwrite("intensity_huber_delta",
                           &OdometryLossParams::intensity_huber_delta_,
                           "float.")
            .def("__repr__", [](const OdometryLossParams &olp) {
                return fmt::format(
                        "OdometryLossParams with "
                        "depth_outlier_trunc={:e}"
                        ", depth_huber_delta={:e}"
                        ", intensity_huber_delta={:e}",
                        olp.depth_outlier_trunc_, olp.depth_huber_delta_,
                        olp.intensity_huber_delta_);
            });
}

// Odometry functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"criteria", "Odometry convergence criteria."},
                {"depth_outlier_trunc",
                 "Depth difference threshold used to filter projective "
                 "associations."},
                {"depth_huber_delta", " "},
                {"depth_scale",
                 "Converts depth pixel values to meters by dividing the scale "
                 "factor."},
                {"init_source_to_target",
                 "(4, 4) initial transformation matrix from source to target."},
                {"intrinsics", "(3, 3) intrinsic matrix for projection."},
                {"intensity_huber_delta", " "},
                {"iterations",
                 "o3d.utility.IntVector Iterations in multiscale "
                 "odometry, from coarse to fine."},
                {"method",
                 "Estimation method used to apply RGBD odometry. "
                 "One of (``PointToPlane``, ``Intensity``, ``Hybrid``)"},
                {"params", "Odometry loss parameters."},
                {"source", "The source RGBD image."},
                {"source_depth_map",
                 "(H, W, 1) Float32 source depth image obtained by "
                 "PreprocessDepth before calling this function."},
                {"source_depth_dx",
                 "(H, W, 1) Float32 source depth gradient image at x-axis "
                 "obtained by FilterSobel before calling this function."},
                {"source_depth_dy",
                 "(H, W, 1) Float32 source depth gradient image at y-axis "
                 "obtained by FilterSobel before calling this function."},
                {"source_intensity",
                 "(H, W, 1) Float32 source intensity image obtained by "
                 "RGBToGray before calling this function"},
                {"source_intensity_dx",
                 "(H, W, 1) Float32 source intensity gradient image at x-axis "
                 "obtained by FilterSobel before calling this function."},
                {"source_intensity_dy",
                 "(H, W, 1) Float32 source intensity gradient image at y-axis "
                 "obtained by FilterSobel before calling this function."},
                {"source_vertex_map",
                 "(H, W, 3) Float32 source vertex image obtained by "
                 "CreateVertexMap before calling this function."},
                {"target", "The target RGBD image."},
                {"target_depth_map",
                 "(H, W, 1) Float32 target depth image obtained by "
                 "PreprocessDepth before calling this function."},
                {"target_intensity",
                 "(H, W, 1) Float32 target intensity image obtained by "
                 "RGBToGray before calling this function"},
                {"target_intensity_dx",
                 "(H, W, 1) Float32 target intensity gradient image at x-axis "
                 "obtained by FilterSobel before calling this function."},
                {"target_intensity_dy",
                 "(H, W, 1) Float32 target intensity gradient image at y-axis "
                 "obtained by FilterSobel before calling this function."},
                {"target_normal_map",
                 "(H, W, 3) Float32 target normal image obtained by "
                 "CreateNormalMap before calling this function."},
                {"target_vertex_map",
                 "(H, W, 3) Float32 target vertex image obtained by "
                 "CreateVertexMap before calling this function."}};

void pybind_odometry_methods(py::module &m) {
    m.def("rgbd_odometry_multi_scale", &RGBDOdometryMultiScale,
          "Function for Multi Scale RGBD odometry", "source"_a, "target"_a,
          "intrinsics"_a,
          "init_source_to_target"_a = core::Tensor::Eye(4, core::Dtype::Float64,
                                                        core::Device("CPU:0")),
          "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f,
          "iterations"_a = std::vector<int>({10, 5, 3}),
          "method"_a = Method::Hybrid, "params"_a = OdometryLossParams(),
          "criteria"_a = OdometryConvergenceCriteria());
    docstring::FunctionDocInject(m, "rgbd_odometry_multi_scale",
                                 map_shared_argument_docstrings);

    m.def("compute_pose_point_to_plane", &ComputePosePointToPlane,
          "Estimates the 4x4 rigid transformation T from source to target. "
          "Performs one iteration of RGBD odometry using loss function "
          "\f$[(V_p - V_q)^T N_p]^2\f$, where "
          "\f$ V_p \f$ denotes the vertex at pixel p in the source, "
          "\f$ V_q \f$ denotes the vertex at pixel q in the target, "
          "\f$ N_p \f$ denotes the normal at pixel p in the source. "
          "q is obtained by transforming p with init_source_to_target then "
          "projecting with intrinsics. "
          "KinectFusion, ISMAR 2011",
          "source_vertex_map"_a, "target_vertex_map"_a, "target_normal_map"_a,
          "intrinsics"_a, "init_source_to_target"_a, "depth_outlier_trunc"_a,
          "depth_huber_delta"_a);
    docstring::FunctionDocInject(m, "compute_pose_point_to_plane",
                                 map_shared_argument_docstrings);

    m.def("compute_pose_intensity", &ComputePoseIntensity,
          "Estimates the OdometryResult. "
          "Performs one iteration of RGBD odometry using loss function "
          "\f$(I_p - I_q)^2\f$, where "
          "\f$ I_p \f$ denotes the intensity at pixel p in the source, "
          "\f$ I_q \f$ denotes the intensity at pixel q in the target. "
          "q is obtained by transforming p with init_source_to_target then "
          "projecting with intrinsics. "
          "Real-time visual odometry from dense RGB-D images, ICCV Workshops, "
          "2011, ",
          "source_depth_map"_a, "target_depth_map"_a, "source_intensity"_a,
          "target_intensity"_a, "target_intensity_dx"_a,
          "target_intensity_dy"_a, "source_vertices_map"_a, "intrinsics"_a,
          "init_source_to_target"_a, "depth_outlier_trunc"_a,
          "intensity_huber_delta"_a);
    docstring::FunctionDocInject(m, "compute_pose_intensity",
                                 map_shared_argument_docstrings);

    m.def("compute_pose_hybrid", &ComputePoseHybrid,
          "Estimates the OdometryResult. "
          "Performs one iteration of RGBD odometry using loss function "
          "\f$(I_p - I_q)^2 + lambda(D_p - (D_q)')^2\f$, where "
          "\f$ I_p \f$ denotes the intensity at pixel p in the source, "
          "\f$ I_q \f$ denotes the intensity at pixel q in the target. "
          "\f$ D_p \f$ denotes the depth pixel p in the source, "
          "\f$ D_q \f$ denotes the depth pixel q in the target. "
          "q is obtained by transforming p with init_source_to_target then "
          "projecting with intrinsics. "
          "Colored ICP Revisited, ICCV 2017 ",
          "source_depth_map"_a, "target_depth_map"_a, "source_intensity"_a,
          "target_intensity"_a, "source_depth_dx"_a, "source_depth_dy"_a,
          "source_intensity_dx"_a, "source_intensity_dy"_a,
          "target_vertices_map"_a, "intrinsics"_a, "init_source_to_target"_a,
          "depth_outlier_trunc"_a, "depth_huber_delta"_a,
          "intensity_huber_delta"_a);
    docstring::FunctionDocInject(m, "compute_pose_hybrid",
                                 map_shared_argument_docstrings);
}

void pybind_odometry(py::module &m) {
    py::module m_submodule =
            m.def_submodule("odometry", "Tensor odometry pipeline.");
    pybind_odometry_classes(m_submodule);
    pybind_odometry_methods(m_submodule);
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
