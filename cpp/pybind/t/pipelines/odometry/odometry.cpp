// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/t/pipelines/odometry/odometry.h"

#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

// Odometry functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"criteria", "Odometry convergence criteria."},
                {"criteria_list", "List of Odometry convergence criteria."},
                {"depth_outlier_trunc",
                 "Depth difference threshold used to filter projective "
                 "associations."},
                {"depth_huber_delta",
                 "Huber norm parameter used in depth loss."},
                {"depth_scale",
                 "Converts depth pixel values to meters by dividing the scale "
                 "factor."},
                {"init_source_to_target",
                 "(4, 4) initial transformation matrix from source to target."},
                {"intrinsics", "(3, 3) intrinsic matrix for projection."},
                {"intensity_huber_delta",
                 "Huber norm parameter used in intensity loss."},
                {"method",
                 "Estimation method used to apply RGBD odometry. "
                 "One of (``PointToPlane``, ``Intensity``, ``Hybrid``)"},
                {"params", "Odometry loss parameters."},
                {"source", "The source RGBD image."},
                {"source_depth",
                 "(row, col, channel = 1) Float32 source depth image obtained "
                 "by PreprocessDepth before calling this function."},
                {"source_intensity",
                 "(row, col, channel = 1) Float32 source intensity image "
                 "obtained by RGBToGray before calling this function"},
                {"source_vertex_map",
                 "(row, col, channel = 3) Float32 source vertex image obtained "
                 "by CreateVertexMap before calling this function."},
                {"target", "The target RGBD image."},
                {"target_depth",
                 "(row, col, channel = 1) Float32 target depth image obtained "
                 "by PreprocessDepth before calling this function."},
                {"target_depth_dx",
                 "(row, col, channel = 1) Float32 target depth gradient image "
                 "along x-axis obtained by FilterSobel before calling this "
                 "function."},
                {"target_depth_dy",
                 "(row, col, channel = 1) Float32 target depth gradient image "
                 "along y-axis obtained by FilterSobel before calling this "
                 "function."},
                {"target_intensity",
                 "(row, col, channel = 1) Float32 target intensity image "
                 "obtained by RGBToGray before calling this function"},
                {"target_intensity_dx",
                 "(row, col, channel = 1) Float32 target intensity gradient "
                 "image along x-axis obtained by FilterSobel before calling "
                 "this function."},
                {"target_intensity_dy",
                 "(row, col, channel = 1) Float32 target intensity gradient "
                 "image along y-axis obtained by FilterSobel before calling "
                 "this function."},
                {"target_normal_map",
                 "(row, col, channel = 3) Float32 target normal image obtained "
                 "by CreateNormalMap before calling this function."},
                {"target_vertex_map",
                 "(row, col, channel = 3) Float32 target vertex image obtained "
                 "by CreateVertexMap before calling this function."}};

void pybind_odometry_declarations(py::module &m) {
    py::module m_odometry =
            m.def_submodule("odometry", "Tensor odometry pipeline.");
    py::enum_<Method>(m_odometry, "Method",
                      "Tensor odometry estimation method.")
            .value("PointToPlane", Method::PointToPlane)
            .value("Intensity", Method::Intensity)
            .value("Hybrid", Method::Hybrid)
            .export_values();
    py::class_<OdometryConvergenceCriteria> odometry_convergence_criteria(
            m_odometry, "OdometryConvergenceCriteria",
            "Convergence criteria of odometry. "
            "Odometry algorithm stops if the relative change of fitness and "
            "rmse hit ``relative_fitness`` and ``relative_rmse`` individually, "
            "or the iteration number exceeds ``max_iteration``.");
    py::class_<OdometryResult> odometry_result(m_odometry, "OdometryResult",
                                               "Odometry results.");
    py::class_<OdometryLossParams> odometry_loss_params(
            m_odometry, "OdometryLossParams", "Odometry loss parameters.");
}

void pybind_odometry_definitions(py::module &m) {
    auto m_odometry = static_cast<py::module>(m.attr("odometry"));
    // open3d.t.pipelines.odometry.OdometryConvergenceCriteria
    auto odometry_convergence_criteria =
            static_cast<py::class_<OdometryConvergenceCriteria>>(
                    m_odometry.attr("OdometryConvergenceCriteria"));
    py::detail::bind_copy_functions<OdometryConvergenceCriteria>(
            odometry_convergence_criteria);
    odometry_convergence_criteria
            .def(py::init<int, double, double>(), "max_iteration"_a,
                 "relative_rmse"_a = 1e-6, "relative_fitness"_a = 1e-6)
            .def_readwrite("max_iteration",
                           &OdometryConvergenceCriteria::max_iteration_,
                           "Maximum iteration before iteration stops.")
            .def_readwrite(
                    "relative_rmse",
                    &OdometryConvergenceCriteria::relative_rmse_,
                    "If relative change (difference) of inliner RMSE score is "
                    "lower than ``relative_rmse``, the iteration stops.")
            .def_readwrite(
                    "relative_fitness",
                    &OdometryConvergenceCriteria::relative_fitness_,
                    "If relative change (difference) of fitness score is lower "
                    "than ``relative_fitness``, the iteration stops.")
            .def("__repr__", [](const OdometryConvergenceCriteria &c) {
                return fmt::format(
                        "OdometryConvergenceCriteria("
                        "max_iteration={:d}, "
                        "relative_rmse={:e}, "
                        "relative_fitness={:e})",
                        c.max_iteration_, c.relative_rmse_,
                        c.relative_fitness_);
            });

    // open3d.t.pipelines.odometry.OdometryResult
    auto odometry_result = static_cast<py::class_<OdometryResult>>(
            m_odometry.attr("OdometryResult"));
    py::detail::bind_copy_functions<OdometryResult>(odometry_result);
    odometry_result
            .def(py::init<core::Tensor, double, double>(),
                 py::arg_v("transformation",
                           core::Tensor::Eye(4, core::Float64,
                                             core::Device("CPU:0")),
                           "open3d.core.Tensor.eye(4, "
                           "dtype=open3d.core.Dtype.Float64)"),
                 "inlier_rmse"_a = 0.0, "fitness"_a = 0.0)
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
                        "OdometryResult[fitness={:e}, inlier_rmse={:e}]."
                        "\nAccess transformation to get result.",
                        odom_result.fitness_, odom_result.inlier_rmse_);
            });

    // open3d.t.pipelines.odometry.OdometryLossParams
    auto odometry_loss_params = static_cast<py::class_<OdometryLossParams>>(
            m_odometry.attr("OdometryLossParams"));
    py::detail::bind_copy_functions<OdometryLossParams>(odometry_loss_params);
    odometry_loss_params
            .def(py::init<double, double, double>(),
                 "depth_outlier_trunc"_a = 0.07, "depth_huber_delta"_a = 0.05,
                 "intensity_huber_delta"_a = 0.1)
            .def_readwrite("depth_outlier_trunc",
                           &OdometryLossParams::depth_outlier_trunc_,
                           "float: Depth difference threshold used to filter "
                           "projective associations.")
            .def_readwrite("depth_huber_delta",
                           &OdometryLossParams::depth_huber_delta_,
                           "float: Huber norm parameter used in depth loss.")
            .def_readwrite(
                    "intensity_huber_delta",
                    &OdometryLossParams::intensity_huber_delta_,
                    "float: Huber norm parameter used in intensity loss.")
            .def("__repr__", [](const OdometryLossParams &olp) {
                return fmt::format(
                        "OdometryLossParams[depth_outlier_trunc={:e}, "
                        "depth_huber_delta={:e}, intensity_huber_delta={:e}].",
                        olp.depth_outlier_trunc_, olp.depth_huber_delta_,
                        olp.intensity_huber_delta_);
            });
    m_odometry.def(
            "rgbd_odometry_multi_scale", &RGBDOdometryMultiScale,
            py::call_guard<py::gil_scoped_release>(),
            "Function for Multi Scale RGBD odometry.", "source"_a, "target"_a,
            "intrinsics"_a,
            "init_source_to_target"_a =
                    core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
            "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f,
            "criteria_list"_a =
                    std::vector<OdometryConvergenceCriteria>({10, 5, 3}),
            "method"_a = Method::Hybrid, "params"_a = OdometryLossParams());
    docstring::FunctionDocInject(m_odometry, "rgbd_odometry_multi_scale",
                                 map_shared_argument_docstrings);

    m_odometry.def(
            "compute_odometry_result_point_to_plane",
            &ComputeOdometryResultPointToPlane,
            py::call_guard<py::gil_scoped_release>(),
            R"(Estimates the OdometryResult (4x4 rigid transformation T from
source to target, with inlier rmse and fitness). Performs one
iteration of RGBD odometry using
Loss function: :math:`[(V_p - V_q)^T N_p]^2`
where,
:math:`V_p` denotes the vertex at pixel p in the source,
:math:`V_q` denotes the vertex at pixel q in the target.
:math:`N_p` denotes the normal at pixel p in the source.
q is obtained by transforming p with init_source_to_target then
projecting with intrinsics.
Reference: KinectFusion, ISMAR 2011.)",
            "source_vertex_map"_a, "target_vertex_map"_a, "target_normal_map"_a,
            "intrinsics"_a, "init_source_to_target"_a, "depth_outlier_trunc"_a,
            "depth_huber_delta"_a);
    docstring::FunctionDocInject(m_odometry,
                                 "compute_odometry_result_point_to_plane",
                                 map_shared_argument_docstrings);

    m_odometry.def(
            "compute_odometry_result_intensity",
            &ComputeOdometryResultIntensity,
            py::call_guard<py::gil_scoped_release>(),
            R"(Estimates the OdometryResult (4x4 rigid transformation T from
source to target, with inlier rmse and fitness). Performs one
iteration of RGBD odometry using
Loss function: :math:`(I_p - I_q)^2`
where,
:math:`I_p` denotes the intensity at pixel p in the source,
:math:`I_q` denotes the intensity at pixel q in the target.
q is obtained by transforming p with init_source_to_target then
projecting with intrinsics.
Reference:
Real-time visual odometry from dense RGB-D images,
ICCV Workshops, 2017.)",
            "source_depth"_a, "target_depth"_a, "source_intensity"_a,
            "target_intensity"_a, "target_intensity_dx"_a,
            "target_intensity_dy"_a, "source_vertex_map"_a, "intrinsics"_a,
            "init_source_to_target"_a, "depth_outlier_trunc"_a,
            "intensity_huber_delta"_a);
    docstring::FunctionDocInject(m_odometry,
                                 "compute_odometry_result_intensity",
                                 map_shared_argument_docstrings);

    m_odometry.def(
            "compute_odometry_result_hybrid", &ComputeOdometryResultHybrid,
            py::call_guard<py::gil_scoped_release>(),
            R"(Estimates the OdometryResult (4x4 rigid transformation T from
source to target, with inlier rmse and fitness). Performs one
iteration of RGBD odometry using
Loss function: :math:`(I_p - I_q)^2 + \lambda(D_p - (D_q)')^2`
where,
:math:`I_p` denotes the intensity at pixel p in the source,
:math:`I_q` denotes the intensity at pixel q in the target.
:math:`D_p` denotes the depth pixel p in the source,
:math:`D_q` denotes the depth pixel q in the target.
q is obtained by transforming p with init_source_to_target then
projecting with intrinsics.
Reference: J. Park, Q.Y. Zhou, and V. Koltun,
Colored Point Cloud Registration Revisited, ICCV, 2017.)",
            "source_depth"_a, "target_depth"_a, "source_intensity"_a,
            "target_intensity"_a, "target_depth_dx"_a, "target_depth_dy"_a,
            "target_intensity_dx"_a, "target_intensity_dy"_a,
            "source_vertex_map"_a, "intrinsics"_a, "init_source_to_target"_a,
            "depth_outlier_trunc"_a, "depth_huber_delta"_a,
            "intensity_huber_delta"_a);
    docstring::FunctionDocInject(m_odometry, "compute_odometry_result_hybrid",
                                 map_shared_argument_docstrings);

    m_odometry.def("compute_odometry_information_matrix",
                   &ComputeOdometryInformationMatrix,
                   py::call_guard<py::gil_scoped_release>(), "source_depth"_a,
                   "target_depth"_a, "intrinsic"_a, "source_to_target"_a,
                   "dist_threshold"_a, "depth_scale"_a = 1000.0,
                   "depth_max"_a = 3.0);
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
