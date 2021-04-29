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

#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/RGBDImage.h"
#include "open3d/t/pipelines/odometry/RGBDOdometry.h"
#include "pybind/docstring.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace odometry {

template <class RGBDOdometryJacobianBase = RGBDOdometryJacobian>
class PyRGBDOdometryJacobian : public RGBDOdometryJacobianBase {
public:
    using RGBDOdometryJacobianBase::RGBDOdometryJacobianBase;
    void ComputeJacobianAndResidual(
            int row,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r,
            std::vector<double> &w,
            const geometry::RGBDImage &source,
            const geometry::RGBDImage &target,
            const geometry::Image &source_xyz,
            const geometry::RGBDImage &target_dx,
            const geometry::RGBDImage &target_dy,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const CorrespondenceSetPixelWise &corresps) const override {
        PYBIND11_OVERLOAD_PURE(void, RGBDOdometryJacobianBase, row, J_r, r,
                               source, target, source_xyz, target_dx, target_dy,
                               extrinsic, corresps, intrinsic);
    }
};

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
                        "with relative_fitness={:e}, relative_rmse={:e}, " c
                                .relative_fitness_,
                        c.relative_rmse_);
            });

    // open3d.t.pipelines.odometry.OdometryResult
    py::class_<OdometryResult> odometry_result(
            m, "OdometryResult", "Class that contains the odometry results.");
    py::detail::bind_default_constructor<OdometryResult>(odometry_result);
    py::detail::bind_copy_functions<OdometryResult>(odometry_result);
    odometry_result
            .def_readwrite("transformation",
                           &RegistrationResult::transformation_,
                           "``4 x 4`` float64 tensor on CPU: The estimated "
                           "transformation matrix.")
            .def_readwrite("inlier_rmse", &OdometryResult::inlier_rmse_,
                           "float: RMSE of all inlier correspondences. Lower "
                           "is better.")
            .def_readwrite(
                    "fitness", &OdometryResult::fitness_,
                    "float: The overlapping area (# of inlier correspondences "
                    "/ # of points in target). Higher is better.")
            .def("__repr__", [](const OdometryResult & or) {
                return fmt::format(
                        "OdometryResult with "
                        "fitness={:e}"
                        ", inlier_rmse={:e}"
                        "\nAccess transformation to get result.",
                        or.fitness_, or.inlier_rmse_);
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
                        olp.fitness_, olp.inlier_rmse_);
            });
}

// Odometry functions have similar arguments, sharing arg docstrings.
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"criteria", "Odometry convergence criteria."},
                {"depth_outlier_trunc",
                 "Depth difference threshold used to filter projective "
                 "associations."},
                {"depth_scale",
                 "Converts depth pixel values to meters by dividing the scale "
                 "factor."},
                {"init_source_to_target",
                 "(4, 4) initial transformation matrix from source to target."},
                {"intrinsics", "(3, 3) intrinsic matrix for projection."},
                {"iterations",
                 "o3d.utility.IntVector Iterations in multiscale "
                 "odometry, from coarse to fine."},
                {"method",
                 "Estimation method used to apply RGBD odometry. "
                 "One of (``PointToPlane``, ``Intensity``, ``Hybrid``)"},
                {"params", "Odometry loss parameters."},
                {"source", "The source RGBD image."},
                {"target", "The target RGBD image."}};

void pybind_odometry_methods(py::module &m) {
    m.def("rgbd_odometry_multi_scale", &RGBDOdometryMultiScale,
          "Function for Multi Scale RGBD odometry", "source"_a, "target"_a,
          "intrinsics"_a,
          "init_source_to_target"_a = core::Tensor::Eye(4, core::Dtype::Float64,
                                                        core::Device("CPU:0")),
          "depth_scale"_a = 1000.0f, "depth_max"_a = 3.0f,
          "iterations"_a = {10, 5, 3}, "method"_a = Method::Hybrid,
          "params"_a = OdometryLossParams(),
          "criteria"_a = OdometryConvergenceCriteria());
    docstring::FunctionDocInject(m, "rgbd_odometry_multi_scale",
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
)  // namespace t
}  // namespace t
