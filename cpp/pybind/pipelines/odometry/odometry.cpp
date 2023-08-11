// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/odometry/Odometry.h"

#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/pipelines/odometry/OdometryOption.h"
#include "open3d/pipelines/odometry/RGBDOdometryJacobian.h"
#include "pybind/docstring.h"
#include "pybind/pipelines/odometry/odometry.h"

namespace open3d {
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
    // open3d.odometry.OdometryOption
    py::class_<OdometryOption> odometry_option(
            m, "OdometryOption", "Class that defines Odometry options.");
    odometry_option
            .def(py::init(
                         [](std::vector<int> iteration_number_per_pyramid_level,
                            double depth_diff_max, double depth_min,
                            double depth_max) {
                             return new OdometryOption(
                                     iteration_number_per_pyramid_level,
                                     depth_diff_max, depth_min, depth_max);
                         }),
                 "iteration_number_per_pyramid_level"_a =
                         std::vector<int>{20, 10, 5},
                 "depth_diff_max"_a = 0.03, "depth_min"_a = 0.0,
                 "depth_max"_a = 4.0)
            .def_readwrite("iteration_number_per_pyramid_level",
                           &OdometryOption::iteration_number_per_pyramid_level_,
                           "List(int): Iteration number per image pyramid "
                           "level, typically larger image in the pyramid have "
                           "lower iteration number to reduce computation "
                           "time.")
            .def_readwrite("depth_diff_max", &OdometryOption::depth_diff_max_,
                           "Maximum depth difference to be considered as "
                           "correspondence. In depth image domain, if two "
                           "aligned pixels have a depth difference less than "
                           "specified value, they are considered as a "
                           "correspondence. Larger value induce more "
                           "aggressive search, but it is prone to unstable "
                           "result.")
            .def_readwrite("depth_min", &OdometryOption::depth_min_,
                           "Pixels that has smaller than specified depth "
                           "values are ignored.")
            .def_readwrite("depth_max", &OdometryOption::depth_max_,
                           "Pixels that has larger than specified depth values "
                           "are ignored.")
            .def("__repr__", [](const OdometryOption &c) {
                int num_pyramid_level =
                        (int)c.iteration_number_per_pyramid_level_.size();
                std::string str_iteration_number_per_pyramid_level_ = "[ ";
                for (int i = 0; i < num_pyramid_level; i++)
                    str_iteration_number_per_pyramid_level_ +=
                            std::to_string(
                                    c.iteration_number_per_pyramid_level_[i]) +
                            ", ";
                str_iteration_number_per_pyramid_level_ += "] ";
                return std::string("OdometryOption class.") +
                       /*std::string("\nodo_init = ") +
                          std::to_string(c.odo_init_) +*/
                       std::string("\niteration_number_per_pyramid_level = ") +
                       str_iteration_number_per_pyramid_level_ +
                       std::string("\ndepth_diff_max = ") +
                       std::to_string(c.depth_diff_max_) +
                       std::string("\ndepth_min = ") +
                       std::to_string(c.depth_min_) +
                       std::string("\ndepth_max = ") +
                       std::to_string(c.depth_max_);
            });

    // open3d.odometry.RGBDOdometryJacobian
    py::class_<RGBDOdometryJacobian,
               PyRGBDOdometryJacobian<RGBDOdometryJacobian>>
            jacobian(
                    m, "RGBDOdometryJacobian",
                    "Base class that computes Jacobian from two RGB-D images.");

    jacobian.def(
            "compute_jacobian_and_residual",
            &RGBDOdometryJacobian::ComputeJacobianAndResidual,
            py::call_guard<py::gil_scoped_release>(),
            "Function to compute i-th row of J and r the vector form of J_r is "
            "basically 6x1 matrix, but it can be easily extendable to 6xn "
            "matrix. See RGBDOdometryJacobianFromHybridTerm for this case."
            "row"_a,
            "J_r"_a, "r"_a, "w"_a, "source"_a, "target"_a, "source_xyz"_a,
            "target_dx"_a, "target_dy"_a, "intrinsic"_a, "extrinsic"_a,
            "corresps"_a);

    // open3d.odometry.RGBDOdometryJacobianFromColorTerm: RGBDOdometryJacobian
    py::class_<RGBDOdometryJacobianFromColorTerm,
               PyRGBDOdometryJacobian<RGBDOdometryJacobianFromColorTerm>,
               RGBDOdometryJacobian>
            jacobian_color(m, "RGBDOdometryJacobianFromColorTerm",
                           R"(Class to Compute Jacobian using color term.

Energy: :math:`(I_p-I_q)^2.`

Reference:

F. Steinbrucker, J. Sturm, and D. Cremers.

Real-time visual odometry from dense RGB-D images.

In ICCV Workshops, 2011.)");
    py::detail::bind_default_constructor<RGBDOdometryJacobianFromColorTerm>(
            jacobian_color);
    py::detail::bind_copy_functions<RGBDOdometryJacobianFromColorTerm>(
            jacobian_color);
    jacobian_color.def(
            "__repr__", [](const RGBDOdometryJacobianFromColorTerm &te) {
                return std::string("RGBDOdometryJacobianFromColorTerm");
            });

    // open3d.odometry.RGBDOdometryJacobianFromHybridTerm: RGBDOdometryJacobian
    py::class_<RGBDOdometryJacobianFromHybridTerm,
               PyRGBDOdometryJacobian<RGBDOdometryJacobianFromHybridTerm>,
               RGBDOdometryJacobian>
            jacobian_hybrid(m, "RGBDOdometryJacobianFromHybridTerm",
                            R"(Class to compute Jacobian using hybrid term

Energy: :math:`(I_p-I_q)^2 + \lambda(D_p-D_q')^2`

Reference:

J. Park, Q.-Y. Zhou, and V. Koltun

Anonymous submission.)");
    py::detail::bind_default_constructor<RGBDOdometryJacobianFromHybridTerm>(
            jacobian_hybrid);
    py::detail::bind_copy_functions<RGBDOdometryJacobianFromHybridTerm>(
            jacobian_hybrid);
    jacobian_hybrid.def(
            "__repr__", [](const RGBDOdometryJacobianFromHybridTerm &te) {
                return std::string("RGBDOdometryJacobianFromHybridTerm");
            });
}

void pybind_odometry_methods(py::module &m) {
    m.def("compute_rgbd_odometry", &ComputeRGBDOdometry,
          py::call_guard<py::gil_scoped_release>(),
          "Function to estimate 6D rigid motion from two RGBD image pairs. "
          "Output: (is_success, 4x4 motion matrix, 6x6 information matrix).",
          "rgbd_source"_a, "rgbd_target"_a,
          "pinhole_camera_intrinsic"_a = camera::PinholeCameraIntrinsic(),
          "odo_init"_a = Eigen::Matrix4d::Identity(),
          "jacobian"_a = RGBDOdometryJacobianFromHybridTerm(),
          "option"_a = OdometryOption());
    docstring::FunctionDocInject(
            m, "compute_rgbd_odometry",
            {
                    {"rgbd_source", "Source RGBD image."},
                    {"rgbd_target", "Target RGBD image."},
                    {"pinhole_camera_intrinsic", "Camera intrinsic parameters"},
                    {"odo_init", "Initial 4x4 motion matrix estimation."},
                    {"jacobian",
                     "The odometry Jacobian method to use. Can be "
                     "``"
                     "RGBDOdometryJacobianFromHybridTerm()`` or "
                     "``RGBDOdometryJacobianFromColorTerm("
                     ").``"},
                    {"option", "Odometry hyper parameters."},
            });

    m.def("compute_correspondence", &ComputeCorrespondence,
          py::call_guard<py::gil_scoped_release>(),
          "Function to estimate point to point correspondences from two depth "
          "images. A vector of u_s, v_s, u_t, v_t which maps the 2d "
          "coordinates of source to target.",
          "intrinsic_matrix"_a, "extrinsic"_a, "depth_s"_a, "depth_t"_a,
          "option"_a = OdometryOption());
    docstring::FunctionDocInject(
            m, "compute_correspondence",
            {
                    {"intrinsic_matrix", "Camera intrinsic parameters."},
                    {"extrinsic",
                     "Estimation of transform from source to target."},
                    {"depth_s", "Source depth image."},
                    {"depth_t", "Target depth image."},
                    {"option", "Odometry hyper parameters."},
            });
}

void pybind_odometry(py::module &m) {
    py::module m_submodule = m.def_submodule("odometry", "Odometry pipeline.");
    pybind_odometry_classes(m_submodule);
    pybind_odometry_methods(m_submodule);
}

}  // namespace odometry
}  // namespace pipelines
}  // namespace open3d
