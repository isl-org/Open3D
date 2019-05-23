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

#include "Python/odometry/odometry.h"

#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/Odometry/Odometry.h"
#include "Open3D/Odometry/OdometryOption.h"
#include "Open3D/Odometry/RGBDOdometryJacobian.h"
#include "Python/docstring.h"

using namespace open3d;

template <class RGBDOdometryJacobianBase = odometry::RGBDOdometryJacobian>
class PyRGBDOdometryJacobian : public RGBDOdometryJacobianBase {
public:
    using RGBDOdometryJacobianBase::RGBDOdometryJacobianBase;
    void ComputeJacobianAndResidual(
            int row,
            std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
            std::vector<double> &r,
            const geometry::RGBDImage &source,
            const geometry::RGBDImage &target,
            const geometry::Image &source_xyz,
            const geometry::RGBDImage &target_dx,
            const geometry::RGBDImage &target_dy,
            const Eigen::Matrix3d &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const odometry::CorrespondenceSetPixelWise &corresps)
            const override {
        PYBIND11_OVERLOAD_PURE(void, RGBDOdometryJacobianBase, row, J_r, r,
                               source, target, source_xyz, target_dx, target_dy,
                               extrinsic, corresps, intrinsic);
    }
};

void pybind_odometry_classes(py::module &m) {
    // open3d.odometry.OdometryOption
    py::class_<odometry::OdometryOption> odometry_option(
            m, "OdometryOption", "Class that defines Odometry options.");
    odometry_option
            .def(py::init(
                         [](std::vector<int> iteration_number_per_pyramid_level,
                            double max_depth_diff, double min_depth,
                            double max_depth) {
                             return new odometry::OdometryOption(
                                     iteration_number_per_pyramid_level,
                                     max_depth_diff, min_depth, max_depth);
                         }),
                 "iteration_number_per_pyramid_level"_a =
                         std::vector<int>{20, 10, 5},
                 "max_depth_diff"_a = 0.03, "min_depth"_a = 0.0,
                 "max_depth"_a = 4.0)
            .def_readwrite("iteration_number_per_pyramid_level",
                           &odometry::OdometryOption::
                                   iteration_number_per_pyramid_level_,
                           "List(int): Iteration number per image pyramid "
                           "level, typically larger image in the pyramid have "
                           "lower interation number to reduce computation "
                           "time.")
            .def_readwrite("max_depth_diff",
                           &odometry::OdometryOption::max_depth_diff_,
                           "Maximum depth difference to be considered as "
                           "correspondence. In depth image domain, if two "
                           "aligned pixels have a depth difference less than "
                           "specified value, they are considered as a "
                           "correspondence. Larger value induce more "
                           "aggressive search, but it is prone to unstable "
                           "result.")
            .def_readwrite("min_depth", &odometry::OdometryOption::min_depth_,
                           "Pixels that has smaller than specified depth "
                           "values are ignored.")
            .def_readwrite("max_depth", &odometry::OdometryOption::max_depth_,
                           "Pixels that has larger than specified depth values "
                           "are ignored.")
            .def("__repr__", [](const odometry::OdometryOption &c) {
                int num_pyramid_level =
                        (int)c.iteration_number_per_pyramid_level_.size();
                std::string str_iteration_number_per_pyramid_level_ = "[ ";
                for (int i = 0; i < num_pyramid_level; i++)
                    str_iteration_number_per_pyramid_level_ +=
                            std::to_string(
                                    c.iteration_number_per_pyramid_level_[i]) +
                            ", ";
                str_iteration_number_per_pyramid_level_ += "] ";
                return std::string("odometry::OdometryOption class.") +
                       /*std::string("\nodo_init = ") +
                          std::to_string(c.odo_init_) +*/
                       std::string("\niteration_number_per_pyramid_level = ") +
                       str_iteration_number_per_pyramid_level_ +
                       std::string("\nmax_depth_diff = ") +
                       std::to_string(c.max_depth_diff_) +
                       std::string("\nmin_depth = ") +
                       std::to_string(c.min_depth_) +
                       std::string("\nmax_depth = ") +
                       std::to_string(c.max_depth_);
            });

    // open3d.odometry.RGBDOdometryJacobian
    py::class_<odometry::RGBDOdometryJacobian,
               PyRGBDOdometryJacobian<odometry::RGBDOdometryJacobian>>
            jacobian(
                    m, "RGBDOdometryJacobian",
                    "Base class that computes Jacobian from two RGB-D images.");

    // open3d.odometry.RGBDOdometryJacobianFromColorTerm: RGBDOdometryJacobian
    py::class_<
            odometry::RGBDOdometryJacobianFromColorTerm,
            PyRGBDOdometryJacobian<odometry::RGBDOdometryJacobianFromColorTerm>,
            odometry::RGBDOdometryJacobian>
            jacobian_color(m, "RGBDOdometryJacobianFromColorTerm",
                           R"(Class to Compute Jacobian using color term.

Energy: :math:`(I_p-I_q)^2.`

Reference:

F. Steinbrucker, J. Sturm, and D. Cremers.

Real-time visual odometry from dense RGB-D images.

In ICCV Workshops, 2011.)");
    py::detail::bind_default_constructor<
            odometry::RGBDOdometryJacobianFromColorTerm>(jacobian_color);
    py::detail::bind_copy_functions<
            odometry::RGBDOdometryJacobianFromColorTerm>(jacobian_color);
    jacobian_color.def(
            "__repr__",
            [](const odometry::RGBDOdometryJacobianFromColorTerm &te) {
                return std::string("RGBDOdometryJacobianFromColorTerm");
            });

    // open3d.odometry.RGBDOdometryJacobianFromHybridTerm: RGBDOdometryJacobian
    py::class_<odometry::RGBDOdometryJacobianFromHybridTerm,
               PyRGBDOdometryJacobian<
                       odometry::RGBDOdometryJacobianFromHybridTerm>,
               odometry::RGBDOdometryJacobian>
            jacobian_hybrid(m, "RGBDOdometryJacobianFromHybridTerm",
                            R"(Class to compute Jacobian using hybrid term

Energy: :math:`(I_p-I_q)^2 + \lambda(D_p-D_q')^2`

Reference:

J. Park, Q.-Y. Zhou, and V. Koltun

Anonymous submission.)");
    py::detail::bind_default_constructor<
            odometry::RGBDOdometryJacobianFromHybridTerm>(jacobian_hybrid);
    py::detail::bind_copy_functions<
            odometry::RGBDOdometryJacobianFromHybridTerm>(jacobian_hybrid);
    jacobian_hybrid.def(
            "__repr__",
            [](const odometry::RGBDOdometryJacobianFromHybridTerm &te) {
                return std::string("RGBDOdometryJacobianFromHybridTerm");
            });
}

void pybind_odometry_methods(py::module &m) {
    m.def("compute_rgbd_odometry", &odometry::ComputeRGBDOdometry,
          "Function to estimate 6D rigid motion from two RGBD image pairs. "
          "Output: (is_success, 4x4 motion matrix, 6x6 information matrix).",
          "rgbd_source"_a, "rgbd_target"_a,
          "pinhole_camera_intrinsic"_a = camera::PinholeCameraIntrinsic(),
          "odo_init"_a = Eigen::Matrix4d::Identity(),
          "jacobian"_a = odometry::RGBDOdometryJacobianFromHybridTerm(),
          "option"_a = odometry::OdometryOption());
    docstring::FunctionDocInject(
            m, "compute_rgbd_odometry",
            {
                    {"rgbd_source", "Source RGBD image."},
                    {"rgbd_target", "Target RGBD image."},
                    {"pinhole_camera_intrinsic", "Camera intrinsic parameters"},
                    {"odo_init", "Initial 4x4 motion matrix estimation."},
                    {"jacobian",
                     "The odometry Jacobian method to use. Can be "
                     "``odometry::RGBDOdometryJacobianFromHybridTerm()`` or "
                     "``odometry::RGBDOdometryJacobianFromColorTerm().``"},
                    {"option", "Odometry hyper parameteres."},
            });
}

void pybind_odometry(py::module &m) {
    py::module m_submodule = m.def_submodule("odometry");
    pybind_odometry_classes(m_submodule);
    pybind_odometry_methods(m_submodule);
}
