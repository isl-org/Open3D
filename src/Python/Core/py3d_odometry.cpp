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

#include <Core/Geometry/Image.h>
#include <Core/Geometry/RGBDImage.h>
#include <Core/Odometry/Odometry.h>
#include <Core/Odometry/OdometryOption.h>
#include <Core/Odometry/RGBDOdometryJacobian.h>
using namespace three;

template <class RGBDOdometryJacobianBase = RGBDOdometryJacobian>
class PyRGBDOdometryJacobian : public RGBDOdometryJacobianBase
{
public:
	using RGBDOdometryJacobianBase::RGBDOdometryJacobianBase;
	void ComputeJacobianAndResidual(
			int row, std::vector<Eigen::Vector6d> &J_r, std::vector<double> &r,
			const RGBDImage &source, const RGBDImage &target,
			const Image &source_xyz,
			const RGBDImage &target_dx, const RGBDImage &target_dy,
			const Eigen::Matrix3d &intrinsic,
			const Eigen::Matrix4d &extrinsic,
			const CorrespondenceSetPixelWise &corresps) const override {
			PYBIND11_OVERLOAD_PURE(
			void,
			RGBDOdometryJacobianBase,
			row, J_r, r,
			source, target, source_xyz, target_dx, target_dy,
			extrinsic, corresps, intrinsic);
	}
};

void pybind_odometry(py::module &m)
{
	py::class_<OdometryOption> odometry_option(m, "OdometryOption");
	odometry_option
		.def(py::init([](double minimum_correspondence_ratio,
				std::vector<int> iteration_number_per_pyramid_level,
				double max_depth_diff, double min_depth, double max_depth) {
			return new OdometryOption(minimum_correspondence_ratio,
				iteration_number_per_pyramid_level,
				max_depth_diff, min_depth, max_depth);
		}), "minimum_correspondence_ratio"_a = 0.1,
				"iteration_number_per_pyramid_level"_a =
				std::vector<int>{ 20,10,5 }, "max_depth_diff"_a = 0.03,
				"min_depth"_a = 0.0, "max_depth"_a = 4.0)
		.def_readwrite("minimum_correspondence_num",
				&OdometryOption::minimum_correspondence_ratio_)
		.def_readwrite("iteration_number_per_pyramid_level",
				&OdometryOption::iteration_number_per_pyramid_level_)
		.def_readwrite("max_depth_diff", &OdometryOption::max_depth_diff_)
		.def_readwrite("min_depth", &OdometryOption::min_depth_)
		.def_readwrite("max_depth", &OdometryOption::max_depth_)
		.def("__repr__", [](const OdometryOption &c) {
		int num_pyramid_level =
				(int)c.iteration_number_per_pyramid_level_.size();
		std::string str_iteration_number_per_pyramid_level_ = "[ ";
		for (int i = 0; i < num_pyramid_level; i++)
			str_iteration_number_per_pyramid_level_ +=
					std::to_string(c.iteration_number_per_pyramid_level_[i]) + ", ";
		str_iteration_number_per_pyramid_level_ += "] ";
		return std::string("OdometryOption class.") +
				/*std::string("\nodo_init = ") + std::to_string(c.odo_init_) +*/
				std::string("\nminimum_correspondence_ratio = ") +
				std::to_string(c.minimum_correspondence_ratio_) +
				std::string("\niteration_number_per_pyramid_level = ") +
				str_iteration_number_per_pyramid_level_ +
				std::string("\nmax_depth_diff = ") +
				std::to_string(c.max_depth_diff_) +
				std::string("\nmin_depth = ") +
				std::to_string(c.min_depth_) +
				std::string("\nmax_depth = ") +
				std::to_string(c.max_depth_);
		});

	py::class_<RGBDOdometryJacobian,
			PyRGBDOdometryJacobian<RGBDOdometryJacobian>>
			jacobian(m, "RGBDOdometryJacobian");
	jacobian
			.def("compute_jacobian_and_residual",
					&RGBDOdometryJacobian::ComputeJacobianAndResidual);

	py::class_<RGBDOdometryJacobianFromColorTerm,
			PyRGBDOdometryJacobian<RGBDOdometryJacobianFromColorTerm>,
			RGBDOdometryJacobian> jacobian_color(m,
			"RGBDOdometryJacobianFromColorTerm");
	py::detail::bind_default_constructor<RGBDOdometryJacobianFromColorTerm>
			(jacobian_color);
	py::detail::bind_copy_functions<RGBDOdometryJacobianFromColorTerm>(
			jacobian_color);
	jacobian_color
		.def("__repr__", [](const RGBDOdometryJacobianFromColorTerm &te) {
		return std::string("RGBDOdometryJacobianFromColorTerm");
	});

	py::class_<RGBDOdometryJacobianFromHybridTerm,
			PyRGBDOdometryJacobian<RGBDOdometryJacobianFromHybridTerm>,
			RGBDOdometryJacobian> jacobian_hybrid(m,
			"RGBDOdometryJacobianFromHybridTerm");
	py::detail::bind_default_constructor<RGBDOdometryJacobianFromHybridTerm>
		(jacobian_hybrid);
	py::detail::bind_copy_functions<RGBDOdometryJacobianFromHybridTerm>(
		jacobian_hybrid);
	jacobian_hybrid
		.def("__repr__", [](const RGBDOdometryJacobianFromHybridTerm &te) {
		return std::string("RGBDOdometryJacobianFromHybridTerm");
	});
}

void pybind_odometry_methods(py::module &m)
{
	m.def("compute_rgbd_odometry", &ComputeRGBDOdometry,
			"Function to estimate 6D rigid motion from two RGBD image pairs",
			"rgbd_source"_a, "rgbd_target"_a,
			"pinhole_camera_intrinsic"_a = PinholeCameraIntrinsic(),
			"odo_init"_a = Eigen::Matrix4d::Identity(),
			"jacobian"_a = RGBDOdometryJacobianFromHybridTerm(),
			"option"_a = OdometryOption());
}
