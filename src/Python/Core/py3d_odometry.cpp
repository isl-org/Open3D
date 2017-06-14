// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Jaesik Park <syncle@gmail.com>
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

#include <Core/Odometry/Odometry.h>
#include <Core/Odometry/OdometryOption.h>
using namespace three;

void pybind_odometry(py::module &m)
{
	py::class_<OdometryOption> odometry_option(m, "OdometryOption");
	odometry_option.def("__init__", [](OdometryOption &c,
		Eigen::Matrix4d odo_init_, double lambda_dep_, int minimum_corr_,
		int num_pyramid_, int num_iter_, double max_depth_diff_,
		double min_depth_, double max_depth_, double sobel_scale_,
		bool is_tum_, bool fast_reject_, std::string intrinsic_path_) {
		new (&c)OdometryOption(odo_init_, lambda_dep_, minimum_corr_,
			num_pyramid_, num_iter_, max_depth_diff_,
			min_depth_, max_depth_, sobel_scale_,
			is_tum_, fast_reject_, intrinsic_path_);
	}, "odo_init"_a = Eigen::Matrix4d::Identity(), "lambda_dep"_a = 0.95,
		"minimum_corr"_a = 30000, "num_pyramid"_a = 4, "num_iter"_a = 10, 
		"max_depth_diff"_a = 0.07, "min_depth"_a = 0.0, "max_depth"_a = 4.0,
		"sobel_scale"_a = 0.125, "is_tum"_a = false, "fast_reject"_a = true, 
		"intrinsic_path"_a = "");
	odometry_option
		.def_readwrite("intrinsic_path", &OdometryOption::intrinsic_path_)
		.def("__repr__", [](const OdometryOption &c) {
		return std::string("OdometryOption class.") +
			/*std::string("\nodo_init = ") + std::to_string(c.odo_init_) +*/
			std::string("\nlambda_dep = ") + std::to_string(c.lambda_dep_) +
			std::string("\nminimum_corr = ") + std::to_string(c.minimum_corr_) +
			std::string("\nnum_pyramid = ") + std::to_string(c.num_pyramid_) +
			std::string("\nnum_iter = ") + std::to_string(c.num_iter_) +
			std::string("\nmax_depth_diff = ") + std::to_string(c.max_depth_diff_) +
			std::string("\nmin_depth = ") + std::to_string(c.min_depth_) +
			std::string("\nmax_depth = ") + std::to_string(c.max_depth_) +
			std::string("\nsobel_scale = ") + std::to_string(c.sobel_scale_) +
			std::string("\nis_tum = ") + std::to_string(c.is_tum_) +
			std::string("\nfast_reject = ") + std::to_string(c.fast_reject_) +
			std::string("\nintrinsic_path = ") + c.intrinsic_path_;
		});
}

void pybind_odometry_methods(py::module &m)
{
	m.def("ComputeRGBDOdometry", &ComputeRGBDOdometry,
			"Function to estimate 6D rigid motion from two RGBD image pairs",
			"color0_8bit"_a, "depth0_16bit"_a, "color1_8bit"_a, "depth1_16bit"_a,
			"option"_a = OdometryOption());
}
