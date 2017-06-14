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

#pragma once

#include <string>

namespace three {

class OdometryOption
{

public:

	OdometryOption(
			Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity(),
			double lambda_dep = 0.95, 
			int minimum_corr = 30000,
			int num_pyramid = 4, 
			int num_iter = 10, 
			double max_depth_diff = 0.07, 
			double min_depth = 0.0,	
			double max_depth = 4.0,
			double sobel_scale = 0.125, 
			bool is_tum = false,
			bool fast_reject = true,
			std::string intrinsic_path = "") :
			odo_init_(odo_init), lambda_dep_(lambda_dep), minimum_corr_(minimum_corr),
			num_pyramid_(num_pyramid), num_iter_(num_iter),
			max_depth_diff_(max_depth_diff), min_depth_(min_depth), 
			max_depth_(max_depth), sobel_scale_(sobel_scale), 
			is_tum_(is_tum), fast_reject_(fast_reject), 
			intrinsic_path_(intrinsic_path) {}
	~OdometryOption() {}

public:
	Eigen::Matrix4d odo_init_;
	double lambda_dep_;
	double minimum_corr_;
	int num_pyramid_;
	int num_iter_;
	double max_depth_diff_;
	double min_depth_;
	double max_depth_;
	double sobel_scale_;
	bool is_tum_;
	bool fast_reject_;
	std::string intrinsic_path_;
};

}
