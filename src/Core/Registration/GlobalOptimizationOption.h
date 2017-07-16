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

namespace three {

namespace {
	double SetValueZeroToOne(double input, double default_value) {
		return input < 0.0 || input > 1.0 ? default_value : input;
	}
}

class GlobalOptimizationOption
{
public:
	GlobalOptimizationOption(
			int max_iteration = 100,
			int max_iteration_lm = 20,
			double line_process_weight = 10000.,
			double edge_prune_threshold = 0.25,
			double upper_scale_factor = 2./3.,
			double lower_scale_factor = 1./3.) :
			max_iteration_(max_iteration),
			max_iteration_lm_(max_iteration_lm),
			line_process_weight_(line_process_weight),
			edge_prune_threshold_(edge_prune_threshold),
			upper_scale_factor_(upper_scale_factor),
			lower_scale_factor_(lower_scale_factor) {
				edge_prune_threshold_ = 
						SetValueZeroToOne(edge_prune_threshold, 0.25);
				upper_scale_factor_ =
						SetValueZeroToOne(upper_scale_factor_, 2./3.);
				lower_scale_factor_ =
						SetValueZeroToOne(lower_scale_factor_, 1./3.);
			}
	~GlobalOptimizationOption() {}

public:
	/// maximum iteration number for iterative optmization module.
	/// max_iteration_lm_ is used for additional Levenberg-Marquardt inner loop 
	/// that automatically changes steepest gradient gain
	int max_iteration_;
	int max_iteration_lm_;
	
	/// Reference list in GlobalOptimization.h
	/// line_process_weight_ is equivalent to mu in [Choi et al 2015].
	double line_process_weight_;	

	/// According to [Choi et al 2015], 
	/// line_process < edge_prune_threshold_ (0.25) is pruned.
	double edge_prune_threshold_;
	
	/// below two variables used for levenberg marquardt algorithm
	/// these are scaling factors that increase/decrease lambda 
	/// used in H_LM = H + lambda * I
	double upper_scale_factor_;
	double lower_scale_factor_;
};

}
