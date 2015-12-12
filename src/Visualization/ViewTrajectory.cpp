// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "ViewTrajectory.h"

#include <Eigen/Dense>

namespace three{

const int ViewTrajectory::INTERVAL_MAX = 59;
const int ViewTrajectory::INTERVAL_MIN = 0;
const int ViewTrajectory::INTERVAL_STEP = 1;
const int ViewTrajectory::INTERVAL_DEFAULT = 29;

ViewTrajectory::Vector11d ViewTrajectory::ViewStatus::ConvertToVector11d()
{
	ViewTrajectory::Vector11d v;
	v(0) = field_of_view;
	v(1) = zoom;
	v.block<3, 1>(2, 0) = lookat;
	v.block<3, 1>(5, 0) = up;
	v.block<3, 1>(8, 0) = front;
	return v;
}

void ViewTrajectory::ViewStatus::ConvertFromVector11d(
		const ViewTrajectory::Vector11d &v)
{
	field_of_view = v(0);
	zoom = v(1);
	lookat = v.block<3, 1>(2, 0);
	up = v.block<3, 1>(5, 0);
	front = v.block<3, 1>(8, 0);
}

void ViewTrajectory::ComputeInterpolationCoefficients()
{
	if (view_status_.empty()) {
		return;
	}

	// num_of_status is used frequently, give it an alias
	int n = int(view_status_.size());
	coeff_.resize(n);

	// Consider ViewStatus as a point in an 11-dimensional space.
	for (int i = 0; i < n; i++) {
		coeff_[i].setZero();
		coeff_[i].block<11, 1>(0, 0) = view_status_[i].ConvertToVector11d();
	}

	// Handle degenerate cases first
	if (n == 1) {
		return;
	} else if (n == 2) {
		coeff_[0].block<11, 1>(0, 1) = coeff_[1].block<11, 1>(0, 0) - 
				coeff_[0].block<11, 1>(0, 0);
		coeff_[1].block<11, 1>(0, 1) = coeff_[0].block<11, 1>(0, 0) - 
				coeff_[1].block<11, 1>(0, 0);
		return;
	}

	Eigen::MatrixXd A(n, n);
	Eigen::VectorXd b(n);

	// Set matrix A first
	A.setZero();
	
	// Set first and last line
	if (is_loop_) {
		A(0, 0) = 4.0;
		A(0, 1) = 1.0;
		A(0, n - 1) = 1.0;
		A(n - 1, 0) = 1.0;
		A(n - 1, n - 2) = 1.0;
		A(n - 1, n - 1) = 4.0;
	} else {
		A(0, 0) = 2.0;
		A(0, 1) = 1.0;
		A(n - 1, n - 2) = 1.0;
		A(n - 1, n - 1) = 2.0;
	}

	// Set middle part
	for (int i = 1; i < n - 1; i++) {
		A(i, i) = 4.0;
		A(i, i - 1) = 1.0;
		A(i, i + 1) = 1.0;
	}

	auto llt_solver = A.llt();

	for (int k = 0; k < 11; k++) {
		// Now we work for the k-th coefficient
		b.setZero();

		// Set first and last line
		if (is_loop_) {
			b(0) = 3.0 * (coeff_[1](k, 0) - coeff_[n - 1](k, 0));
			b(n - 1) = 3.0 * (coeff_[0](k, 0) - coeff_[n - 2](k, 0));
		} else {
			b(0) = 3.0 * (coeff_[1](k, 0) - coeff_[0](k, 0));
			b(n - 1) = 3.0 * (coeff_[n - 1](k, 0) - coeff_[n - 2](k, 0));
		}

		// Set middle part
		for (int i = 1; i < n - 1; i++) {
			b(i) = 3.0 * (coeff_[i + 1](k, 0) - coeff_[i - 1](k, 0));
		}

		// Solve the linear system
		Eigen::VectorXd x = llt_solver.solve(b);
	
		for (int i = 0; i < n; i++) {
			int i1 = (i + 1) % n;
			coeff_[i](k, 1) = x(i);
			coeff_[i](k, 2) = 3.0 * (coeff_[i1](k, 0) - coeff_[i](k, 0))
					- 2.0 * x(i) - x(i1);
			coeff_[i](k, 3) = 2.0 * (coeff_[i](k, 0) - coeff_[i1](k, 0))
					+ x(i) + x(i1);
		}
	}
}

std::tuple<bool, ViewTrajectory::ViewStatus> 
		ViewTrajectory::GetInterpolatedFrame(size_t k)
{
	ViewStatus status;
	if (view_status_.empty() || k >= NumOfFrames()) {
		return std::make_tuple(false, status);
	}
	size_t segment_index = k / (interval_ + 1);
	double segment_fraction = double(k - segment_index * (interval_ + 1)) / 
			double(interval_ + 1);
	Eigen::Vector4d s(1.0, segment_fraction, 
			segment_fraction * segment_fraction,
			segment_fraction * segment_fraction * segment_fraction);
	Vector11d status_in_vector = coeff_[segment_index] * s;
	status.ConvertFromVector11d(status_in_vector);
	return std::make_tuple(true, status);
}

}	// namespace three
