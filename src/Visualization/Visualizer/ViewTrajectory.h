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

#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <IO/IJsonConvertible.h>

namespace three {

class ViewTrajectory : IJsonConvertible
{
public:
	typedef Eigen::Matrix<double, 11, 4, Eigen::RowMajor> Matrix11x4d;
	typedef Eigen::Matrix<double, 11, 1> Vector11d;

	struct ViewStatus {
	public:
		double field_of_view;
		double zoom;
		Eigen::Vector3d lookat;
		Eigen::Vector3d up;
		Eigen::Vector3d front;

	public:
		Vector11d ConvertToVector11d();
		void ConvertFromVector11d(const ViewTrajectory::Vector11d &v);
	};

	static const int INTERVAL_MAX;
	static const int INTERVAL_MIN;
	static const int INTERVAL_STEP;
	static const int INTERVAL_DEFAULT;

public:
	ViewTrajectory() {}
	virtual ~ViewTrajectory() {}

public:
	/// Function to compute a Cubic Spline Interpolation
	/// See this paper for details:
	/// Bartels, R. H.; Beatty, J. C.; and Barsky, B. A. "Hermite and Cubic 
	/// Spline Interpolation." Ch. 3 in An Introduction to Splines for Use in 
	/// Computer Graphics and Geometric Modelling. San Francisco, CA: Morgan 
	/// Kaufmann, pp. 9-17, 1998.
	/// Also see explanation on this page:
	/// http://mathworld.wolfram.com/CubicSpline.html
	void ComputeInterpolationCoefficients();

	void ChangeInterval(int change) {
		int new_interval = interval_ + change * INTERVAL_STEP;
		if (new_interval >= INTERVAL_MIN && new_interval <= INTERVAL_MAX)
		{
			interval_ = new_interval;
		}
	}

	size_t NumOfFrames() {
		if (view_status_.empty()) {
			return 0;
		} else {
			return is_loop_ ? (interval_ + 1) * view_status_.size() :
					(interval_ + 1) * (view_status_.size() - 1) + 1;
		}
	}

	void Reset() {
		is_loop_ = false;
		interval_ = INTERVAL_DEFAULT;
		view_status_.clear();
	}

	std::tuple<bool, ViewStatus> GetInterpolatedFrame(size_t k);

	virtual bool ConvertToJsonValue(Json::Value &value) const override;
	virtual bool ConvertFromJsonValue(const Json::Value &value) override;

public:
	std::vector<ViewStatus> view_status_;
	bool is_loop_ = false;
	int interval_ = INTERVAL_DEFAULT;
	std::vector<Matrix11x4d> coeff_;
};

}	// namespace three
