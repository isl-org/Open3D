// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
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
#include <Eigen/Core>

namespace three {

class ViewTrajectory {
public:
	struct ViewStatus {
	public:
		double field_of_view;
		double zoom;
		Eigen::Vector3d lookat;
		Eigen::Vector3d up;
		Eigen::Vector3d front;
	};

	const int INTERVAL_MAX = 59;
	const int INTERVAL_MIN = 0;
	const int INTERVAL_STEP = 1;
	const int INTERVAL_DEFAULT = 29;

public:
	void ComputeInterpolationCoefficients(bool close_loop = false);

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

	ViewStatus GetInterpolatedFrame(size_t k);

public:
	std::vector<ViewStatus> view_status_;
	bool is_loop_ = false;
	int interval_ = INTERVAL_DEFAULT;
};

}	// namespace three
