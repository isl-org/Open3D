// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2017 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "CorrespondenceChecker.h"

#include <Eigen/Dense>
#include <Core/Utility/Console.h>
#include <Core/Geometry/PointCloud.h>

namespace three{

bool CorrespondenceCheckerBasedOnEdgeLength::Check(const PointCloud &source,
		const PointCloud &target) const
{
	if (source.points_.size() != target.points_.size()) {
		PrintDebug("[CorrespondenceCheckerBasedOnEdgeLength::Check] Input mismatch.\n");
		return true;
	}
	for (auto i = 0; i < source.points_.size(); i++) {
		for (auto j = 1; j < source.points_.size(); j++) {
			// check edge ij
			double dis_source = (source.points_[i] - source.points_[j]).norm();
			double dis_target = (target.points_[i] - target.points_[j]).norm();
			if (dis_source < dis_target * similarity_threshold_ ||
					dis_target < dis_source * similarity_threshold_) {
				return false;
			}
		}
	}
	return true;
}

bool CorrespondenceCheckerBasedOnDistance::Check(const PointCloud &source,
		const PointCloud &target) const
{
	if (source.points_.size() != target.points_.size()) {
		PrintDebug("[CorrespondenceCheckerBasedOnDistance::Check] Input mismatch.\n");
		return true;
	}
	for (auto i = 0; i < source.points_.size(); i++) {
		if ((target.points_[i] - source.points_[i]).norm() >
				distance_threshold_) {
			return false;
		}
	}
	return true;
}

bool CorrespondenceCheckerBasedOnNormal::Check(const PointCloud &source,
		const PointCloud &target) const
{
	if (source.HasNormals() == false || target.HasNormals() == false ||
			source.points_.size() != target.points_.size()) {
		PrintDebug("[CorrespondenceCheckerBasedOnNormal::Check] Something is wrong.\n");
		return true;
	}
	double cos_normal_angle_threshold = std::cos(normal_angle_threshold_);
	for (auto i = 0; i < source.points_.size(); i++) {
		if (target.normals_[i].dot(source.normals_[i]) <
				cos_normal_angle_threshold) {
			return false;
		}
	}
	return true;
}

}	// namespace three
