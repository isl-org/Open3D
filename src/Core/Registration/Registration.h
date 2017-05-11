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

#include <Core/Registration/ConvergenceCriteria.h>
#include <Core/Registration/TransformationEstimation.h>

namespace three {

class PointCloud;

class RegistrationResult
{
public:
	Eigen::Matrix4d transformation;
	double inlier_rmse;
	double fitness;
};

/// Function for evaluation
RegistrationResult EvaluateRegistration(const PointCloud &source,
		const PointCloud &target, double max_correspondence_distance,
		const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity());

/// Functions for registration
RegistrationResult RegistrationICP(const PointCloud &source,
		const PointCloud &target, double max_correspondence_distance,
		const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
		const TransformationEstimation &estimation =
		TransformationEstimationPointToPoint(false),
		const ConvergenceCriteria &criteria = ConvergenceCriteria());

RegistrationResult RegistrationRANSAC(const PointCloud &source,
		const PointCloud &target, const CorrespondenceSet &corres,
		double max_correspondence_distance,
		const TransformationEstimation &estimation =
		TransformationEstimationPointToPoint(false),
		int ransac_n = 6, int max_ransac_iteration = 1000);

}	// namespace three
