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

#include <iostream> 
#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <Core/Odometry/OdometryOption.h>

namespace three {

class Image;

class RGBDImage;

typedef std::vector<Eigen::Vector4i> CorrespondenceSetPixelWise;

/// Base class that computes Jacobian from two RGB-D images
class RGBDOdometryJacobian
{
public:
	RGBDOdometryJacobian() {}
	virtual ~RGBDOdometryJacobian() {}

public:
	virtual std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeJacobian(
			const RGBDImage &source, const RGBDImage &target,
			const Image &source_xyz,
			const RGBDImage &target_dx, const RGBDImage &target_dy,
			const Eigen::Matrix4d &odo,
			const CorrespondenceSetPixelWise &corresps,
			const Eigen::Matrix3d &camera_matrix,
			const OdometryOption &option) const = 0;
};

/// Compute Jacobian using color term (I_p-I_q)^2
class RGBDOdometryJacobianfromColorTerm : public RGBDOdometryJacobian
{
public:
	RGBDOdometryJacobianfromColorTerm() {}
	~RGBDOdometryJacobianfromColorTerm() override {}

public:
	std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeJacobian(
			const RGBDImage &source, const RGBDImage &target,
			const Image &source_xyz,
			const RGBDImage &target_dx, const RGBDImage &target_dy,
			const Eigen::Matrix4d &odo,
			const CorrespondenceSetPixelWise &corresps,
			const Eigen::Matrix3d &camera_matrix,
			const OdometryOption &option) const override;
};

/// Compute Jacobian using hybrid term (I_p-I_q)^2 + lambda(D_p-(D_q)')^2
class RGBDOdometryJacobianfromHybridTerm : public RGBDOdometryJacobian
{
public:
	RGBDOdometryJacobianfromHybridTerm() {}
	~RGBDOdometryJacobianfromHybridTerm() override {}

public:
	std::tuple<Eigen::MatrixXd, Eigen::VectorXd> ComputeJacobian(
			const RGBDImage &source, const RGBDImage &target,
			const Image &source_xyz,
			const RGBDImage &target_dx, const RGBDImage &target_dy,
			const Eigen::Matrix4d &odo,
			const CorrespondenceSetPixelWise &corresps,
			const Eigen::Matrix3d &camera_matrix,
			const OdometryOption &option) const override;
};

}	// namespace three
