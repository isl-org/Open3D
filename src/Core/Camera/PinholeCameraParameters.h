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

#include <Eigen/Core>
#include <IO/IJsonConvertible.h>

namespace three {

class PinholeCameraParameters : public IJsonConvertible
{
public:
	PinholeCameraParameters();
	virtual ~PinholeCameraParameters();

public:
	std::pair<double, double> GetFocalLength() const {
		return std::make_pair(intrinsic_matrix_(0, 0), intrinsic_matrix_(1, 1));
	}

	std::pair<double, double> GetPrincipalPoint() const {
		return std::make_pair(intrinsic_matrix_(0, 2), intrinsic_matrix_(1, 2));
	}

	double GetSkew() const { return intrinsic_matrix_(0, 1); }

	Eigen::Matrix4d GetCameraPose() const;

	virtual bool ConvertToJsonValue(Json::Value &value) const override;
	virtual bool ConvertFromJsonValue(const Json::Value &value) override;

public:
	Eigen::Matrix3d intrinsic_matrix_;
	Eigen::Matrix4d extrinsic_matrix_;
};

}	// namespace three
