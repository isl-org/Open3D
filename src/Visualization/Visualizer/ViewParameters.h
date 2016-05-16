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

class ViewParameters : public IJsonConvertible
{
public:
	typedef Eigen::Matrix<double, 11, 4, Eigen::RowMajor> Matrix11x4d;
	typedef Eigen::Matrix<double, 11, 1> Vector11d;

public:
	ViewParameters() {}
	virtual ~ViewParameters() {}

public:
	Vector11d ConvertToVector11d();
	void ConvertFromVector11d(const Vector11d &v);
	virtual bool ConvertToJsonValue(Json::Value &value) const override;
	virtual bool ConvertFromJsonValue(const Json::Value &value) override;

public:
	double field_of_view_;
	double zoom_;
	Eigen::Vector3d lookat_;
	Eigen::Vector3d up_;
	Eigen::Vector3d front_;
};

}	// namespace three
