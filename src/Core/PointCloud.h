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

class PointCloud
{
public:
	PointCloud(void);
	~PointCloud(void);

public:
	void CloneFrom(const PointCloud &reference);
	Eigen::Vector3d GetMinBound() const;
	Eigen::Vector3d GetMaxBound() const;

public:
	bool HasPoints() const {
		return points_.size() > 0;
	}

	bool HasNormals() const {
		return points_.size() > 0 && normals_.size() == points_.size();
	}

	bool HasColors() const {
		return points_.size() > 0 && colors_.size() == points_.size();
	}

	void Clear() { points_.clear(); normals_.clear(); colors_.clear(); }
	
	void NormalizeNormal() {
		for (size_t i = 0; i < normals_.size(); i++) {
			normals_[i].normalize();
		}
	}
	
public:
	std::vector<Eigen::Vector3d> points_;
	std::vector<Eigen::Vector3d> normals_;
	std::vector<Eigen::Vector3d> colors_;
};

}	// namespace three
