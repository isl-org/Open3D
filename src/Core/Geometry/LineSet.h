// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
#include <memory>
#include <Eigen/Core>
#include <Core/Geometry/Geometry3D.h>

namespace three {

class PointCloud;

class LineSet : public Geometry3D
{
public:
	typedef std::pair<int, int> LineSegment;

public:
	LineSet() : Geometry3D(Geometry::GeometryType::LineSet) {}
	~LineSet() override {}

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector3d GetMinBound() const override;
	Eigen::Vector3d GetMaxBound() const override;
	void Transform(const Eigen::Matrix4d &transformation) override;

public:
	LineSet &operator+=(const LineSet &lineset);
	LineSet operator+(const LineSet &lineset) const;

public:
	bool HasPoints() const {
		return point_set_[0].size() > 0 && point_set_[1].size() > 0;
	}

	bool HasLines() const {
		return HasPoints() && lines_.size() > 0;
	}

	bool HasColors() const {
		return HasLines() && colors_.size() == lines_.size();
	}

	std::pair<Eigen::Vector3d, Eigen::Vector3d> GetLineCoordinate(
			size_t i) const {
		return std::make_pair(point_set_[0][lines_[i].first],
				point_set_[1][lines_[i].second]);
	}

public:
	std::vector<Eigen::Vector3d> point_set_[2];
	std::vector<LineSegment> lines_;
	std::vector<Eigen::Vector3d> colors_;
};

/// Factory function to create a lineset from two pointclouds and a
/// correspondence set (LineSetFactory.cpp)
std::shared_ptr<LineSet> CreateLineSetFromPointCloudCorrespondences(
		const PointCloud &cloud0, const PointCloud &cloud1,
		const std::vector<std::pair<int, int>> &correspondences);

}	// namespace three
