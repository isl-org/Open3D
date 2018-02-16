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

/// A utility class to store picked points of a pointcloud
class PointCloudPicker : public Geometry3D
{
public:
	PointCloudPicker() : Geometry3D(Geometry::GeometryType::Unspecified) {}
	~PointCloudPicker() override {}

public:
	void Clear() override;
	bool IsEmpty() const override;
	Eigen::Vector3d GetMinBound() const final;
	Eigen::Vector3d GetMaxBound() const final;
	void Transform(const Eigen::Matrix4d & transformation) override;
	bool SetPointCloud(std::shared_ptr<const Geometry> ptr);

public:
	std::shared_ptr<const Geometry> pointcloud_ptr_;
	std::vector<size_t> picked_indices_;
};

}	// namespace three
