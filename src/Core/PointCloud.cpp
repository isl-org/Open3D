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

#include "PointCloud.h"

namespace three{

PointCloud::PointCloud()
{
	SetGeometryType(GEOMETRY_POINTCLOUD);
}

PointCloud::~PointCloud()
{
}
	
bool PointCloud::CloneFrom(const Geometry &reference)
{
	if (reference.GetGeometryType() != GetGeometryType()) {
		// always return when the types do not match
		return false;
	}
	
	Clear();
	const PointCloud &pointcloud = static_cast<const PointCloud &>(reference);
	points_.resize(pointcloud.points_.size());
	for (size_t i = 0; i < pointcloud.points_.size(); i++) {
		points_[i] = pointcloud.points_[i];
	}
	normals_.resize(pointcloud.normals_.size());
	for (size_t i = 0; i < pointcloud.normals_.size(); i++) {
		normals_[i] = pointcloud.normals_[i];
	}
	colors_.resize(pointcloud.colors_.size());
	for (size_t i = 0; i < pointcloud.colors_.size(); i++) {
		colors_[i] = pointcloud.colors_[i];
	}
	return true;
}

Eigen::Vector3d PointCloud::GetMinBound() const
{
	if (!HasPoints()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::min_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(0) < b(0); });
	auto itr_y = std::min_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(1) < b(1); });
	auto itr_z = std::min_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d PointCloud::GetMaxBound() const
{
	if (!HasPoints()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x = std::max_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(0) < b(0); });
	auto itr_y = std::max_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(1) < b(1); });
	auto itr_z = std::max_element(points_.begin(), points_.end(),
		[](Eigen::Vector3d a, Eigen::Vector3d b) { return a(2) < b(2); });
	return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}
	
void PointCloud::Clear()
{
	points_.clear();
	normals_.clear();
	colors_.clear();
}
	
bool PointCloud::IsEmpty() const
{
	return !HasPoints();
}
	
void PointCloud::Transform(const Eigen::Matrix4d &transformation)
{
	for (size_t i = 0; i < points_.size(); i++) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				points_[i](0), points_[i](1), points_[i](2), 1.0);
		points_[i] = new_point.block<3, 1>(0, 0);
	}
	
	for (size_t i = 0; i < normals_.size(); i++) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				normals_[i](0), normals_[i](1), normals_[i](2), 0.0);
		normals_[i] = new_normal.block<3, 1>(0, 0);
	}
}

}	// namespace three
