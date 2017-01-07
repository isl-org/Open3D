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

#include <Core/Geometry/KDTreeFlann.h>

namespace three{

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
	
void PointCloud::Transform(const Eigen::Matrix4d &transformation)
{
	for (auto &point : points_) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				point(0), point(1), point(2), 1.0);
		point = new_point.block<3, 1>(0, 0);
	}
	for (auto &normal : normals_) {
		Eigen::Vector4d new_normal = transformation * Eigen::Vector4d(
				normal(0), normal(1), normal(2), 0.0);
		normal = new_normal.block<3, 1>(0, 0);
	}
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud)
{
	// We do not use std::vector::insert to combine std::vector because it will
	// crash if the pointcloud is added to itself.
	size_t old_vert_num = points_.size();
	size_t add_vert_num = cloud.points_.size();
	size_t new_vert_num = old_vert_num + add_vert_num;
	if (HasNormals() && cloud.HasNormals()) {
		normals_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			normals_[old_vert_num + i] = cloud.normals_[i];
	}
	if (HasColors() && cloud.HasColors()) {
		colors_.resize(new_vert_num);
		for (size_t i = 0; i < add_vert_num; i++)
			colors_[old_vert_num + i] = cloud.colors_[i];
	}
	points_.resize(new_vert_num);
	for (size_t i = 0; i < add_vert_num; i++)
		points_[old_vert_num + i] = cloud.points_[i];
	return (*this);
}

const PointCloud PointCloud::operator+(const PointCloud &cloud)
{
	return (PointCloud(*this) += cloud);
}

void ComputePointCloudToPointCloudDistance(const PointCloud &source,
		const PointCloud &target, std::vector<double> &distances)
{
	distances.resize(source.points_.size());
	KDTreeFlann kdtree;
	kdtree.SetGeometry(target);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < (int)source.points_.size(); i++) {
		std::vector<int> indices(1);
		std::vector<double> dists(1);
		kdtree.SearchKNN(source.points_[i], 1, indices, dists);
		distances[i] = std::sqrt(dists[0]);
	}
}

}	// namespace three
