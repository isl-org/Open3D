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

#include "LineSet.h"

namespace three{

void LineSet::Clear()
{
	point_set_[0].clear();
	point_set_[1].clear();
	lines_.clear();
	colors_.clear();
}

bool LineSet::IsEmpty() const
{
	return !HasPoints();
}

Eigen::Vector3d LineSet::GetMinBound() const
{
	if (!HasPoints()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x0 = std::min_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y0 = std::min_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z0 = std::min_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	auto itr_x1 = std::min_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y1 = std::min_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z1 = std::min_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	return Eigen::Vector3d(
			std::min((*itr_x0)(0), (*itr_x1)(0)),
			std::min((*itr_y0)(1), (*itr_y1)(1)),
			std::min((*itr_z0)(2), (*itr_z1)(2)));
}

Eigen::Vector3d LineSet::GetMaxBound() const
{
	if (!HasPoints()) {
		return Eigen::Vector3d(0.0, 0.0, 0.0);
	}
	auto itr_x0 = std::max_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y0 = std::max_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z0 = std::max_element(point_set_[0].begin(), point_set_[0].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	auto itr_x1 = std::max_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
	auto itr_y1 = std::max_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
	auto itr_z1 = std::max_element(point_set_[1].begin(), point_set_[1].end(),
		[](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
	return Eigen::Vector3d(
			std::max((*itr_x0)(0), (*itr_x1)(0)),
			std::max((*itr_y0)(1), (*itr_y1)(1)),
			std::max((*itr_z0)(2), (*itr_z1)(2)));
}

void LineSet::Transform(const Eigen::Matrix4d &transformation)
{
	for (auto &point : point_set_[0]) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				point(0), point(1), point(2), 1.0);
		point = new_point.block<3, 1>(0, 0);
	}
	for (auto &point : point_set_[1]) {
		Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
				point(0), point(1), point(2), 1.0);
		point = new_point.block<3, 1>(0, 0);
	}
}

LineSet &LineSet::operator+=(const LineSet &lineset)
{
	if (lineset.IsEmpty()) return (*this);
	size_t old_point_num[2] = {point_set_[0].size(), point_set_[1].size()};
	size_t add_point_num[2] = {lineset.point_set_[0].size(),
			lineset.point_set_[1].size()};
	size_t new_point_num[2] = {old_point_num[0] + add_point_num[0],
			old_point_num[1] + add_point_num[1]};
	size_t old_line_num = lines_.size();
	size_t add_line_num = lineset.lines_.size();
	size_t new_line_num = old_line_num + add_line_num;

	if ((!HasLines() || HasColors()) && lineset.HasColors()) {
		colors_.resize(new_line_num);
		for (size_t i = 0; i < add_line_num; i++) {
			colors_[old_line_num + i] = lineset.colors_[i];
		}
	} else {
		colors_.clear();
	}
	point_set_[0].resize(new_point_num[0]);
	for (size_t i = 0; i < add_point_num[0]; i++) {
		point_set_[0][old_point_num[0] + i] = lineset.point_set_[0][i];
	}
	point_set_[1].resize(new_point_num[1]);
	for (size_t i = 0; i < add_point_num[1]; i++) {
		point_set_[1][old_point_num[1] + i] = lineset.point_set_[1][i];
	}
	lines_.resize(new_line_num);
	for (size_t i = 0; i < add_line_num; i++) {
		lines_[old_line_num + i] = std::make_pair(
				lineset.lines_[i].first + (int)old_point_num[0],
				lineset.lines_[i].second + (int)old_point_num[1]);
	}
	return (*this);
}

LineSet LineSet::operator+(const LineSet &lineset) const
{
	return (LineSet(*this) += lineset);
}

}	// namespace three
