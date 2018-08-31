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

namespace open3d{

void LineSet::Clear()
{
    points_.clear();
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
    auto itr_x = std::min_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::min_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::min_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d LineSet::GetMaxBound() const
{
    if (!HasPoints()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::max_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::max_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::max_element(points_.begin(), points_.end(),
        [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

void LineSet::Transform(const Eigen::Matrix4d &transformation)
{
    for (auto &point : points_) {
        Eigen::Vector4d new_point = transformation * Eigen::Vector4d(
                point(0), point(1), point(2), 1.0);
        point = new_point.block<3, 1>(0, 0);
    }
}

LineSet &LineSet::operator+=(const LineSet &lineset)
{
    if (lineset.IsEmpty()) return (*this);
    size_t old_point_num = points_.size();
    size_t add_point_num = lineset.points_.size();
    size_t new_point_num = old_point_num + add_point_num;
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
    points_.resize(new_point_num);
    for (size_t i = 0; i < add_point_num; i++) {
        points_[old_point_num + i] = lineset.points_[i];
    }
    lines_.resize(new_line_num);
    for (size_t i = 0; i < add_line_num; i++) {
        lines_[old_line_num + i] = Eigen::Vector2i(
                lineset.lines_[i](0) + (int)old_point_num,
                lineset.lines_[i](1) + (int)old_point_num);
    }
    return (*this);
}

LineSet LineSet::operator+(const LineSet &lineset) const
{
    return (LineSet(*this) += lineset);
}

}    // namespace open3d
