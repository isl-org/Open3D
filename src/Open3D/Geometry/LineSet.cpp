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

#include "Open3D/Geometry/LineSet.h"

#include <numeric>

namespace open3d {
namespace geometry {

LineSet &LineSet::Clear() {
    points_.clear();
    lines_.clear();
    colors_.clear();
    return *this;
}

bool LineSet::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3d LineSet::GetMinBound() const {
    if (!HasPoints()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points_.begin(), points_.end(), points_[0],
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a.array().min(b.array()).matrix();
            });
}

Eigen::Vector3d LineSet::GetMaxBound() const {
    if (!HasPoints()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points_.begin(), points_.end(), points_[0],
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a.array().max(b.array()).matrix();
            });
}

LineSet &LineSet::Transform(const Eigen::Matrix4d &transformation) {
    for (auto &point : points_) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.block<3, 1>(0, 0);
    }
    return *this;
}

LineSet &LineSet::Translate(const Eigen::Vector3d &translation) {
    for (auto &point : points_) {
        point += translation;
    }
    return *this;
}

LineSet &LineSet::Scale(const double scale, bool center) {
    Eigen::Vector3d point_center(0, 0, 0);
    if (center && !points_.empty()) {
        point_center =
                std::accumulate(points_.begin(), points_.end(), point_center);
        point_center /= points_.size();
    }
    for (auto &point : points_) {
        point = (point - point_center) * scale + point_center;
    }
    return *this;
}

LineSet &LineSet::Rotate(const Eigen::Vector3d &rotation,
                         bool center,
                         RotationType type) {
    Eigen::Vector3d point_center(0, 0, 0);
    if (center && !points_.empty()) {
        point_center =
                std::accumulate(points_.begin(), points_.end(), point_center);
        point_center /= points_.size();
    }
    const Eigen::Matrix3d R = GetRotationMatrix(rotation, type);
    for (auto &point : points_) {
        point = R * (point - point_center) + point_center;
    }
    return *this;
}

LineSet &LineSet::operator+=(const LineSet &lineset) {
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
        lines_[old_line_num + i] =
                Eigen::Vector2i(lineset.lines_[i](0) + (int)old_point_num,
                                lineset.lines_[i](1) + (int)old_point_num);
    }
    return (*this);
}

LineSet LineSet::operator+(const LineSet &lineset) const {
    return (LineSet(*this) += lineset);
}

}  // namespace geometry
}  // namespace open3d
