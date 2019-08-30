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
#include "Open3D/Geometry/BoundingVolume.h"

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
    return ComputeMinBound(points_);
}

Eigen::Vector3d LineSet::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d LineSet::GetCenter() const { return ComputeCenter(points_); }

AxisAlignedBoundingBox LineSet::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
}

OrientedBoundingBox LineSet::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(points_);
}

LineSet &LineSet::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet &LineSet::Translate(const Eigen::Vector3d &translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet &LineSet::Scale(const double scale, bool center) {
    ScalePoints(scale, points_, center);
    return *this;
}

LineSet &LineSet::Rotate(const Eigen::Vector3d &rotation,
                         bool center,
                         RotationType type) {
    RotatePoints(rotation, points_, center, type);
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
