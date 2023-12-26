// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/LineSet.h"

#include <numeric>

#include "open3d/geometry/BoundingVolume.h"

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

OrientedBoundingBox LineSet::GetOrientedBoundingBox(bool robust) const {
    return OrientedBoundingBox::CreateFromPoints(points_, robust);
}

OrientedBoundingBox LineSet::GetMinimalOrientedBoundingBox(bool robust) const {
    return OrientedBoundingBox::CreateFromPointsMinimal(points_, robust);
}

LineSet &LineSet::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet &LineSet::Translate(const Eigen::Vector3d &translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet &LineSet::Scale(const double scale, const Eigen::Vector3d &center) {
    ScalePoints(scale, points_, center);
    return *this;
}

LineSet &LineSet::Rotate(const Eigen::Matrix3d &R,
                         const Eigen::Vector3d &center) {
    RotatePoints(R, points_, center);
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
