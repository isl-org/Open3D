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

#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/Qhull.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

#include <numeric>

#include <Eigen/Eigenvalues>

namespace open3d {
namespace geometry {

OrientedBoundingBox& OrientedBoundingBox::Clear() {
    center_.setZero();
    x_axis_.setZero();
    y_axis_.setZero();
    z_axis_.setZero();
    return *this;
}

bool OrientedBoundingBox::IsEmpty() const { return Volume() == 0; }

Eigen::Vector3d OrientedBoundingBox::GetMinBound() const {
    auto points = GetBoxPoints();
    return ComputeMinBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetMaxBound() const {
    auto points = GetBoxPoints();
    return ComputeMaxBound(points);
}

Eigen::Vector3d OrientedBoundingBox::GetCenter() const { return center_; }

AxisAlignedBoundingBox OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(GetBoxPoints());
}

OrientedBoundingBox OrientedBoundingBox::GetOrientedBoundingBox() const {
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    Eigen::Vector4d c;
    c << center_, 1;
    Eigen::Vector4d x;
    x << center_ + x_axis_, 1;
    Eigen::Vector4d y;
    y << center_ + y_axis_, 1;
    Eigen::Vector4d z;
    z << center_ + z_axis_, 1;
    c = transformation * c;
    x = transformation * x;
    y = transformation * y;
    z = transformation * z;
    center_ = c.head<3>() / c(3);
    x_axis_ = x.head<3>() / x(3) - center_;
    y_axis_ = y.head<3>() / y(3) - center_;
    z_axis_ = z.head<3>() / z(3) - center_;
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Scale(const double scale,
                                                bool center) {
    if (center) {
        x_axis_ *= scale;
        y_axis_ *= scale;
        z_axis_ *= scale;
    } else {
        Eigen::Vector3d x = scale * (center_ + x_axis_);
        Eigen::Vector3d y = scale * (center_ + y_axis_);
        Eigen::Vector3d z = scale * (center_ + z_axis_);
        center_ *= scale;
        x_axis_ = x - center_;
        y_axis_ = y - center_;
        z_axis_ = z - center_;
    }
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Rotate(
        const Eigen::Vector3d& rotation, bool center, RotationType type) {
    const Eigen::Matrix3d R = GetRotationMatrix(rotation, type);
    if (center) {
        x_axis_ = R * x_axis_;
        y_axis_ = R * y_axis_;
        z_axis_ = R * z_axis_;
    } else {
        Eigen::Vector3d x = R * (center_ + x_axis_);
        Eigen::Vector3d y = R * (center_ + y_axis_);
        Eigen::Vector3d z = R * (center_ + z_axis_);
        center_ = R * center_;
        x_axis_ = x - center_;
        y_axis_ = y - center_;
        z_axis_ = z - center_;
    }
    return *this;
}

double OrientedBoundingBox::Volume() const {
    return (2 * x_axis_.norm()) * (2 * y_axis_.norm()) * (2 * z_axis_.norm());
}

std::vector<Eigen::Vector3d> OrientedBoundingBox::GetBoxPoints() const {
    std::vector<Eigen::Vector3d> points(8);
    points[0] = center_ - x_axis_ - y_axis_ - z_axis_;
    points[1] = center_ + x_axis_ - y_axis_ - z_axis_;
    points[2] = center_ - x_axis_ + y_axis_ - z_axis_;
    points[3] = center_ - x_axis_ - y_axis_ + z_axis_;
    points[4] = center_ + x_axis_ + y_axis_ + z_axis_;
    points[5] = center_ - x_axis_ + y_axis_ + z_axis_;
    points[6] = center_ + x_axis_ - y_axis_ + z_axis_;
    points[7] = center_ + x_axis_ + y_axis_ - z_axis_;
    return points;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox& aabox) {
    Eigen::Vector3d half_extend = aabox.GetHalfExtend();
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.x_axis_ << half_extend(0), 0, 0;
    obox.y_axis_ << 0, half_extend(1), 0;
    obox.z_axis_ << 0, 0, half_extend(2);
    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points) {
    PointCloud hull_pcd;
    hull_pcd.points_ = Qhull::ComputeConvexHull(points)->vertices_;

    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = hull_pcd.ComputeMeanAndCovariance();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d evals = es.eigenvalues();
    Eigen::Matrix3d R = es.eigenvectors();
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) /= R.col(2).norm();

    if (evals(1) > evals(0)) {
        std::swap(evals(1), evals(0));
        Eigen::Vector3d tmp = R.col(1);
        R.col(1) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(0)) {
        std::swap(evals(2), evals(0));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(0);
        R.col(0) = tmp;
    }
    if (evals(2) > evals(1)) {
        std::swap(evals(2), evals(1));
        Eigen::Vector3d tmp = R.col(2);
        R.col(2) = R.col(1);
        R.col(1) = tmp;
    }

    for (auto& pt : hull_pcd.points_) {
        pt = R.transpose() * (pt - mean);
    }
    const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
    const Eigen::Vector3d half_extend = aabox.GetHalfExtend();

    OrientedBoundingBox obox;
    obox.center_ = R * aabox.GetCenter() + mean;
    obox.x_axis_ = R * Eigen::Vector3d(half_extend(0), 0, 0);
    obox.y_axis_ = R * Eigen::Vector3d(0, half_extend(1), 0);
    obox.z_axis_ = R * Eigen::Vector3d(0, 0, half_extend(2));

    return obox;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

bool AxisAlignedBoundingBox::IsEmpty() const { return Volume() == 0; }
Eigen::Vector3d AxisAlignedBoundingBox::GetMinBound() const {
    return min_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetMaxBound() const {
    return max_bound_;
}

Eigen::Vector3d AxisAlignedBoundingBox::GetCenter() const {
    return (min_bound_ + max_bound_) * 0.5;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::GetAxisAlignedBoundingBox()
        const {
    return *this;
}

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogWarning(
            "A general transform of a AxisAlignedBoundingBox would not be axis "
            "aligned anymore, convert it to a OrientedBoundingBox first\n");
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Vector3d half_extend = GetHalfExtend();
        min_bound_ = translation - half_extend;
        max_bound_ = translation + half_extend;
    }
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Scale(const double scale,
                                                      bool center) {
    if (center) {
        Eigen::Vector3d center = GetCenter();
        min_bound_ = center + scale * (min_bound_ - center);
        max_bound_ = center + scale * (max_bound_ - center);
    } else {
        min_bound_ *= scale;
        max_bound_ *= scale;
    }
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Rotate(
        const Eigen::Vector3d& rotation, bool center, RotationType type) {
    utility::LogWarning(
            "A rotation of a AxisAlignedBoundingBox would not be axis aligned "
            "anymore, convert it to a OrientedBoundingBox first\n");
    return *this;
}

std::string AxisAlignedBoundingBox::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points) {
    AxisAlignedBoundingBox box;
    if (points.empty()) {
        box.min_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
        box.max_bound_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    } else {
        box.min_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().min(b.array()).matrix();
                });
        box.max_bound_ = std::accumulate(
                points.begin(), points.end(), points[0],
                [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                    return a.array().max(b.array()).matrix();
                });
    }
    return box;
}

double AxisAlignedBoundingBox::Volume() const { return GetExtend().prod(); }

std::vector<Eigen::Vector3d> AxisAlignedBoundingBox::GetBoxPoints() const {
    std::vector<Eigen::Vector3d> points(8);
    Eigen::Vector3d extend = GetExtend();
    points[0] = min_bound_;
    points[1] = min_bound_ + Eigen::Vector3d(extend(0), 0, 0);
    points[2] = min_bound_ + Eigen::Vector3d(0, extend(1), 0);
    points[3] = min_bound_ + Eigen::Vector3d(0, 0, extend(2));
    points[4] = max_bound_;
    points[5] = max_bound_ - Eigen::Vector3d(extend(0), 0, 0);
    points[6] = max_bound_ - Eigen::Vector3d(0, extend(1), 0);
    points[7] = max_bound_ - Eigen::Vector3d(0, 0, extend(2));
    return points;
}

}  // namespace geometry
}  // namespace open3d
