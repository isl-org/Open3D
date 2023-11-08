// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/BoundingVolume.h"

#include <Eigen/Eigenvalues>
#include <numeric>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/Qhull.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

OrientedBoundingBox& OrientedBoundingBox::Clear() {
    center_.setZero();
    extent_.setZero();
    R_ = Eigen::Matrix3d::Identity();
    color_.setOnes();
    return *this;
}

bool OrientedBoundingBox::IsEmpty() const { return Volume() <= 0; }

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

OrientedBoundingBox OrientedBoundingBox::GetOrientedBoundingBox(bool) const {
    return *this;
}

OrientedBoundingBox OrientedBoundingBox::GetMinimalOrientedBoundingBox(
        bool) const {
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of an OrientedBoundingBox is not implemented. "
            "Call Translate, Scale, and Rotate.");
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
                                                const Eigen::Vector3d& center) {
    extent_ *= scale;
    center_ = scale * (center_ - center) + center;
    return *this;
}

OrientedBoundingBox& OrientedBoundingBox::Rotate(
        const Eigen::Matrix3d& R, const Eigen::Vector3d& center) {
    R_ = R * R_;
    center_ = R * (center_ - center) + center;
    return *this;
}

double OrientedBoundingBox::Volume() const {
    return extent_(0) * extent_(1) * extent_(2);
}

std::vector<Eigen::Vector3d> OrientedBoundingBox::GetBoxPoints() const {
    Eigen::Vector3d x_axis = R_ * Eigen::Vector3d(extent_(0) / 2, 0, 0);
    Eigen::Vector3d y_axis = R_ * Eigen::Vector3d(0, extent_(1) / 2, 0);
    Eigen::Vector3d z_axis = R_ * Eigen::Vector3d(0, 0, extent_(2) / 2);
    std::vector<Eigen::Vector3d> points(8);
    points[0] = center_ - x_axis - y_axis - z_axis;
    points[1] = center_ + x_axis - y_axis - z_axis;
    points[2] = center_ - x_axis + y_axis - z_axis;
    points[3] = center_ - x_axis - y_axis + z_axis;
    points[4] = center_ + x_axis + y_axis + z_axis;
    points[5] = center_ - x_axis + y_axis + z_axis;
    points[6] = center_ + x_axis - y_axis + z_axis;
    points[7] = center_ + x_axis + y_axis - z_axis;
    return points;
}

std::vector<size_t> OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    Eigen::Vector3d dx = R_ * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d dy = R_ * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d dz = R_ * Eigen::Vector3d(0, 0, 1);
    for (size_t idx = 0; idx < points.size(); idx++) {
        Eigen::Vector3d d = points[idx] - center_;
        if (std::abs(d.dot(dx)) <= extent_(0) / 2 &&
            std::abs(d.dot(dy)) <= extent_(1) / 2 &&
            std::abs(d.dot(dz)) <= extent_(2) / 2) {
            indices.push_back(idx);
        }
    }
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox& aabox) {
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.extent_ = aabox.GetExtent();
    obox.R_ = Eigen::Matrix3d::Identity();
    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    PointCloud hull_pcd;
    std::vector<size_t> hull_point_indices;
    {
        std::shared_ptr<TriangleMesh> mesh;
        std::tie(mesh, hull_point_indices) =
                Qhull::ComputeConvexHull(points, robust);
        hull_pcd.points_ = mesh->vertices_;
    }

    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    std::tie(mean, cov) = hull_pcd.ComputeMeanAndCovariance();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d evals = es.eigenvalues();
    Eigen::Matrix3d R = es.eigenvectors();

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
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) = R.col(0).cross(R.col(1));

    for (size_t i = 0; i < hull_point_indices.size(); ++i) {
        hull_pcd.points_[i] =
                R.transpose() * (points[hull_point_indices[i]] - mean);
    }

    const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();

    OrientedBoundingBox obox;
    obox.center_ = R * aabox.GetCenter() + mean;
    obox.R_ = R;
    obox.extent_ = aabox.GetExtent();

    return obox;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromPointsMinimal(
        const std::vector<Eigen::Vector3d>& points, bool robust) {
    std::shared_ptr<TriangleMesh> mesh;
    std::tie(mesh, std::ignore) = Qhull::ComputeConvexHull(points, robust);
    double min_vol = -1;
    OrientedBoundingBox min_box;
    PointCloud hull_pcd;
    for (auto& tri : mesh->triangles_) {
        hull_pcd.points_ = mesh->vertices_;
        Eigen::Vector3d a = mesh->vertices_[tri(0)];
        Eigen::Vector3d b = mesh->vertices_[tri(1)];
        Eigen::Vector3d c = mesh->vertices_[tri(2)];
        Eigen::Vector3d u = b - a;
        Eigen::Vector3d v = c - a;
        Eigen::Vector3d w = u.cross(v);
        v = w.cross(u);
        u = u / u.norm();
        v = v / v.norm();
        w = w / w.norm();
        Eigen::Matrix3d m_rot;
        m_rot << u[0], v[0], w[0], u[1], v[1], w[1], u[2], v[2], w[2];
        hull_pcd.Rotate(m_rot.inverse(), a);

        const auto aabox = hull_pcd.GetAxisAlignedBoundingBox();
        double volume = aabox.Volume();
        if (min_vol == -1. || volume < min_vol) {
            min_vol = volume;
            min_box = aabox.GetOrientedBoundingBox();
            min_box.Rotate(m_rot, a);
        }
    }
    return min_box;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

bool AxisAlignedBoundingBox::IsEmpty() const { return Volume() <= 0; }

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

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

OrientedBoundingBox AxisAlignedBoundingBox::GetMinimalOrientedBoundingBox(
        bool robust) const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const Eigen::Vector3d& min_bound,
                                               const Eigen::Vector3d& max_bound)
    : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
      min_bound_(min_bound),
      max_bound_(max_bound),
      color_(1, 1, 1) {
    if ((max_bound_.array() < min_bound_.array()).any()) {
        open3d::utility::LogWarning(
                "max_bound {} of bounding box is smaller than min_bound {} in "
                "one or more axes. Fix input values to remove this warning.",
                max_bound_, min_bound_);
        max_bound_ = max_bound.cwiseMax(min_bound);
        min_bound_ = max_bound.cwiseMin(min_bound);
    }
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4d& transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be axis "
            "aligned anymore, convert it to a OrientedBoundingBox first");
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Translate(
        const Eigen::Vector3d& translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Vector3d half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Scale(
        const double scale, const Eigen::Vector3d& center) {
    min_bound_ = center + scale * (min_bound_ - center);
    max_bound_ = center + scale * (max_bound_ - center);
    return *this;
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::Rotate(
        const Eigen::Matrix3d& rotation, const Eigen::Vector3d& center) {
    utility::LogError(
            "A rotation of a AxisAlignedBoundingBox would not be axis aligned "
            "anymore, convert it to an OrientedBoundingBox first");
    return *this;
}

std::string AxisAlignedBoundingBox::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

AxisAlignedBoundingBox& AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox& other) {
    if (IsEmpty()) {
        min_bound_ = other.min_bound_;
        max_bound_ = other.max_bound_;
    } else if (!other.IsEmpty()) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
    }
    return *this;
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

double AxisAlignedBoundingBox::Volume() const { return GetExtent().prod(); }

std::vector<Eigen::Vector3d> AxisAlignedBoundingBox::GetBoxPoints() const {
    std::vector<Eigen::Vector3d> points(8);
    Eigen::Vector3d extent = GetExtent();
    points[0] = min_bound_;
    points[1] = min_bound_ + Eigen::Vector3d(extent(0), 0, 0);
    points[2] = min_bound_ + Eigen::Vector3d(0, extent(1), 0);
    points[3] = min_bound_ + Eigen::Vector3d(0, 0, extent(2));
    points[4] = max_bound_;
    points[5] = max_bound_ - Eigen::Vector3d(extent(0), 0, 0);
    points[6] = max_bound_ - Eigen::Vector3d(0, extent(1), 0);
    points[7] = max_bound_ - Eigen::Vector3d(0, 0, extent(2));
    return points;
}

std::vector<size_t> AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox(
        const std::vector<Eigen::Vector3d>& points) const {
    std::vector<size_t> indices;
    for (size_t idx = 0; idx < points.size(); idx++) {
        const auto& point = points[idx];
        if (point(0) >= min_bound_(0) && point(0) <= max_bound_(0) &&
            point(1) >= min_bound_(1) && point(1) <= max_bound_(1) &&
            point(2) >= min_bound_(2) && point(2) <= max_bound_(2)) {
            indices.push_back(idx);
        }
    }
    return indices;
}

}  // namespace geometry
}  // namespace open3d
