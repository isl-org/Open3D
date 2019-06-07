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

#include "Open3D/Geometry/PointCloud.h"

#include <Eigen/Dense>
#include <numeric>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/Qhull.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3d PointCloud::GetMinBound() const {
    if (!HasPoints()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points_.begin(), points_.end(), points_[0],
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a.array().min(b.array()).matrix();
            });
}

Eigen::Vector3d PointCloud::GetMaxBound() const {
    if (!HasPoints()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points_.begin(), points_.end(), points_[0],
            [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
                return a.array().max(b.array()).matrix();
            });
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4d &transformation) {
    for (auto &point : points_) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.block<3, 1>(0, 0);
    }
    for (auto &normal : normals_) {
        Eigen::Vector4d new_normal =
                transformation *
                Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
        normal = new_normal.block<3, 1>(0, 0);
    }
    return *this;
}

PointCloud &PointCloud::Translate(const Eigen::Vector3d &translation) {
    for (auto &point : points_) {
        point += translation;
    }
    return *this;
}

PointCloud &PointCloud::Scale(const double scale, bool center) {
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

PointCloud &PointCloud::Rotate(const Eigen::Vector3d &rotation,
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
    for (auto &normal : normals_) {
        normal = R * normal;
    }
    return *this;
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            normals_[old_vert_num + i] = cloud.normals_[i];
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            colors_[old_vert_num + i] = cloud.colors_[i];
    } else {
        colors_.clear();
    }
    points_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        points_[old_vert_num + i] = cloud.points_[i];
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

std::vector<double> PointCloud::ComputePointCloudDistance(
        const PointCloud &target) {
    std::vector<double> distances(points_.size());
    KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)points_.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        if (kdtree.SearchKNN(points_[i], 1, indices, dists) == 0) {
            utility::PrintDebug(
                    "[ComputePointCloudToPointCloudDistance] Found a point "
                    "without neighbors.\n");
            distances[i] = 0.0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    return distances;
}

std::tuple<Eigen::Vector3d, Eigen::Matrix3d>
PointCloud::ComputeMeanAndCovariance() const {
    if (IsEmpty()) {
        return std::make_tuple(Eigen::Vector3d::Zero(),
                               Eigen::Matrix3d::Identity());
    }
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto &point : points_) {
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)points_.size();
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    mean(0) = cumulants(0);
    mean(1) = cumulants(1);
    mean(2) = cumulants(2);
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return std::make_tuple(mean, covariance);
}

std::vector<double> PointCloud::ComputeMahalanobisDistance() const {
    std::vector<double> mahalanobis(points_.size());
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    std::tie(mean, covariance) = ComputeMeanAndCovariance();
    Eigen::Matrix3d cov_inv = covariance.inverse();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)points_.size(); i++) {
        Eigen::Vector3d p = points_[i] - mean;
        mahalanobis[i] = std::sqrt(p.transpose() * cov_inv * p);
    }
    return mahalanobis;
}

std::vector<double> PointCloud::ComputeNearestNeighborDistance() const {
    std::vector<double> nn_dis(points_.size());
    KDTreeFlann kdtree(*this);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < (int)points_.size(); i++) {
        std::vector<int> indices(2);
        std::vector<double> dists(2);
        if (kdtree.SearchKNN(points_[i], 2, indices, dists) <= 1) {
            utility::PrintDebug(
                    "[ComputePointCloudNearestNeighborDistance] Found a point "
                    "without neighbors.\n");
            nn_dis[i] = 0.0;
        } else {
            nn_dis[i] = std::sqrt(dists[1]);
        }
    }
    return nn_dis;
}

std::shared_ptr<TriangleMesh> PointCloud::ComputeConvexHull() const {
    return Qhull::ComputeConvexHull(points_);
}

}  // namespace geometry
}  // namespace open3d
