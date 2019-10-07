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
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/TriangleMesh.h"

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
    return ComputeMinBound(points_);
}

Eigen::Vector3d PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d PointCloud::GetCenter() const { return ComputeCenter(points_); }

AxisAlignedBoundingBox PointCloud::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
}

OrientedBoundingBox PointCloud::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(points_);
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    TransformNormals(transformation, normals_);
    return *this;
}

PointCloud &PointCloud::Translate(const Eigen::Vector3d &translation,
                                  bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

PointCloud &PointCloud::Scale(const double scale, bool center) {
    ScalePoints(scale, points_, center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Eigen::Matrix3d &R, bool center) {
    RotatePoints(R, points_, center);
    RotateNormals(R, normals_, center);
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
            utility::LogDebug(
                    "[ComputePointCloudToPointCloudDistance] Found a point "
                    "without neighbors.\n");
            distances[i] = 0.0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    return distances;
}

PointCloud &PointCloud::RemoveNoneFinitePoints(bool remove_nan,
                                               bool remove_infinite) {
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    size_t old_point_num = points_.size();
    size_t k = 0;                                 // new index
    for (size_t i = 0; i < old_point_num; i++) {  // old index
        bool is_nan = remove_nan &&
                      (std::isnan(points_[i](0)) || std::isnan(points_[i](1)) ||
                       std::isnan(points_[i](2)));
        bool is_infinite = remove_infinite && (std::isinf(points_[i](0)) ||
                                               std::isinf(points_[i](1)) ||
                                               std::isinf(points_[i](2)));
        if (!is_nan && !is_infinite) {
            points_[k] = points_[i];
            if (has_normal) normals_[k] = normals_[i];
            if (has_color) colors_[k] = colors_[i];
            k++;
        }
    }
    points_.resize(k);
    if (has_normal) normals_.resize(k);
    if (has_color) colors_.resize(k);
    utility::LogDebug(
            "[RemoveNoneFinitePoints] {:d} nan points have been removed.\n",
            (int)(old_point_num - k));
    return *this;
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
            utility::LogDebug(
                    "[ComputePointCloudNearestNeighborDistance] Found a point "
                    "without neighbors.\n");
            nn_dis[i] = 0.0;
        } else {
            nn_dis[i] = std::sqrt(dists[1]);
        }
    }
    return nn_dis;
}

std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
PointCloud::ComputeConvexHull() const {
    return Qhull::ComputeConvexHull(points_);
}

std::tuple<std::shared_ptr<TriangleMesh>, std::vector<size_t>>
PointCloud::HiddenPointRemoval(const Eigen::Vector3d &camera_location,
                               const double radius) const {
    if (radius <= 0) {
        utility::LogWarning(
                "[HiddenPointRemoval] radius must be larger than zero.\n");
        return std::make_tuple(std::make_shared<TriangleMesh>(),
                               std::vector<size_t>());
    }

    // perform spherical projection
    std::vector<Eigen::Vector3d> spherical_projection;
    for (size_t pidx = 0; pidx < points_.size(); ++pidx) {
        Eigen::Vector3d projected_point = points_[pidx] - camera_location;
        double norm = projected_point.norm();
        spherical_projection.push_back(
                projected_point + 2 * (radius - norm) * projected_point / norm);
    }

    // add origin
    size_t origin_pidx = spherical_projection.size();
    spherical_projection.push_back(Eigen::Vector3d(0, 0, 0));

    // calculate convex hull of spherical projection
    std::shared_ptr<TriangleMesh> visible_mesh;
    std::vector<size_t> pt_map;
    std::tie(visible_mesh, pt_map) =
            Qhull::ComputeConvexHull(spherical_projection);

    // reassign original points to mesh
    int origin_vidx = pt_map.size();
    for (size_t vidx = 0; vidx < pt_map.size(); vidx++) {
        size_t pidx = pt_map[vidx];
        visible_mesh->vertices_[vidx] = points_[pidx];
        if (pidx == origin_pidx) {
            origin_vidx = vidx;
            visible_mesh->vertices_[vidx] = camera_location;
        }
    }

    // erase origin if part of mesh
    if (origin_vidx < (int)(visible_mesh->vertices_.size())) {
        visible_mesh->vertices_.erase(visible_mesh->vertices_.begin() +
                                      origin_vidx);
        pt_map.erase(pt_map.begin() + origin_vidx);
        for (size_t tidx = visible_mesh->triangles_.size(); tidx-- > 0;) {
            if (visible_mesh->triangles_[tidx](0) == origin_vidx ||
                visible_mesh->triangles_[tidx](1) == origin_vidx ||
                visible_mesh->triangles_[tidx](2) == origin_vidx) {
                visible_mesh->triangles_.erase(
                        visible_mesh->triangles_.begin() + tidx);
            } else {
                if (visible_mesh->triangles_[tidx](0) > origin_vidx)
                    visible_mesh->triangles_[tidx](0) -= 1;
                if (visible_mesh->triangles_[tidx](1) > origin_vidx)
                    visible_mesh->triangles_[tidx](1) -= 1;
                if (visible_mesh->triangles_[tidx](2) > origin_vidx)
                    visible_mesh->triangles_[tidx](2) -= 1;
            }
        }
    }
    return std::make_tuple(visible_mesh, pt_map);
}

}  // namespace geometry
}  // namespace open3d
