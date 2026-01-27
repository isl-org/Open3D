// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Eigenvalues>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace geometry {
namespace {
Eigen::Vector3d ComputeCentroid(const std::vector<Eigen::Vector3d>& points,
                                const std::vector<int>& indices) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if (indices.empty()) {
        return centroid;
    }

    const int n_points = static_cast<int>(points.size());
    for (int idx : indices) {
        if (idx < 0 || idx >= n_points) {
            continue;
        }
        centroid += points[idx];
    }
    return centroid / static_cast<double>(indices.size());
}

Eigen::Vector3d ComputeWeightedCentroid(const open3d::geometry::PointCloud& pcd,
                                        const std::vector<int>& indices,
                                        const std::vector<double>& weights) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();

    if (indices.size() != weights.size() || indices.empty()) {
        return centroid;
    }

    double sum_w = 0.0;
    const int n_points = static_cast<int>(pcd.points_.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0 || idx >= n_points) {
            continue;
        }

        const double w = weights[i];
        centroid += w * pcd.points_[idx];
        sum_w += w;
    }

    if (sum_w > 0.0) {
        centroid /= sum_w;
    }

    return centroid;
}

/// \brief Compute weighted (unnormalized) covariance matrix around a centroid.
///
/// Computes:
///     C = sum_i w_i * (p_i - centroid) * (p_i - centroid)^T
///
/// Notes:
/// - The covariance matrix is NOT normalized by the sum of weights.
/// - Invalid indices are skipped.
/// - If input sizes mismatch or no valid points exist, a zero matrix is
/// returned.
Eigen::Matrix3d ComputeWeightedCovariance(
        const open3d::geometry::PointCloud& pcd,
        const std::vector<int>& indices,
        const std::vector<double>& weights,
        const Eigen::Vector3d& centroid) {
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();

    // ----------------------------
    // 1. Validate input sizes
    // ----------------------------
    if (indices.size() != weights.size() || indices.empty()) {
        return C;
    }

    const int n_points = static_cast<int>(pcd.points_.size());

    // ----------------------------
    // 2. Accumulate covariance
    // ----------------------------
    for (size_t i = 0; i < indices.size(); ++i) {
        const int idx = indices[i];

        // Skip invalid indices
        if (idx < 0 || idx >= n_points) {
            continue;
        }

        const Eigen::Vector3d diff = pcd.points_[idx] - centroid;
        C.noalias() += weights[i] * diff * diff.transpose();
    }

    return C;
}

Eigen::Vector3d ProjectOntoPlane(const Eigen::Vector3d& p,
                                 const Eigen::Vector3d& centroid,
                                 const Eigen::Vector3d& normal) {
    return p - normal * ((p - centroid).dot(normal));
}

}  // namespace

PointCloud PointCloud::SmoothMLS(const KDTreeSearchParam& search_param) const {
    if (points_.empty()) {
        utility::LogWarning(
                "PointCloud::SmoothMLS called on empty point cloud.");
        return PointCloud();
    }

    double radius = 0.0;
    int k = -1;

    // Extract parameters from search_param
    if (auto* r = dynamic_cast<const KDTreeSearchParamRadius*>(&search_param)) {
        radius = r->radius_;
    } else if (auto* knn = dynamic_cast<const KDTreeSearchParamKNN*>(
                       &search_param)) {
        k = knn->knn_;
    } else if (auto* hybrid = dynamic_cast<const KDTreeSearchParamHybrid*>(
                       &search_param)) {
        radius = hybrid->radius_;
        k = hybrid->max_nn_;
    } else {
        utility::LogError("Unsupported search param type.");
    }

    // ----------------------------
    // 1. Parameter validation
    // ----------------------------
    if (radius <= 0.0 && k <= 0.0) {
        utility::LogWarning("Both radius and k are invalid. Returning copy.");
        return *this;
    }

    // ----------------------------
    // 2. Prepare output cloud
    // ----------------------------
    PointCloud smoothed_cloud;
    smoothed_cloud.points_.resize(points_.size());

    const bool has_normals = HasNormals();
    if (has_normals) {
        smoothed_cloud.normals_.resize(normals_.size());
    }

    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);

    // Precompute Gaussian factor once
    double inv_sigma2 = (radius > 0.0) ? 1.0 / (radius * radius) : 0.0;

#pragma omp parallel for schedule(static, 64) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < (int)points_.size(); i++) {
        const Eigen::Vector3d& p = points_[i];

        std::vector<int> indices;
        std::vector<double> distances2;
        int nb_neighbors = 0;

        // ----------------------------
        // 3. Neighborhood search
        // ----------------------------
        if (radius > 0.0 && k > 0) {
            // hybrid: radius + max_nn
            nb_neighbors =
                    kdtree.SearchHybrid(p, radius, k, indices, distances2);
        } else if (k > 0) {
            nb_neighbors = kdtree.SearchKNN(p, k, indices, distances2);
        } else {
            nb_neighbors = kdtree.SearchRadius(p, radius, indices, distances2);
        }

        // ----------------------------
        // 4. Not enough neighbors
        // ----------------------------
        if (nb_neighbors < 3) {
            smoothed_cloud.points_[i] = p;
            if (has_normals) {
                smoothed_cloud.normals_[i] = normals_[i].normalized();
            }
            continue;
        }

        // ----------------------------
        // 5. Compute weights
        // ----------------------------
        std::vector<double> w(nb_neighbors);
        for (int j = 0; j < nb_neighbors; ++j) {
            w[j] = std::exp(-distances2[j] * inv_sigma2);
        }

        // ----------------------------
        // 6. Centroid + covariance
        // ----------------------------
        Eigen::Vector3d centroid = ComputeWeightedCentroid(*this, indices, w);
        Eigen::Matrix3d C =
                ComputeWeightedCovariance(*this, indices, w, centroid);

        // ----------------------------
        // 7. PCA -> normal
        // ----------------------------
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
        Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();

        // ----------------------------
        // 8. Project onto tangent plane
        // ----------------------------
        smoothed_cloud.points_[i] = ProjectOntoPlane(p, centroid, normal);

        if (has_normals) {
            smoothed_cloud.normals_[i] = normal;
        }
    }

    // ----------------------------
    // 9. Make normals consistent
    // ----------------------------
    if (has_normals) {
        smoothed_cloud.OrientNormalsConsistentTangentPlane(30, 0.0, 1.0);
    }

    return smoothed_cloud;
}

PointCloud PointCloud::SmoothLaplacian(size_t iterations,
                                       double lambda,
                                       int knn) const {
    if (points_.empty() || iterations == 0) {
        return *this;
    }

    PointCloud smoothed_cloud = *this;
    int n_points = static_cast<int>(smoothed_cloud.points_.size());

    for (size_t i = 0; i < iterations; ++i) {
        PointCloud temp_cloud = smoothed_cloud;
        KDTreeFlann kdtree(temp_cloud);

        std::vector<Eigen::Vector3d> next_points = smoothed_cloud.points_;

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int j = 0; j < n_points; ++j) {
            std::vector<int> indices;
            std::vector<double> dists;
            if (kdtree.SearchKNN(temp_cloud.points_[j], knn, indices, dists) >
                1) {
                Eigen::Vector3d centroid =
                        ComputeCentroid(temp_cloud.points_, indices);
                next_points[j] += lambda * (centroid - temp_cloud.points_[j]);
            }
        }
        smoothed_cloud.points_ = next_points;
    }

    return smoothed_cloud;
}

PointCloud PointCloud::SmoothTaubin(size_t iterations,
                                    double lambda,
                                    double mu,
                                    int knn) const {
    if (points_.empty() || iterations == 0) {
        return *this;
    }

    PointCloud smoothed_cloud = *this;
    int n_points = static_cast<int>(smoothed_cloud.points_.size());

    for (size_t i = 0; i < iterations; ++i) {
        // Lambda pass
        {
            PointCloud temp_cloud = smoothed_cloud;
            KDTreeFlann kdtree(temp_cloud);
            std::vector<Eigen::Vector3d> next_points = smoothed_cloud.points_;
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
            for (int j = 0; j < n_points; ++j) {
                std::vector<int> indices;
                std::vector<double> dists;
                if (kdtree.SearchKNN(temp_cloud.points_[j], knn, indices,
                                     dists) > 1) {
                    Eigen::Vector3d centroid =
                            ComputeCentroid(temp_cloud.points_, indices);
                    next_points[j] +=
                            lambda * (centroid - temp_cloud.points_[j]);
                }
            }
            smoothed_cloud.points_ = next_points;
        }

        // Mu pass
        {
            PointCloud temp_cloud = smoothed_cloud;
            KDTreeFlann kdtree(temp_cloud);
            std::vector<Eigen::Vector3d> next_points = smoothed_cloud.points_;
#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
            for (int j = 0; j < n_points; ++j) {
                std::vector<int> indices;
                std::vector<double> dists;
                if (kdtree.SearchKNN(temp_cloud.points_[j], knn, indices,
                                     dists) > 1) {
                    Eigen::Vector3d centroid =
                            ComputeCentroid(temp_cloud.points_, indices);
                    next_points[j] += mu * (centroid - temp_cloud.points_[j]);
                }
            }
            smoothed_cloud.points_ = next_points;
        }
    }

    return smoothed_cloud;
}

PointCloud PointCloud::SmoothBilateral(const KDTreeSearchParam& search_param,
                                       double sigma_s,
                                       double sigma_r) const {
    if (points_.empty()) {
        return *this;
    }
    if (sigma_s <= 0 || sigma_r <= 0) {
        utility::LogError("Sigma values must be positive.");
    }

    PointCloud smoothed_cloud = *this;

    // The filter needs normals.
    if (!HasNormals()) {
        utility::LogWarning(
                "PointCloud has no normals. Estimating normals with default "
                "parameters.");
        smoothed_cloud.EstimateNormals();
    }

    KDTreeFlann kdtree(*this);
    double inv_sigma_s2 = 1.0 / (2 * sigma_s * sigma_s);
    double inv_sigma_r2 = 1.0 / (2 * sigma_r * sigma_r);

    int n_points = static_cast<int>(points_.size());

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
    for (int i = 0; i < n_points; ++i) {
        std::vector<int> indices;
        std::vector<double> dists2;
        int nb_neighbors =
                kdtree.Search(points_[i], search_param, indices, dists2);

        if (nb_neighbors > 1) {
            double weight_sum = 0.0;
            Eigen::Vector3d point_sum = Eigen::Vector3d::Zero();
            const Eigen::Vector3d& p_i = points_[i];
            const Eigen::Vector3d& n_i = smoothed_cloud.normals_[i];

            std::vector<double> weights(nb_neighbors);

            for (int k = 0; k < nb_neighbors; ++k) {
                const Eigen::Vector3d& p_k = points_[indices[k]];
                double spatial_weight = std::exp(-dists2[k] * inv_sigma_s2);
                double range_dist = (p_i - p_k).dot(n_i);
                double range_weight =
                        std::exp(-(range_dist * range_dist) * inv_sigma_r2);
                weights[k] = spatial_weight * range_weight;
            }

            Eigen::Vector3d new_point =
                    ComputeWeightedCentroid(*this, indices, weights);
            smoothed_cloud.points_[i] = new_point;
        }
    }

    return smoothed_cloud;
}

}  // namespace geometry
}  // namespace open3d