// @file      ISSDetector.cpp
// @author    Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, all rights reserved
#include "Open3D/Keypoints/ISSDetector.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <memory>
#include <vector>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

namespace {

inline bool IsLocalMaxima(int i,
                          const std::vector<int>& nn_indices,
                          const std::vector<double>& third_eigen_values) {
    bool is_max = true;
    for (const auto& n_idx : nn_indices) {
        if (third_eigen_values[i] < third_eigen_values[n_idx]) {
            is_max = false;
        }
    }
    return is_max;
}
}  // namespace

namespace keypoints {

double ISSDetector::ComputeModelResolution() const {
    std::vector<int> indices(2);
    std::vector<double> distances(2);
    double resolution = 0.0;
#pragma omp parallel for reduction(+ : resolution)
    for (size_t i = 0; i < cloud_->points_.size(); i++) {
        if (kdtree_.SearchKNN(cloud_->points_[i], 2, indices, distances) != 0) {
            resolution += std::sqrt(distances[1]);
        }
    }
    resolution /= cloud_->points_.size();
    return resolution;
}

Eigen::Matrix3d ISSDetector::ComputeScatterMatrix(
        const Eigen::Vector3d& p) const {
    std::vector<int> indices;
    std::vector<double> dist;
    int nb_neighbors = kdtree_.SearchRadius(p, salient_radius_, indices, dist);
    if (nb_neighbors < min_neighbors_) {
        return {};
    }

    // sample mean vector
    Eigen::Vector3d up = Eigen::Vector3d::Zero();
    for (const auto& n_idx : indices) {
        up += cloud_->points_[n_idx];
    }

    // Compute the scatter matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& n_idx : indices) {
        const auto& n_point = cloud_->points_[n_idx];
        cov += (n_point - up) * (n_point - up).transpose();
    }
    return cov;
}

std::shared_ptr<geometry::PointCloud> ISSDetector::ComputeKeypoints() const {
    const auto& points = cloud_->points_;
    std::vector<double> third_eigen_values(points.size());
#pragma omp parallel for shared(third_eigen_values)
    for (size_t i = 0; i < points.size(); i++) {
        Eigen::Matrix3d cov = ComputeScatterMatrix(points[i]);
        if (cov.isZero()) {
            continue;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        const double& e1c = solver.eigenvalues()[2];
        const double& e2c = solver.eigenvalues()[1];
        const double& e3c = solver.eigenvalues()[0];

        if ((e2c / e1c) < gamma_21_ && e3c / e2c < gamma_32_) {
            third_eigen_values[i] = e3c;
        }
    }

    std::vector<Eigen::Vector3d> keypoints;
    keypoints.reserve(points.size());
#pragma omp parallel for shared(keypoints)
    for (size_t i = 0; i < points.size(); i++) {
        if (third_eigen_values[i] > 0.0) {
            std::vector<int> nn_indices;
            std::vector<double> dist;
            int nb_neighbors = kdtree_.SearchRadius(points[i], non_max_radius_,
                                                    nn_indices, dist);

            if (nb_neighbors >= min_neighbors_ &&
                IsLocalMaxima(i, nn_indices, third_eigen_values)) {
                keypoints.emplace_back(points[i]);
            }
        }
    }

    return std::make_shared<geometry::PointCloud>(keypoints);
}

std::shared_ptr<geometry::PointCloud> ComputeISSKeypoints(
        const std::shared_ptr<geometry::PointCloud>& input,
        double salient_radius /* = 0.0 */,
        double non_max_radius /* = 0.0 */) {
    ISSDetector detector(input, salient_radius, non_max_radius);
    return detector.ComputeKeypoints();
}

}  // namespace keypoints
}  // namespace open3d
