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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <memory>
#include <vector>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"

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

double ComputeModelResolution(const std::vector<Eigen::Vector3d>& points,
                              const geometry::KDTreeFlann& kdtree) {
    std::vector<int> indices(2);
    std::vector<double> distances(2);
    double resolution = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : resolution)
#endif
    for (int i = 0; i < (int)points.size(); i++) {
        if (kdtree.SearchKNN(points[i], 2, indices, distances) != 0) {
            resolution += std::sqrt(distances[1]);
        }
    }
    resolution /= points.size();
    return resolution;
}

Eigen::Matrix3d ComputeScatterMatrix(const std::vector<Eigen::Vector3d>& points,
                                     const Eigen::Vector3d& p,
                                     double salient_radius,
                                     int min_neighbors,
                                     const geometry::KDTreeFlann& kdtree) {
    std::vector<int> indices;
    std::vector<double> dist;
    int nb_neighbors = kdtree.SearchRadius(p, salient_radius, indices, dist);
    if (nb_neighbors < min_neighbors) {
        return {};
    }

    // sample mean vector
    Eigen::Vector3d up = Eigen::Vector3d::Zero();
    for (const auto& n_idx : indices) {
        up += points[n_idx];
    }

    // Compute the scatter matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& n_idx : indices) {
        const auto& n_point = points[n_idx];
        cov += (n_point - up) * (n_point - up).transpose();
    }
    return cov;
}
}  // namespace

namespace geometry {
std::shared_ptr<geometry::PointCloud> PointCloud::ComputeISSKeypoints(
        double salient_radius /* = 0.0 */,
        double non_max_radius /* = 0.0 */,
        double gamma_21 /* = 0.975 */,
        double gamma_32 /* = 0.975 */,
        int min_neighbors /*= 5 */) {
    KDTreeFlann kdtree(*this);

    if (salient_radius == 0.0 || non_max_radius == 0.0) {
        const double resolution = ComputeModelResolution(points_, kdtree);
        salient_radius = 6 * resolution;
        non_max_radius = 4 * resolution;
    }
    std::vector<double> third_eigen_values(points_.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(third_eigen_values)
#endif
    for (int i = 0; i < (int)points_.size(); i++) {
        Eigen::Matrix3d cov = ComputeScatterMatrix(
                points_, points_[i], salient_radius, min_neighbors, kdtree);
        if (cov.isZero()) {
            continue;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        const double& e1c = solver.eigenvalues()[2];
        const double& e2c = solver.eigenvalues()[1];
        const double& e3c = solver.eigenvalues()[0];

        if ((e2c / e1c) < gamma_21 && e3c / e2c < gamma_32) {
            third_eigen_values[i] = e3c;
        }
    }

    std::vector<Eigen::Vector3d> keypoints;
    keypoints.reserve(points_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared(keypoints)
#endif
    for (int i = 0; i < (int)points_.size(); i++) {
        if (third_eigen_values[i] > 0.0) {
            std::vector<int> nn_indices;
            std::vector<double> dist;
            int nb_neighbors = kdtree.SearchRadius(points_[i], non_max_radius,
                                                   nn_indices, dist);

            if (nb_neighbors >= min_neighbors &&
                IsLocalMaxima(i, nn_indices, third_eigen_values)) {
                keypoints.emplace_back(points_[i]);
            }
        }
    }

    return std::make_shared<geometry::PointCloud>(keypoints);
}  // namespace geometry

}  // namespace geometry
}  // namespace open3d
