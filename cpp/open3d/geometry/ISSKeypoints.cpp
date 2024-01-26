// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/Keypoint.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {

namespace {

bool IsLocalMaxima(int query_idx,
                   const std::vector<int>& indices,
                   const std::vector<double>& third_eigen_values) {
    return std::none_of(
            indices.begin(), indices.end(),
            [&third_eigen_values, value = third_eigen_values[query_idx]](
                    const int idx) { return value < third_eigen_values[idx]; });
}

double ComputeModelResolution(const std::vector<Eigen::Vector3d>& points,
                              const geometry::KDTreeFlann& kdtree) {
    std::vector<int> indices(2);
    std::vector<double> distances(2);
    const double resolution = std::accumulate(
            points.begin(), points.end(), 0.,
            [&](double state, const Eigen::Vector3d& point) {
                if (kdtree.SearchKNN(point, 2, indices, distances) >= 2) {
                    state += std::sqrt(distances[1]);
                }
                return state;
            });

    return resolution / static_cast<double>(points.size());
}

}  // namespace

namespace geometry {
namespace keypoint {
std::shared_ptr<PointCloud> ComputeISSKeypoints(
        const PointCloud& input,
        double salient_radius /* = 0.0 */,
        double non_max_radius /* = 0.0 */,
        double gamma_21 /* = 0.975 */,
        double gamma_32 /* = 0.975 */,
        int min_neighbors /*= 5 */) {
    if (input.points_.empty()) {
        utility::LogWarning("[ComputeISSKeypoints] Input PointCloud is empty!");
        return std::make_shared<PointCloud>();
    }
    const auto& points = input.points_;
    KDTreeFlann kdtree(input);

    if (salient_radius == 0.0 || non_max_radius == 0.0) {
        const double resolution = ComputeModelResolution(points, kdtree);
        salient_radius = 6 * resolution;
        non_max_radius = 4 * resolution;
        utility::LogDebug(
                "[ComputeISSKeypoints] Computed salient_radius = {}, "
                "non_max_radius = {} from input model",
                salient_radius, non_max_radius);
    }

    std::vector<double> third_eigen_values(points.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(
            0, points.size(), utility::DefaultGrainSizeTBB()),
            [&](const tbb::blocked_range<std::size_t>& range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            std::vector<int> indices;
            std::vector<double> dist;
            int nb_neighbors =
                    kdtree.SearchRadius(points[i],
                        salient_radius, indices, dist);
            if (nb_neighbors < min_neighbors) {
                continue;
            }

            Eigen::Matrix3d cov = utility::ComputeCovariance(points, indices);
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
    });

    tbb::concurrent_vector<std::size_t> kp_indices;
    kp_indices.reserve(points.size());
    tbb::parallel_for(tbb::blocked_range<std::size_t>(
            0, points.size(), utility::DefaultGrainSizeTBB()),
            [&](const tbb::blocked_range<std::size_t>& range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
            if (third_eigen_values[i] > 0.0) {
                std::vector<int> nn_indices;
                std::vector<double> dist;
                int nb_neighbors = kdtree.SearchRadius(
                        points[i], non_max_radius, nn_indices, dist);

                if (nb_neighbors >= min_neighbors &&
                    IsLocalMaxima(i, nn_indices, third_eigen_values)) {
                    kp_indices.emplace_back(i);
                }
            }
        }
    });


    utility::LogDebug("[ComputeISSKeypoints] Extracted {} keypoints",
                      kp_indices.size());
    return input.SelectByIndex({kp_indices.begin(), kp_indices.end()});
}

}  // namespace keypoint
}  // namespace geometry
}  // namespace open3d
