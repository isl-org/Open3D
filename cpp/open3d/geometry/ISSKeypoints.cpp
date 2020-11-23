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
#include <tuple>
#include <vector>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/Keypoint.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {

namespace {

bool IsLocalMaxima(int query_idx,
                   const std::vector<int>& indices,
                   const std::vector<double>& third_eigen_values) {
    for (const auto& idx : indices) {
        if (third_eigen_values[query_idx] < third_eigen_values[idx]) {
            return false;
        }
    }
    return true;
}

double ComputeModelResolution(const std::vector<Eigen::Vector3d>& points,
                              const geometry::KDTreeFlann& kdtree) {
    std::vector<int> indices(2);
    std::vector<double> distances(2);
    double resolution = 0.0;

    for (const auto& point : points) {
        if (kdtree.SearchKNN(point, 2, indices, distances) != 0) {
            resolution += std::sqrt(distances[1]);
        }
    }
    resolution /= points.size();
    return resolution;
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
#pragma omp parallel for schedule(static) shared(third_eigen_values)
    for (int i = 0; i < (int)points.size(); i++) {
        std::vector<int> indices;
        std::vector<double> dist;
        int nb_neighbors =
                kdtree.SearchRadius(points[i], salient_radius, indices, dist);
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

    std::vector<size_t> kp_indices;
    kp_indices.reserve(points.size());
#pragma omp parallel for schedule(static) shared(kp_indices)
    for (int i = 0; i < (int)points.size(); i++) {
        if (third_eigen_values[i] > 0.0) {
            std::vector<int> nn_indices;
            std::vector<double> dist;
            int nb_neighbors = kdtree.SearchRadius(points[i], non_max_radius,
                                                   nn_indices, dist);

            if (nb_neighbors >= min_neighbors &&
                IsLocalMaxima(i, nn_indices, third_eigen_values)) {
                kp_indices.emplace_back(i);
            }
        }
    }

    utility::LogDebug("[ComputeISSKeypoints] Extracted {} keypoints",
                      kp_indices.size());
    return input.SelectByIndex(kp_indices);
}

}  // namespace keypoint
}  // namespace geometry
}  // namespace open3d
