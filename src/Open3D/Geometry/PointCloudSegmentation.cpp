// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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
#include <algorithm>
#include <iterator>
#include <random>
#include <unordered_set>

#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace geometry {

// std::exchange introduced in C++14
template <class T, class U = T>
T exchange(T &obj, U &&new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}

class RANSACResult {
public:
    RANSACResult() : inlier_rmse_(0), fitness_(0) {}
    ~RANSACResult() {}

public:
    double inlier_rmse_;
    double fitness_;
};

RANSACResult EvaluateRANSACBasedOnDistance(
        const std::vector<int> &inliers,
        const std::vector<Eigen::Vector3d> &points,
        const double error) {
    RANSACResult result;
    int inlier_num = inliers.size();

    if (inlier_num == 0) {
        result.fitness_ = 0;
        result.inlier_rmse_ = 0;
    } else {
        result.fitness_ = (double)inlier_num / (double)points.size();
        result.inlier_rmse_ = error / std::sqrt((double)inlier_num);
    }
    return result;
}

std::tuple<Eigen::Vector4d, std::vector<int>> PointCloud::SegmentPlane(
        double distance_threshold /* = 0.01 */,
        int ransac_n /* = 3 */,
        const registration::RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria() */) const {
    RANSACResult result;
    double error = 0;

    // Initialize the plane model ax + by + cz + d = 0.
    Eigen::Vector4d plane_model = Eigen::Vector4d(0, 0, 0, 0);
    // Initialize the best plane model.
    Eigen::Vector4d best_plane_model = Eigen::Vector4d(0, 0, 0, 0);
    const int num_model_parameters = 3;

    // Initialize consensus set.
    std::vector<int> inliers;
    int prev_inliers_num = 0;

    int num_points = points_.size();
    std::vector<int> indices(num_points);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int itr = 0;
         itr < criteria.max_iteration_ && itr < criteria.max_validation_;
         itr++) {
        if (prev_inliers_num == inliers.size()) {
            for (int i = 0; i < ransac_n; ++i) {
                indices[i] = exchange(indices[rng() % num_points], indices[i]);
            }

            inliers.clear();
            for (size_t idx = 0; idx < ransac_n; ++idx) {
                inliers.emplace_back(indices[idx]);
            }
        }
        prev_inliers_num = inliers.size();

        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        for (int i = 0; i < num_model_parameters; ++i) {
            inliers[i] = exchange(inliers[rng() % inliers.size()], indices[i]);
        }
        plane_model = TriangleMesh::ComputeTrianglePlane(
                points_[inliers[0]], points_[inliers[1]], points_[inliers[2]]);
        if (plane_model.isZero(0)) {
            continue;
        }

        inliers.clear();
        error = 0;
        for (int idx = 0; idx < points_.size(); ++idx) {
            Eigen::Vector4d point(points_[idx](0), points_[idx](1),
                                  points_[idx](2), 1);
            double distance = std::abs(plane_model.dot(point));

            if (distance < distance_threshold) {
                error += distance;
                inliers.emplace_back(idx);
            }
        }

        auto this_result =
                EvaluateRANSACBasedOnDistance(inliers, points_, error);
        if (this_result.fitness_ > result.fitness_ ||
            (this_result.fitness_ == result.fitness_ &&
             this_result.inlier_rmse_ < result.inlier_rmse_)) {
            result = this_result;
            best_plane_model = plane_model;
        }
    }

    // Find the final inliers using best_plane_model.
    inliers.clear();
    for (size_t idx = 0; idx < points_.size(); ++idx) {
        Eigen::Vector4d point(points_[idx](0), points_[idx](1), points_[idx](2),
                              1);
        double distance = std::abs(best_plane_model.dot(point));

        if (distance < distance_threshold) {
            inliers.emplace_back(idx);
        }
    }

    utility::LogDebug("RANSAC | Inliers: {:d}, Fitness: {:.4f}, RMSE: {:.4f}",
                      inliers.size(), result.fitness_, result.inlier_rmse_);
    return std::make_tuple(best_plane_model, inliers);
}

}  // namespace geometry
}  // namespace open3d