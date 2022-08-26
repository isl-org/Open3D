// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <unordered_set>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace geometry {

/// \class RandomSampler
///
/// \brief Helper class for random sampling
template <typename T>
class RandomSampler {
public:
    explicit RandomSampler(const size_t total_size) : total_size_(total_size) {}

    std::vector<T> operator()(size_t sample_size) {
        std::vector<T> samples;
        samples.reserve(sample_size);

        size_t valid_sample = 0;
        while (valid_sample < sample_size) {
            const size_t idx = utility::random::RandUint32() % total_size_;
            // Well, this is slow. But typically the sample_size is small.
            if (std::find(samples.begin(), samples.end(), idx) ==
                samples.end()) {
                samples.push_back(idx);
                valid_sample++;
            }
        }

        return samples;
    }

private:
    size_t total_size_;
};

/// \class RANSACResult
///
/// \brief Stores the current best result in the RANSAC algorithm.
class RANSACResult {
public:
    RANSACResult() : fitness_(0), inlier_rmse_(0) {}
    ~RANSACResult() {}

public:
    double fitness_;
    double inlier_rmse_;
};

// Calculates the number of inliers given a list of points and a plane model,
// and the total distance between the inliers and the plane. These numbers are
// then used to evaluate how well the plane model fits the given points.
RANSACResult EvaluateRANSACBasedOnDistance(
        const std::vector<Eigen::Vector3d> &points,
        const Eigen::Vector4d plane_model,
        std::vector<size_t> &inliers,
        double distance_threshold,
        double error) {
    RANSACResult result;

    for (size_t idx = 0; idx < points.size(); ++idx) {
        Eigen::Vector4d point(points[idx](0), points[idx](1), points[idx](2),
                              1);
        double distance = std::abs(plane_model.dot(point));

        if (distance < distance_threshold) {
            error += distance;
            inliers.emplace_back(idx);
        }
    }

    size_t inlier_num = inliers.size();
    if (inlier_num == 0) {
        result.fitness_ = 0;
        result.inlier_rmse_ = 0;
    } else {
        result.fitness_ = (double)inlier_num / (double)points.size();
        result.inlier_rmse_ = error / std::sqrt((double)inlier_num);
    }
    return result;
}

// Find the plane such that the summed squared distance from the
// plane to all points is minimized.
//
// Reference:
// https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
Eigen::Vector4d GetPlaneFromPoints(const std::vector<Eigen::Vector3d> &points,
                                   const std::vector<size_t> &inliers) {
    Eigen::Vector3d centroid(0, 0, 0);
    for (size_t idx : inliers) {
        centroid += points[idx];
    }
    centroid /= double(inliers.size());

    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    for (size_t idx : inliers) {
        Eigen::Vector3d r = points[idx] - centroid;
        xx += r(0) * r(0);
        xy += r(0) * r(1);
        xz += r(0) * r(2);
        yy += r(1) * r(1);
        yz += r(1) * r(2);
        zz += r(2) * r(2);
    }

    double det_x = yy * zz - yz * yz;
    double det_y = xx * zz - xz * xz;
    double det_z = xx * yy - xy * xy;

    Eigen::Vector3d abc;
    if (det_x > det_y && det_x > det_z) {
        abc = Eigen::Vector3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    } else if (det_y > det_z) {
        abc = Eigen::Vector3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    } else {
        abc = Eigen::Vector3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    }

    double norm = abc.norm();
    // Return invalid plane if the points don't span a plane.
    if (norm == 0) {
        return Eigen::Vector4d(0, 0, 0, 0);
    }
    abc /= abc.norm();
    double d = -abc.dot(centroid);
    return Eigen::Vector4d(abc(0), abc(1), abc(2), d);
}

std::tuple<Eigen::Vector4d, std::vector<size_t>> PointCloud::SegmentPlane(
        const double distance_threshold /* = 0.01 */,
        const int ransac_n /* = 3 */,
        const int num_iterations /* = 100 */,
        const double probability /* = 0.99999999 */) const {
    if (probability <= 0 || probability > 1) {
        utility::LogError("Probability must be > 0 or <= 1.0");
    }

    RANSACResult result;

    // Initialize the best plane model.
    Eigen::Vector4d best_plane_model = Eigen::Vector4d(0, 0, 0, 0);

    size_t num_points = points_.size();
    RandomSampler<size_t> sampler(num_points);

    // Return if ransac_n is less than the required plane model parameters.
    if (ransac_n < 3) {
        utility::LogError(
                "ransac_n should be set to higher than or equal to 3.");
        return std::make_tuple(Eigen::Vector4d(0, 0, 0, 0),
                               std::vector<size_t>{});
    }
    if (num_points < size_t(ransac_n)) {
        utility::LogError("There must be at least 'ransac_n' points.");
        return std::make_tuple(Eigen::Vector4d(0, 0, 0, 0),
                               std::vector<size_t>{});
    }

    // Use size_t here to avoid large integer which acceed max of int.
    size_t break_iteration = std::numeric_limits<size_t>::max();
    int iteration_count = 0;

#pragma omp parallel for schedule(static)
    for (int itr = 0; itr < num_iterations; itr++) {
        if ((size_t)iteration_count > break_iteration) {
            continue;
        }

        const std::vector<size_t> sampled_indices = sampler(ransac_n);
        std::vector<size_t> inliers = sampled_indices;

        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        Eigen::Vector4d plane_model;
        if (ransac_n == 3) {
            plane_model = TriangleMesh::ComputeTrianglePlane(
                    points_[inliers[0]], points_[inliers[1]],
                    points_[inliers[2]]);
        } else {
            plane_model = GetPlaneFromPoints(points_, inliers);
        }

        if (plane_model.isZero(0)) {
            continue;
        }

        double error = 0;
        inliers.clear();
        auto this_result = EvaluateRANSACBasedOnDistance(
                points_, plane_model, inliers, distance_threshold, error);
#pragma omp critical
        {
            if (this_result.fitness_ > result.fitness_ ||
                (this_result.fitness_ == result.fitness_ &&
                 this_result.inlier_rmse_ < result.inlier_rmse_)) {
                result = this_result;
                best_plane_model = plane_model;
                if (result.fitness_ < 1.0) {
                    break_iteration = std::min(
                            log(1 - probability) /
                                    log(1 - pow(result.fitness_, ransac_n)),
                            (double)num_iterations);
                } else {
                    // Set break_iteration to 0 to force to break the loop.
                    break_iteration = 0;
                }
            }
            iteration_count++;
        }
    }

    // Find the final inliers using best_plane_model.
    std::vector<size_t> final_inliers;
    if (!best_plane_model.isZero(0)) {
        for (size_t idx = 0; idx < points_.size(); ++idx) {
            Eigen::Vector4d point(points_[idx](0), points_[idx](1),
                                  points_[idx](2), 1);
            double distance = std::abs(best_plane_model.dot(point));

            if (distance < distance_threshold) {
                final_inliers.emplace_back(idx);
            }
        }
    }

    // Improve best_plane_model using the final inliers.
    best_plane_model = GetPlaneFromPoints(points_, final_inliers);

    utility::LogDebug(
            "RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}, Iteration: "
            "{:d}",
            final_inliers.size(), result.fitness_, result.inlier_rmse_,
            iteration_count);
    return std::make_tuple(best_plane_model, final_inliers);
}

}  // namespace geometry
}  // namespace open3d
