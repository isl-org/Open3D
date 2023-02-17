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

#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "open3d/pipelines/registration/CorrespondenceChecker.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Optional.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {
class Feature;

/// \class ICPConvergenceCriteria
///
/// \brief Class that defines the convergence criteria of ICP.
///
/// ICP algorithm stops if the relative change of fitness and rmse hit
/// \p relative_fitness_ and \p relative_rmse_ individually, or the iteration
/// number exceeds \p max_iteration_.
class ICPConvergenceCriteria {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param relative_fitness If relative change (difference) of fitness score
    /// is lower than relative_fitness, the iteration stops. \param
    /// relative_rmse If relative change (difference) of inliner RMSE score is
    /// lower than relative_rmse, the iteration stops. \param max_iteration
    /// Maximum iteration before iteration stops.
    ICPConvergenceCriteria(double relative_fitness = 1e-6,
                           double relative_rmse = 1e-6,
                           int max_iteration = 30)
        : relative_fitness_(relative_fitness),
          relative_rmse_(relative_rmse),
          max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteria() {}

public:
    /// If relative change (difference) of fitness score is lower than
    /// `relative_fitness`, the iteration stops.
    double relative_fitness_;
    /// If relative change (difference) of inliner RMSE score is lower than
    /// `relative_rmse`, the iteration stops.
    double relative_rmse_;
    /// Maximum iteration before iteration stops.
    int max_iteration_;
};

/// \class RANSACConvergenceCriteria
///
/// \brief Class that defines the convergence criteria of RANSAC.
///
/// RANSAC algorithm stops if the iteration number hits max_iteration_, or the
/// fitness measured during validation suggests that the algorithm can be
/// terminated early with some confidence_. Early termination takes place when
/// the number of iteration reaches k = log(1 - confidence)/log(1 -
/// fitness^{ransac_n}), where ransac_n is the number of points used during a
/// ransac iteration. Use confidence=1.0 to avoid early termination.
class RANSACConvergenceCriteria {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param max_iteration Maximum iteration before iteration stops.
    /// \param confidence Desired probability of success. Used for estimating
    /// early termination.
    RANSACConvergenceCriteria(int max_iteration = 100000,
                              double confidence = 0.999)
        : max_iteration_(max_iteration),
          confidence_(std::max(std::min(confidence, 1.0), 0.0)) {}

    ~RANSACConvergenceCriteria() {}

public:
    /// Maximum iteration before iteration stops.
    int max_iteration_;
    /// Desired probability of success.
    double confidence_;
};

/// \class RegistrationResult
///
/// Class that contains the registration results.
class RegistrationResult {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param transformation The estimated transformation matrix.
    RegistrationResult(
            const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity())
        : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}
    ~RegistrationResult() {}
    bool IsBetterRANSACThan(const RegistrationResult &other) const {
        return fitness_ > other.fitness_ || (fitness_ == other.fitness_ &&
                                             inlier_rmse_ < other.inlier_rmse_);
    }

public:
    /// The estimated transformation matrix.
    Eigen::Matrix4d_u transformation_;
    /// Correspondence set between source and target point cloud.
    CorrespondenceSet correspondence_set_;
    /// RMSE of all inlier correspondences. Lower is better.
    double inlier_rmse_;
    /// For ICP: the overlapping area (# of inlier correspondences / # of points
    /// in target). Higher is better.
    /// For RANSAC: inlier ratio (# of inlier correspondences / # of
    /// all correspondences)
    double fitness_;
};

/// \brief Function for evaluating registration between point clouds.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance. \param transformation The 4x4 transformation matrix to transform
/// source to target. Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.],
/// [0., 0., 1., 0.], [0., 0., 0., 1.]]).
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity());

/// \brief Functions for ICP registration.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance. \param init Initial transformation estimation.
///  Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
///  [0., 0., 0., 1.]])
/// \param estimation Estimation method.
/// \param criteria Convergence criteria.
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

/// \brief Function for global RANSAC registration based on a given set of
/// correspondences.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param corres Correspondence indices between source and target point clouds.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param estimation Estimation method.
/// \param ransac_n Fit ransac with `ransac_n` correspondences.
/// \param checkers Correspondence checker.
/// \param criteria Convergence criteria.
RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        int ransac_n = 3,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers = {},
        const RANSACConvergenceCriteria &criteria =
                RANSACConvergenceCriteria());

/// \brief Function for global RANSAC registration based on feature matching.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param source_feature Source point cloud feature.
/// \param target_feature Target point cloud feature.
/// \param mutual_filter Enables mutual filter such that the correspondence of
/// the source point's correspondence is itself.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param ransac_n Fit ransac with `ransac_n` correspondences.
/// \param checkers Correspondence checker.
/// \param criteria Convergence criteria.
RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Feature &source_feature,
        const Feature &target_feature,
        bool mutual_filter,
        double max_correspondence_distance,
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        int ransac_n = 3,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers = {},
        const RANSACConvergenceCriteria &criteria =
                RANSACConvergenceCriteria());

/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance. \param transformation The 4x4 transformation matrix to transform
/// `source` to `target`.
Eigen::Matrix6d GetInformationMatrixFromPointClouds(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation);

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
