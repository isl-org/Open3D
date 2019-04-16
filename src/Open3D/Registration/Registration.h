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

#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "Open3D/Registration/CorrespondenceChecker.h"
#include "Open3D/Registration/TransformationEstimation.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace registration {
class Feature;

/// Class that defines the convergence criteria of ICP
/// ICP algorithm stops if the relative change of fitness and rmse hit
/// relative_fitness_ and relative_rmse_ individually, or the iteration number
/// exceeds max_iteration_.
class ICPConvergenceCriteria {
public:
    ICPConvergenceCriteria(double relative_fitness = 1e-6,
                           double relative_rmse = 1e-6,
                           int max_iteration = 30)
        : relative_fitness_(relative_fitness),
          relative_rmse_(relative_rmse),
          max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteria() {}

public:
    double relative_fitness_;
    double relative_rmse_;
    int max_iteration_;
};

/// Class that defines the convergence criteria of RANSAC
/// RANSAC algorithm stops if the iteration number hits max_iteration_, or the
/// validation has been run for max_validation_ times.
/// Note that the validation is the most computational expensive operator in an
/// iteration. Most iterations do not do full validation. It is crucial to
/// control max_validation_ so that the computation time is acceptable.
class RANSACConvergenceCriteria {
public:
    RANSACConvergenceCriteria(int max_iteration = 1000,
                              int max_validation = 1000)
        : max_iteration_(max_iteration), max_validation_(max_validation) {}
    ~RANSACConvergenceCriteria() {}

public:
    int max_iteration_;
    int max_validation_;
};

/// Class that contains the registration results
class RegistrationResult {
public:
    RegistrationResult(
            const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity())
        : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}
    ~RegistrationResult() {}

public:
    Eigen::Matrix4d_u transformation_;
    CorrespondenceSet correspondence_set_;
    double inlier_rmse_;
    double fitness_;
};

/// Function for evaluation
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation = Eigen::Matrix4d::Identity());

/// Functions for ICP registration
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

/// Function for global RANSAC registration based on a given set of
/// correspondences
RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        int ransac_n = 6,
        const RANSACConvergenceCriteria &criteria =
                RANSACConvergenceCriteria());

/// Function for global RANSAC registration based on feature matching
RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Feature &source_feature,
        const Feature &target_feature,
        double max_correspondence_distance,
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(false),
        int ransac_n = 4,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers = {},
        const RANSACConvergenceCriteria &criteria =
                RANSACConvergenceCriteria());

/// Function for computing information matrix from transformation matrix
Eigen::Matrix6d GetInformationMatrixFromPointClouds(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation);

}  // namespace registration
}  // namespace open3d
