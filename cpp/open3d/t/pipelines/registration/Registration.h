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

#include <tuple>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace t {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {
class Feature;

/// \class ICPConvergenceCriteria
///
/// \brief Class that defines the convergence criteria of ICP.
class ICPConvergenceCriteria {
public:
    /// \brief Parameterized Constructor.
    /// ICP algorithm stops if the relative change of fitness and rmse hit
    /// \p relative_fitness_ and \p relative_rmse_ individually, or the
    /// iteration number exceeds \p max_iteration_.
    ///
    /// \param relative_fitness If relative change (difference) of fitness score
    /// is lower than relative_fitness, the iteration stops.
    /// \param relative_rmse If relative change (difference) of inliner RMSE
    /// score is lower than relative_rmse, the iteration stops.
    /// \param max_iteration Maximum iteration before iteration stops.
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

/// \class RegistrationResult
///
/// Class that contains the registration results.
class RegistrationResult {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param transformation The estimated transformation matrix of dtype
    /// Float64 on CPU device.
    RegistrationResult(const core::Tensor &transformation = core::Tensor::Eye(
                               4, core::Dtype::Float64, core::Device("CPU:0")))
        : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}

    ~RegistrationResult() {}

    bool IsBetterRANSACThan(const RegistrationResult &other) const {
        return fitness_ > other.fitness_ || (fitness_ == other.fitness_ &&
                                             inlier_rmse_ < other.inlier_rmse_);
    }

public:
    /// The estimated transformation matrix of dtype Float64 on CPU device.
    core::Tensor transformation_;
    /// Correspondence Set. Refer to the definition in
    /// `TransformationEstimation.h`.
    CorrespondenceSet correspondence_set_;
    /// RMSE of all inlier correspondences. Lower is better.
    double inlier_rmse_;
    /// For ICP: the overlapping area (# of inlier correspondences / # of points
    /// in target). Higher is better.
    double fitness_;
};

/// \brief Function for evaluating registration between point clouds.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param transformation The 4x4 transformation matrix to transform
/// source to target of dtype Float64 on CPU device.
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const core::Tensor &transformation = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0")));

/// \brief Functions for ICP registration.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param init_source_to_target Initial transformation estimation of type
/// Float64 on CPU.
/// \param estimation Estimation method.
/// \param criteria Convergence criteria.
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const core::Tensor &init_source_to_target = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0")),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

/// \brief Functions for Multi-Scale ICP registration.
/// It will run ICP on different voxel level, from coarse to dense.
/// The vector of ICPConvergenceCriteria(relative fitness, relative rmse,
/// max_iterations) contains the stoping condition for each voxel level.
/// The length of voxel_sizes vector, criteria vector,
/// max_correspondence_distances vector must be same, and voxel_sizes must
/// contain positive values in strictly decreasing order [Lower the voxel size,
/// higher is the resolution]. Only the last value of the voxel_sizes vector can
/// be {-1}, as it allows to run on the original scale without downsampling.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param voxel_sizes VectorDouble of voxel scales of type double.
/// \param criteria_list Vector of ICPConvergenceCriteria objects for each
/// scale.
/// \param max_correspondence_distances VectorDouble of maximum
/// correspondence points-pair distances of type double, for each iteration.
/// Must be of same length as voxel_sizes and criterias.
/// \param init_source_to_target Initial transformation estimation of type
/// Float64 on CPU.
/// \param estimation Estimation method.
RegistrationResult RegistrationMultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criteria_list,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init_source_to_target = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0")),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint());

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
