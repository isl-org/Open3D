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

#include <tuple>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TensorMap.h"
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
    /// Float64 on CPU device. Default: Identity tensor.
    RegistrationResult(const core::Tensor &transformation = core::Tensor::Eye(
                               4, core::Float64, core::Device("CPU:0")))
        : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}

    ~RegistrationResult() {}

    bool IsBetterThan(const RegistrationResult &other) const {
        return fitness_ > other.fitness_ || (fitness_ == other.fitness_ &&
                                             inlier_rmse_ < other.inlier_rmse_);
    }

public:
    /// The estimated transformation matrix of dtype Float64 on CPU device.
    core::Tensor transformation_;
    /// Tensor containing indices of corresponding target points, where the
    /// value is the target index and the index of the value itself is the
    /// source index. It contains -1 as value at index with no correspondence.
    core::Tensor correspondences_;
    /// RMSE of all inlier correspondences. Lower is better.
    double inlier_rmse_;
    /// For ICP: the overlapping area (# of inlier correspondences / # of points
    /// in target). Higher is better.
    double fitness_;
};

/// \brief Function for evaluating registration between point clouds.
///
/// \param source The source point cloud. (Float32 or Float64 type).
/// \param target The target point cloud. (Float32 or Float64 type).
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param transformation The 4x4 transformation matrix to transform
/// source to target of dtype Float64 on CPU device.
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const core::Tensor &transformation =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")));

/// \brief Functions for ICP registration.
///
/// \param source The source point cloud. (Float32 or Float64 type).
/// \param target The target point cloud. (Float32 or Float64 type).
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param init_source_to_target Initial transformation estimation of type
/// Float64 on CPU.
/// \param estimation Estimation method.
/// \param criteria Convergence criteria.
/// \param voxel_size The input pointclouds will be down-sampled to this
/// `voxel_size` scale. If voxel_size < 0, original scale will be used.
/// However it is highly recommended to down-sample the point-cloud for
/// performance. By default original scale of the point-cloud will be used.
/// \param callback_after_iteration Optional lambda function, saves string to
/// tensor map of attributes such as "iteration_index", "scale_index",
/// "scale_iteration_index", "inlier_rmse", "fitness", "transformation", on CPU
/// device, updated after each iteration.
RegistrationResult
ICP(const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const double max_correspondence_distance,
    const core::Tensor &init_source_to_target =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
    const TransformationEstimation &estimation =
            TransformationEstimationPointToPoint(),
    const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria(),
    const double voxel_size = -1.0,
    const std::function<void(const std::unordered_map<std::string, core::Tensor>
                                     &)> &callback_after_iteration = nullptr);

/// \brief Functions for Multi-Scale ICP registration.
/// It will run ICP on different voxel level, from coarse to dense.
/// The vector of ICPConvergenceCriteria(relative fitness, relative rmse,
/// max_iterations) contains the stopping condition for each voxel level.
/// The length of voxel_sizes vector, criteria vector,
/// max_correspondence_distances vector must be same, and voxel_sizes must
/// contain positive values in strictly decreasing order [Lower the voxel size,
/// higher is the resolution]. Only the last value of the voxel_sizes vector can
/// be {-1}, as it allows to run on the original scale without downsampling.
///
/// \param source The source point cloud. (Float32 or Float64 type).
/// \param target The target point cloud. (Float32 or Float64 type).
/// \param voxel_sizes VectorDouble of voxel scales of type double.
/// \param criteria_list Vector of ICPConvergenceCriteria objects for each
/// scale.
/// \param max_correspondence_distances VectorDouble of maximum
/// correspondence points-pair distances of type double, for each iteration.
/// Must be of same length as voxel_sizes and criterias.
/// \param init_source_to_target Initial transformation estimation of type
/// Float64 on CPU.
/// \param estimation Estimation method.
/// \param callback_after_iteration Optional lambda function, saves string to
/// tensor map of attributes such as "iteration_index", "scale_index",
/// "scale_iteration_index", "inlier_rmse", "fitness", "transformation", on CPU
/// device, updated after each iteration.
RegistrationResult MultiScaleICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<double> &voxel_sizes,
        const std::vector<ICPConvergenceCriteria> &criteria_list,
        const std::vector<double> &max_correspondence_distances,
        const core::Tensor &init_source_to_target =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0")),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(),
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration = nullptr);

/// \brief Computes `Information Matrix`, from the transformation between source
/// and target pointcloud. It returns the `Information Matrix` of shape {6, 6},
/// of dtype `Float64` on device `CPU:0`.
///
/// \param source The source point cloud. (Float32 or Float64 type).
/// \param target The target point cloud. (Float32 or Float64 type).
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance.
/// \param transformation The 4x4 transformation matrix to transform
/// `source` to `target`.
core::Tensor GetInformationMatrix(const geometry::PointCloud &source,
                                  const geometry::PointCloud &target,
                                  const double max_correspondence_distance,
                                  const core::Tensor &transformation);

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
