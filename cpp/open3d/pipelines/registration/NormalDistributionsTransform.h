// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include "open3d/pipelines/registration/Registration.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

class RegistrationResult;

/// \class NormalDistributionsTransformOption
///
/// \brief Class that defines options for 3D Normal Distributions Transform
/// registration.
class NormalDistributionsTransformOption {
public:
    /// \brief Parameterized Constructor.
    ///
    /// \param voxel_size Target voxel size used to build the Gaussian model.
    /// \param min_points_per_voxel Minimum number of target points needed for a
    /// voxel Gaussian.
    /// \param covariance_regularization Minimum eigenvalue ratio used to keep
    /// voxel covariance inverses well-conditioned.
    /// \param transformation_epsilon Stop optimization when the update vector
    /// norm is lower than this value.
    /// \param relative_objective Stop optimization when the relative change in
    /// mean Mahalanobis objective is lower than this value.
    /// \param max_iteration Maximum number of Gauss-Newton iterations.
    /// \param outlier_threshold Maximum squared Mahalanobis distance accepted
    /// for a point-to-voxel residual.
    /// \param neighbor_search_type 0 uses only the rounded center voxel. 1
    /// also uses the six face-adjacent voxels.
    NormalDistributionsTransformOption(double voxel_size = 1.0,
                                       int min_points_per_voxel = 6,
                                       double covariance_regularization = 1e-3,
                                       double transformation_epsilon = 1e-6,
                                       double relative_objective = 1e-6,
                                       int max_iteration = 30,
                                       double outlier_threshold = 9.0,
                                       int neighbor_search_type = 1);

    ~NormalDistributionsTransformOption() {}

public:
    /// Target voxel size used to build the Gaussian model.
    double voxel_size_;
    /// Minimum number of target points needed for a voxel Gaussian.
    int min_points_per_voxel_;
    /// Minimum eigenvalue ratio for regularizing voxel covariance inverses.
    double covariance_regularization_;
    /// Stop threshold for the Gauss-Newton update vector norm.
    double transformation_epsilon_;
    /// Stop threshold for relative mean Mahalanobis objective change.
    double relative_objective_;
    /// Maximum number of Gauss-Newton iterations.
    int max_iteration_;
    /// Maximum squared Mahalanobis distance accepted for a residual.
    double outlier_threshold_;
    /// 0 uses the rounded center voxel; 1 also uses the six face-adjacent
    /// voxels.
    int neighbor_search_type_;
};

/// \brief Function for 3D Normal Distributions Transform registration.
///
/// This implementation builds a voxelized Gaussian model from the target point
/// cloud and optimizes the rigid source-to-target transformation with
/// Gauss-Newton iterations.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param option NDT voxel model and optimization options.
/// \param init Initial transformation estimation.
RegistrationResult RegistrationNDT(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const NormalDistributionsTransformOption &option =
                NormalDistributionsTransformOption(),
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
