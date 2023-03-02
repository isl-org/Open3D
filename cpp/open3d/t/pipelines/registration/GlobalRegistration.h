// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tuple>
#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/pipelines/registration/Registration.h"
#include "open3d/t/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace t {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

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
/// NOTE(wei): this is a duplicate from the legacy pipeline.
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

/// \brief Function to compute 3-point RANSAC from point-wise features for
/// global registration. Target is used to construct a nearest neighbor search
/// object, in order to query source.
/// \param source (N, 3) point cloud
/// \param target (M, 3) point cloud
/// \param source_feats (N, D) features where D is the feature dimension
/// \param target_feats (M, D) features where D is the feature dimension
/// \param max_correspondence_distance Max correspondence distance threshold to
/// judge where a correspondent point pair is an inlier
/// \param estimation Model to estimate the transformation from the sampled
/// correspondence set
/// \param criteria Convergence criteria of RANSAC
RegistrationResult RANSACFromFeatures(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &source_feats,
        const core::Tensor &target_feats,
        const double max_correspondence_distance,
        const RANSACConvergenceCriteria &criteria = RANSACConvergenceCriteria(),
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration = nullptr);

/// \brief Function to compute 3-point RANSAC from known correspondences for
/// global registration. Correspondences are organized by index pairs
/// (source_idx, target_idx).
/// \param source (N, 3) point cloud
/// \param target (M, 3) point cloud
/// \param correspondences (M, 2, Int64) tensor
/// \param max_correspondence_distance Max correspondence distance threshold to
/// judge where a correspondent point pair is an inlier
/// \param estimation Model to estimate the transformation from the sampled
/// correspondence set
/// \param criteria Convergence criteria of RANSAC
RegistrationResult RANSACFromCorrespondences(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const core::Tensor &correspondences,
        const double max_correspondence_distance,
        const RANSACConvergenceCriteria &criteria = RANSACConvergenceCriteria(),
        const std::function<
                void(const std::unordered_map<std::string, core::Tensor> &)>
                &callback_after_iteration = nullptr);

}  // namespace registration
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
