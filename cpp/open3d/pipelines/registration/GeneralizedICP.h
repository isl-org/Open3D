// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2021 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <memory>

#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"

namespace open3d {
namespace pipelines {
namespace registration {

class RegistrationResult;

class TransformationEstimationForGeneralizedICP
    : public TransformationEstimation {
public:
    ~TransformationEstimationForGeneralizedICP() override = default;

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    /// \brief Constructor that takes as input a RobustKernel.
    /// \param kernel Any of the implemented statistical robust kernel for
    /// outlier rejection.
    explicit TransformationEstimationForGeneralizedICP(
            double epsilon = 1e-3,
            std::shared_ptr<RobustKernel> kernel = std::make_shared<L2Loss>())
        : epsilon_(epsilon), kernel_(std::move(kernel)) {}

public:
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;

    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    /// Small constant representing covariance along the normal.
    double epsilon_ = 1e-3;

    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::GeneralizedICP;
};

/// \brief Function for Generalized ICP registration.
///
/// This is implementation of following paper
//  A. Segal, D .Haehnel, S. Thrun
/// Generalized-ICP, RSS 2009.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance.
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]). \param criteria  Convergence criteria. \param
RegistrationResult RegistrationGeneralizedICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationForGeneralizedICP &estimation =
                TransformationEstimationForGeneralizedICP(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
