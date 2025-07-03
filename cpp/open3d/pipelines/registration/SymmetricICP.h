// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/pipelines/registration/TransformationEstimation.h"

namespace open3d {

namespace geometry {
class PointCloud;
}

namespace pipelines {
namespace registration {

class RegistrationResult;

/// \brief Transformation estimation for symmetric point-to-plane ICP.
class TransformationEstimationSymmetric : public TransformationEstimation {
public:
    ~TransformationEstimationSymmetric() override {};

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    explicit TransformationEstimationSymmetric(
            std::shared_ptr<RobustKernel> kernel = std::make_shared<L2Loss>())
        : kernel_(std::move(kernel)) {}
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    std::tuple<std::shared_ptr<const geometry::PointCloud>,
               std::shared_ptr<const geometry::PointCloud>>
    InitializePointCloudsForTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            double max_correspondence_distance) const override;

    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

/// \brief Function for symmetric ICP registration using point-to-plane error.
RegistrationResult RegistrationSymmetricICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationSymmetric &estimation =
                TransformationEstimationSymmetric(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
