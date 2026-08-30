// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
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

/// \brief Estimates a source-to-target transformation with symmetric ICP.
///
/// For each correspondence, \f$p\f$ is the source point and \f$q\f$ is the
/// target point in the current aligned frame, with corresponding normals
/// \f$n_p\f$ and \f$n_q\f$. After aligning the normal directions, the
/// objective uses the single residual \f$(p-q)^T(n_p+n_q)\f$.
class TransformationEstimationSymmetric : public TransformationEstimation {
public:
    ~TransformationEstimationSymmetric() override = default;

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    /// \brief Constructs a symmetric transformation estimator.
    /// \param kernel Robust kernel applied to the symmetric residual.
    explicit TransformationEstimationSymmetric(
            std::shared_ptr<RobustKernel> kernel = std::make_shared<L2Loss>())
        : kernel_(std::move(kernel)) {}

    /// \brief Computes the symmetric point-to-plane RMSE.
    /// \param source Source point cloud in the current aligned frame.
    /// \param target Target point cloud in the current aligned frame.
    /// \param corres Source-to-target correspondence indices.
    /// \return The symmetric point-to-plane RMSE.
    /// \throw std::runtime_error If either point cloud lacks normals or a
    /// correspondence index is out of range.
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;

    /// \brief Estimates a source-to-target transformation update.
    /// \param source Source point cloud in the current aligned frame.
    /// \param target Target point cloud in the current aligned frame.
    /// \param corres Source-to-target correspondence indices.
    /// \return The source-to-target transformation update.
    /// \throw std::runtime_error If either point cloud lacks normals or a
    /// correspondence index is out of range.
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    /// \brief Validates and initializes point clouds for symmetric ICP.
    /// \param source Source point cloud.
    /// \param target Target point cloud.
    /// \param max_correspondence_distance Maximum correspondence distance.
    /// \return The initialized source and target point clouds.
    /// \throw std::runtime_error If either point cloud lacks normals.
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
            TransformationEstimationType::SymmetricICP;
};

/// \brief Registers source to target with symmetric point-to-plane ICP.
///
/// For each correspondence in the current aligned frame, \f$p\f$ and
/// \f$n_p\f$ denote the source point and normal, while \f$q\f$ and \f$n_q\f$
/// denote the target point and normal. After aligning normal directions, the
/// objective uses the single residual \f$(p-q)^T(n_p+n_q)\f$.
/// \param source Source point cloud with normals.
/// \param target Target point cloud with normals.
/// \param max_correspondence_distance Maximum correspondence distance.
/// \param init Initial source-to-target transformation.
/// \param estimation Symmetric transformation estimator.
/// \param criteria ICP convergence criteria.
/// \return The registration result with a source-to-target transformation.
/// \throw std::runtime_error If either point cloud lacks normals.
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
