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
#include <memory>

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

/// \class TransformationEstimationForDopplerICP
///
/// Class to estimate a transformation for DICP with point to plane distance.
class TransformationEstimationForDopplerICP : public TransformationEstimation {
public:
    ~TransformationEstimationForDopplerICP() override{};

    /// \brief Constructor that takes as input a RobustKernel
    /// \param kernel Any of the implemented statistical robust kernel for
    /// outlier rejection.
    explicit TransformationEstimationForDopplerICP(
            double lambda_doppler = 0.01,
            bool reject_dynamic_outliers = false,
            double doppler_outlier_threshold = 2.0,
            size_t outlier_rejection_min_iteration = 2,
            size_t geometric_robust_loss_min_iteration = 0,
            size_t doppler_robust_loss_min_iteration = 2,
            std::shared_ptr<RobustKernel> geometric_kernel =
                    std::make_shared<L2Loss>(),
            std::shared_ptr<RobustKernel> doppler_kernel =
                    std::make_shared<L2Loss>())
        : lambda_doppler_(lambda_doppler),
          reject_dynamic_outliers_(reject_dynamic_outliers),
          doppler_outlier_threshold_(doppler_outlier_threshold),
          outlier_rejection_min_iteration_(outlier_rejection_min_iteration),
          geometric_robust_loss_min_iteration_(
                  geometric_robust_loss_min_iteration),
          doppler_robust_loss_min_iteration_(doppler_robust_loss_min_iteration),
          geometric_kernel_(std::move(geometric_kernel)),
          doppler_kernel_(std::move(doppler_kernel)) {
        if (lambda_doppler_ < 0 || lambda_doppler_ > 1.0) {
            lambda_doppler_ = 0.01;
        }
    }

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    double ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;
    Eigen::Matrix4d ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres,
            const std::vector<Eigen::Vector3d> &source_directions,
            const double period,
            const Eigen::Matrix4d &transformation,
            const Eigen::Matrix4d &T_V_to_S,
            const size_t iteration) const;

public:
    /// Factor that weighs the Doppler residual term in DICP objective.
    double lambda_doppler_{0.01};
    /// Whether or not to prune dynamic point outlier correspondences.
    bool reject_dynamic_outliers_{false};
    /// Correspondences with Doppler error greater than this threshold are
    /// rejected from optimization.
    double doppler_outlier_threshold_{2.0};
    /// Number of iterations of ICP after which outlier rejection is enabled.
    size_t outlier_rejection_min_iteration_{2};
    /// Number of iterations of ICP after which robust loss kicks in.
    size_t geometric_robust_loss_min_iteration_{0};
    size_t doppler_robust_loss_min_iteration_{2};

    /// shared_ptr to an Abstract RobustKernel that could mutate at runtime.
    std::shared_ptr<RobustKernel> default_kernel_ = std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> geometric_kernel_ =
            std::make_shared<L2Loss>();
    std::shared_ptr<RobustKernel> doppler_kernel_ = std::make_shared<L2Loss>();

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::DopplerICP;
};

/// \brief Function for Doppler ICP registration.
///
/// This is the implementation of the following paper:
/// B. Hexsel, H. Vhavle, Y. Chen,
/// DICP: Doppler Iterative Closest Point Algorithm, RSS 2022.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance (meters).
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]).
/// \param estimation TransformationEstimationForDopplerICP method. Can only
/// change the lambda_doppler value and the robust kernel used in the
/// optimization.
/// \param criteria Convergence criteria.
/// \param period Time period (in seconds) between the source and the target
/// point clouds. Default value: 0.1.
/// \param T_V_to_S The 4x4 transformation matrix to transform
/// sensor to vehicle frame.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]])
RegistrationResult RegistrationDopplerICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<Eigen::Vector3d> &source_directions,
        double max_distance,
        const Eigen::Matrix4d &init = Eigen::Matrix4d::Identity(),
        const TransformationEstimationForDopplerICP &estimation =
                TransformationEstimationForDopplerICP(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria(),
        const double period = 0.1F,
        const Eigen::Matrix4d &T_V_to_S = Eigen::Matrix4d::Identity());

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
