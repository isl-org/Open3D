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

#include "open3d/pipelines/registration/DopplerICP.h"

#include <Eigen/Dense>
#include <iostream>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

Eigen::Matrix4d TransformationEstimationForDopplerICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    utility::LogError(
            "This method should not be called for DopplerICP. DopplerICP "
            "requires the period, current transformation, and T_V_to_S "
            "calibration.");
    return Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d TransformationEstimationForDopplerICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const std::vector<Eigen::Vector3d> &source_directions,
        const double period,
        const Eigen::Matrix4d &transformation,
        const Eigen::Matrix4d &T_V_to_S,
        const size_t iteration) const {
    if (corres.empty()) {
        utility::LogError(
                "No correspondences found between source and target "
                "pointcloud.");
    }
    if (!target.HasNormals()) {
        utility::LogError(
                "DopplerICP requires target pointcloud to have normals.");
    }
    if (!source.HasDopplers()) {
        utility::LogError(
                "DopplerICP requires source pointcloud to have Dopplers.");
    }
    if (std::abs(period) < 1e-3) {
        utility::LogError("Time period too small.");
    }

    const double lambda_geometric = 1.0 - lambda_doppler_;
    const double sqrt_lambda_doppler = std::sqrt(lambda_doppler_);
    const double sqrt_lambda_geometric = std::sqrt(lambda_geometric);
    const double sqrt_lambda_doppler_by_dt = sqrt_lambda_doppler / period;

    const Eigen::Vector6d state_vector =
            utility::TransformMatrix4dToVector6d(transformation);
    const Eigen::Matrix3d R_S_to_V = T_V_to_S.block<3, 3>(0, 0).inverse();
    const Eigen::Vector3d r_v_to_s_in_V = T_V_to_S.block<3, 1>(0, 3);
    const Eigen::Vector3d w_v_in_V = -state_vector.block<3, 1>(0, 0) / period;
    const Eigen::Vector3d v_v_in_V = -state_vector.block<3, 1>(3, 0) / period;
    const Eigen::Vector3d v_s_in_V = v_v_in_V + w_v_in_V.cross(r_v_to_s_in_V);
    const Eigen::Vector3d v_s_in_S = R_S_to_V * v_s_in_V;

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                const size_t cs = corres[i][0];
                const size_t ct = corres[i][1];
                const Eigen::Vector3d &ps_in_V = source.points_[cs];
                const Eigen::Vector3d &pt_in_V = target.points_[ct];
                const Eigen::Vector3d &nt_in_V = target.normals_[ct];
                const Eigen::Vector3d &ds_in_V = source_directions[cs];
                const double &doppler_in_S = source.dopplers_[cs];

                J_r.resize(2);
                r.resize(2);
                w.resize(2);

                // Compute predicted Doppler velocity.
                const Eigen::Vector3d ds_in_S = R_S_to_V * ds_in_V;
                const double doppler_pred_in_S = -ds_in_S.dot(v_s_in_S);
                const double doppler_error = doppler_in_S - doppler_pred_in_S;

                // Dynamic point outlier pruning of correspondences.
                bool optimize{true};
                if (reject_dynamic_outliers_ &&
                    iteration >= outlier_rejection_min_iteration_ &&
                    std::abs(doppler_error) > doppler_outlier_threshold_) {
                    optimize = false;
                }

                if (optimize) {
                    // Compute geometric point-to-plane error and Jacobian.
                    const double geometric_error =
                            (ps_in_V - pt_in_V).dot(nt_in_V);
                    r[0] = sqrt_lambda_geometric * geometric_error;
                    w[0] = (iteration >= geometric_robust_loss_min_iteration_)
                                   ? geometric_kernel_->Weight(r[0])
                                   : default_kernel_->Weight(r[0]);
                    J_r[0].block<3, 1>(0, 0) =
                            sqrt_lambda_geometric * ps_in_V.cross(nt_in_V);
                    J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric * nt_in_V;

                    // Compute Doppler error and Jacobian.
                    r[1] = sqrt_lambda_doppler * doppler_error;
                    w[1] = (iteration >= doppler_robust_loss_min_iteration_)
                                   ? doppler_kernel_->Weight(r[1])
                                   : default_kernel_->Weight(r[1]);
                    J_r[1].block<3, 1>(0, 0) = sqrt_lambda_doppler_by_dt *
                                               ds_in_V.cross(r_v_to_s_in_V);
                    J_r[1].block<3, 1>(3, 0) =
                            sqrt_lambda_doppler_by_dt * -ds_in_V;
                } else {
                    r[0] = 0.F;
                    w[0] = 0.F;
                    J_r[0].block<3, 1>(0, 0) = Eigen::Vector3d::Zero();
                    J_r[0].block<3, 1>(3, 0) = Eigen::Vector3d::Zero();

                    r[1] = 0.F;
                    w[1] = 0.F;
                    J_r[1].block<3, 1>(0, 0) = Eigen::Vector3d::Zero();
                    J_r[1].block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
                }
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

double TransformationEstimationForDopplerICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) return 0.0;
    double err = 0.0, r;
    for (const auto &c : corres) {
        r = (source.points_[c[0]] - target.points_[c[1]])
                    .dot(target.normals_[c[1]]);
        err += r * r;
    }
    return std::sqrt(err / (double)corres.size());
};

RegistrationResult RegistrationDopplerICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const std::vector<Eigen::Vector3d> &source_directions,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForDopplerICP &estimation
        /* = TransformationEstimationForDopplerICP()*/,
        const ICPConvergenceCriteria &criteria /* = ICPConvergenceCriteria()*/,
        const double period /* = 0.1F*/,
        const Eigen::Matrix4d &T_V_to_S /* = Eigen::Matrix4d::Identity()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }

    if ((estimation.GetTransformationEstimationType() ==
         TransformationEstimationType::DopplerICP) &&
        (!target.HasNormals() || !source.HasDopplers())) {
        utility::LogError(
                "TransformationEstimationDopplerICP requires Doppler "
                "velocities for source PointCloud and pre-computed normal "
                "vectors for target PointCloud.");
    }

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }

    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);

    int i;
    bool converged{false};
    for (i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);

        // Compute the transform update.
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_, source_directions,
                period, transformation, T_V_to_S, i);
        transformation = update * transformation;
        pcd.Transform(update);

        // Update the registration result.
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);

        // Check for convergence.
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            converged = true;
            break;
        }
    }

    result.num_iterations_ = i;
    result.converged_ = converged;
    return result;
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
