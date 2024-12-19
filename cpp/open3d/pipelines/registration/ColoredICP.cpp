// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/ColoredICP.h"

#include <Eigen/Dense>
#include <iostream>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

namespace {

class PointCloudForColoredICP : public geometry::PointCloud {
public:
    std::vector<Eigen::Vector3d> color_gradient_;
};

std::shared_ptr<PointCloudForColoredICP> InitializePointCloudForColoredICP(
        const geometry::PointCloud &target,
        const geometry::KDTreeSearchParamHybrid &search_param) {
    utility::LogDebug("InitializePointCloudForColoredICP");

    geometry::KDTreeFlann tree;
    tree.SetGeometry(target);

    auto output = std::make_shared<PointCloudForColoredICP>();
    output->colors_ = target.colors_;
    output->normals_ = target.normals_;
    output->points_ = target.points_;

    size_t n_points = output->points_.size();
    output->color_gradient_.resize(n_points, Eigen::Vector3d::Zero());

    for (size_t k = 0; k < n_points; k++) {
        const Eigen::Vector3d &vt = output->points_[k];
        const Eigen::Vector3d &nt = output->normals_[k];
        double it = (output->colors_[k](0) + output->colors_[k](1) +
                     output->colors_[k](2)) /
                    3.0;

        std::vector<int> point_idx;
        std::vector<double> point_squared_distance;

        if (tree.SearchHybrid(vt, search_param.radius_, search_param.max_nn_,
                              point_idx, point_squared_distance) >= 4) {
            // approximate image gradient of vt's tangential plane
            size_t nn = point_idx.size();
            Eigen::MatrixXd A(nn, 3);
            Eigen::MatrixXd b(nn, 1);
            A.setZero();
            b.setZero();
            for (size_t i = 1; i < nn; i++) {
                int P_adj_idx = point_idx[i];
                const Eigen::Vector3d &vt_adj = output->points_[P_adj_idx];
                double it_adj = (output->colors_[P_adj_idx](0) +
                                 output->colors_[P_adj_idx](1) +
                                 output->colors_[P_adj_idx](2)) /
                                3.0;
                A(i - 1, 0) = (vt_adj(0) - vt(0));
                A(i - 1, 1) = (vt_adj(1) - vt(1));
                A(i - 1, 2) = (vt_adj(2) - vt(2));
                b(i - 1, 0) = (it_adj - it);
            }
            // adds orthogonal constraint
            A(nn - 1, 0) = (nn - 1) * nt(0);
            A(nn - 1, 1) = (nn - 1) * nt(1);
            A(nn - 1, 2) = (nn - 1) * nt(2);
            b(nn - 1, 0) = 0;
            // solving linear equation
            bool is_success = false;
            Eigen::MatrixXd x;
            std::tie(is_success, x) = utility::SolveLinearSystemPSD(
                    A.transpose() * A, A.transpose() * b);
            if (is_success) {
                output->color_gradient_[k] = x;
            }
        }
    }
    return output;
}

}  // namespace

Eigen::Matrix4d TransformationEstimationForColoredICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) {
        utility::LogError(
                "No correspondences found between source and target "
                "pointcloud.");
    }
    if (!target.HasNormals()) {
        utility::LogError(
                "ColoredICP requires target pointcloud to have normals.");
    }
    if (!target.HasColors()) {
        utility::LogError(
                "ColoredICP requires target pointcloud to have colors.");
    }
    if (!source.HasColors()) {
        utility::LogError(
                "ColoredICP requires source pointcloud to have colors.");
    }

    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);

    const auto &target_c = (const PointCloudForColoredICP &)target;

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                size_t cs = corres[i][0];
                size_t ct = corres[i][1];
                const Eigen::Vector3d &vs = source.points_[cs];
                const Eigen::Vector3d &vt = target.points_[ct];
                const Eigen::Vector3d &nt = target.normals_[ct];

                J_r.resize(2);
                r.resize(2);
                w.resize(2);

                J_r[0].block<3, 1>(0, 0) = sqrt_lambda_geometric * vs.cross(nt);
                J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric * nt;
                r[0] = sqrt_lambda_geometric * (vs - vt).dot(nt);
                w[0] = kernel_->Weight(r[0]);

                // project vs into vt's tangential plane
                Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;
                double is = (source.colors_[cs](0) + source.colors_[cs](1) +
                             source.colors_[cs](2)) /
                            3.0;
                double it = (target.colors_[ct](0) + target.colors_[ct](1) +
                             target.colors_[ct](2)) /
                            3.0;
                const Eigen::Vector3d &dit = target_c.color_gradient_[ct];
                double is_proj = (dit.dot(vs_proj - vt)) + it;

                const Eigen::Matrix3d &M =
                        Eigen::Matrix3d::Identity() - nt * nt.transpose();
                const Eigen::Vector3d &ditM = dit.transpose() * M;

                J_r[1].block<3, 1>(0, 0) =
                        sqrt_lambda_photometric * vs.cross(ditM);
                J_r[1].block<3, 1>(3, 0) = sqrt_lambda_photometric * ditM;
                r[1] = sqrt_lambda_photometric * (is_proj - is);
                w[1] = kernel_->Weight(r[1]);
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

double TransformationEstimationForColoredICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    double sqrt_lambda_geometric = sqrt(lambda_geometric_);
    double lambda_photometric = 1.0 - lambda_geometric_;
    double sqrt_lambda_photometric = sqrt(lambda_photometric);
    const auto &target_c = (const PointCloudForColoredICP &)target;

    double residual = 0.0;
    for (size_t i = 0; i < corres.size(); i++) {
        size_t cs = corres[i][0];
        size_t ct = corres[i][1];
        const Eigen::Vector3d &vs = source.points_[cs];
        const Eigen::Vector3d &vt = target.points_[ct];
        const Eigen::Vector3d &nt = target.normals_[ct];
        Eigen::Vector3d vs_proj = vs - (vs - vt).dot(nt) * nt;
        double is = (source.colors_[cs](0) + source.colors_[cs](1) +
                     source.colors_[cs](2)) /
                    3.0;
        double it = (target.colors_[ct](0) + target.colors_[ct](1) +
                     target.colors_[ct](2)) /
                    3.0;
        const Eigen::Vector3d &dit = target_c.color_gradient_[ct];
        double is0_proj = (dit.dot(vs_proj - vt)) + it;
        double residual_geometric = sqrt_lambda_geometric * (vs - vt).dot(nt);
        double residual_photometric = sqrt_lambda_photometric * (is - is0_proj);
        residual += residual_geometric * residual_geometric +
                    residual_photometric * residual_photometric;
    }
    return residual;
};

std::tuple<std::shared_ptr<const geometry::PointCloud>,
           std::shared_ptr<const geometry::PointCloud>>
TransformationEstimationForColoredICP::InitializePointCloudsForTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance) const {
    if (!target.HasNormals()) {
        utility::LogError(
                "ColoredICP requires target pointcloud to have normals.");
    }
    if (!target.HasColors()) {
        utility::LogError(
                "ColoredICP requires target pointcloud to have colors.");
    }
    if (!source.HasColors()) {
        utility::LogError(
                "ColoredICP requires source pointcloud to have colors.");
    }
    std::shared_ptr<const geometry::PointCloud> source_initialized_c(
            &source, [](const geometry::PointCloud *) {});
    std::shared_ptr<geometry::PointCloud> target_initialized_c(
            InitializePointCloudForColoredICP(
                    target, geometry::KDTreeSearchParamHybrid(
                                    max_correspondence_distance * 2.0, 30)));
    if (!source_initialized_c || !target_initialized_c) {
        utility::LogError(
                "Internal error: InitializePointCloudsForTransformation "
                "returns "
                "nullptr.");
    }
    return std::make_tuple(source_initialized_c,
                           std::const_pointer_cast<const geometry::PointCloud>(
                                   target_initialized_c));
}

RegistrationResult RegistrationColoredICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForColoredICP &estimation
        /* = TransformationEstimationForColoredICP()*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    return RegistrationICP(source, target, max_distance, init, estimation,
                           criteria);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
