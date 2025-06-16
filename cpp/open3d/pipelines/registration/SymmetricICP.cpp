// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/SymmetricICP.h"

#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {

double TransformationEstimationSymmetric::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals() || !source.HasNormals()) {
        return 0.0;
    }
    double err = 0.0;
    for (const auto &c : corres) {
        const Eigen::Vector3d &vs = source.points_[c[0]];
        const Eigen::Vector3d &vt = target.points_[c[1]];
        const Eigen::Vector3d &ns = source.normals_[c[0]];
        const Eigen::Vector3d &nt = target.normals_[c[1]];
        Eigen::Vector3d d = vs - vt;
        double r1 = d.dot(nt);
        double r2 = d.dot(ns);
        err += r1 * r1 + r2 * r2;
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d TransformationEstimationSymmetric::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals() || !source.HasNormals()) {
        return Eigen::Matrix4d::Identity();
    }

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                const Eigen::Vector3d &vs = source.points_[corres[i][0]];
                const Eigen::Vector3d &vt = target.points_[corres[i][1]];
                const Eigen::Vector3d &ns = source.normals_[corres[i][0]];
                const Eigen::Vector3d &nt = target.normals_[corres[i][1]];
                const Eigen::Vector3d d = vs - vt;

                J_r.resize(2);
                r.resize(2);
                w.resize(2);

                r[0] = d.dot(nt);
                w[0] = kernel_->Weight(r[0]);
                J_r[0].block<3, 1>(0, 0) = vs.cross(nt);
                J_r[0].block<3, 1>(3, 0) = nt;

                r[1] = d.dot(ns);
                w[1] = kernel_->Weight(r[1]);
                J_r[1].block<3, 1>(0, 0) = vs.cross(ns);
                J_r[1].block<3, 1>(3, 0) = ns;
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

std::tuple<std::shared_ptr<const geometry::PointCloud>,
           std::shared_ptr<const geometry::PointCloud>>
TransformationEstimationSymmetric::InitializePointCloudsForTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance) const {
    if (!target.HasNormals() || !source.HasNormals()) {
        utility::LogError(
                "SymmetricICP requires both source and target to "
                "have normals.");
    }
    std::shared_ptr<const geometry::PointCloud> source_initialized_c(
            &source, [](const geometry::PointCloud *) {});
    std::shared_ptr<const geometry::PointCloud> target_initialized_c(
            &target, [](const geometry::PointCloud *) {});
    if (!source_initialized_c || !target_initialized_c) {
        utility::LogError(
                "Internal error: InitializePointCloudsFor"
                "Transformation returns nullptr.");
    }
    return std::make_tuple(source_initialized_c, target_initialized_c);
}

RegistrationResult RegistrationSymmetricICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init,
        const TransformationEstimationSymmetric &estimation,
        const ICPConvergenceCriteria &criteria) {
    return RegistrationICP(source, target, max_correspondence_distance, init,
                           estimation, criteria);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
