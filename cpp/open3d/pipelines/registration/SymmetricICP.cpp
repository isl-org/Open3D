// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/SymmetricICP.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <cmath>
#include <cstddef>

#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/SymmetricICPImpl.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace pipelines {
namespace registration {
namespace {

void ValidateSymmetricICPNormals(const geometry::PointCloud &source,
                                 const geometry::PointCloud &target) {
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogError(
                "SymmetricICP requires both source and target to have "
                "normals.");
    }
}

void ValidateSymmetricICPCorrespondences(const geometry::PointCloud &source,
                                         const geometry::PointCloud &target,
                                         const CorrespondenceSet &corres) {
    for (const Eigen::Vector2i &correspondence : corres) {
        if (correspondence[0] < 0 || correspondence[1] < 0 ||
            static_cast<std::size_t>(correspondence[0]) >=
                    source.points_.size() ||
            static_cast<std::size_t>(correspondence[1]) >=
                    target.points_.size()) {
            utility::LogError(
                    "SymmetricICP correspondence ({}, {}) is out of range for "
                    "source size {} and target size {}.",
                    correspondence[0], correspondence[1], source.points_.size(),
                    target.points_.size());
        }
    }
}

Eigen::Vector3d GetSymmetricNormal(const Eigen::Vector3d &source_normal,
                                   const Eigen::Vector3d &target_normal) {
    if (source_normal.dot(target_normal) < 0.0) {
        return target_normal - source_normal;
    }
    return target_normal + source_normal;
}

struct CorrespondenceSums {
    Eigen::Vector3d source = Eigen::Vector3d::Zero();
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
};

struct NormalEquations {
    Eigen::Matrix6d JTJ = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr = Eigen::Vector6d::Zero();
};

}  // namespace

double TransformationEstimationSymmetric::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    ValidateSymmetricICPNormals(source, target);
    ValidateSymmetricICPCorrespondences(source, target, corres);
    if (corres.empty()) {
        return 0.0;
    }

    double err = 0.0;
    for (const auto &c : corres) {
        const Eigen::Vector3d &source_point = source.points_[c[0]];
        const Eigen::Vector3d &target_point = target.points_[c[1]];
        const Eigen::Vector3d normal = GetSymmetricNormal(
                source.normals_[c[0]], target.normals_[c[1]]);
        const double residual = (source_point - target_point).dot(normal);
        err += residual * residual;
    }
    return std::sqrt(err / static_cast<double>(corres.size()));
}

Eigen::Matrix4d TransformationEstimationSymmetric::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    ValidateSymmetricICPNormals(source, target);
    ValidateSymmetricICPCorrespondences(source, target, corres);
    if (corres.empty()) {
        return Eigen::Matrix4d::Identity();
    }

    const CorrespondenceSums sums = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, corres.size()),
            CorrespondenceSums(),
            [&](const tbb::blocked_range<std::size_t> &range,
                CorrespondenceSums local) {
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    local.source += source.points_[corres[i][0]];
                    local.target += target.points_[corres[i][1]];
                }
                return local;
            },
            [](CorrespondenceSums lhs, const CorrespondenceSums &rhs) {
                lhs.source += rhs.source;
                lhs.target += rhs.target;
                return lhs;
            });
    const double inverse_count = 1.0 / static_cast<double>(corres.size());
    const Eigen::Vector3d source_mean = sums.source * inverse_count;
    const Eigen::Vector3d target_mean = sums.target * inverse_count;

    // Centering the correspondences decouples the symmetric rotation and
    // translation system while the robust weight uses the raw residual.
    const NormalEquations equations = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, corres.size()),
            NormalEquations(),
            [&](const tbb::blocked_range<std::size_t> &range,
                NormalEquations local) {
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    const Eigen::Vector3d &source_point =
                            source.points_[corres[i][0]];
                    const Eigen::Vector3d &target_point =
                            target.points_[corres[i][1]];
                    const Eigen::Vector3d normal =
                            GetSymmetricNormal(source.normals_[corres[i][0]],
                                               target.normals_[corres[i][1]]);
                    const Eigen::Vector3d source_centered =
                            source_point - source_mean;
                    const Eigen::Vector3d target_centered =
                            target_point - target_mean;
                    const double raw_residual =
                            (source_point - target_point).dot(normal);
                    const double residual =
                            (source_centered - target_centered).dot(normal);

                    Eigen::Vector6d jacobian;
                    jacobian.head<3>() =
                            (source_centered + target_centered).cross(normal);
                    jacobian.tail<3>() = normal;

                    const double weight = kernel_->Weight(raw_residual);
                    local.JTJ.noalias() +=
                            weight * jacobian * jacobian.transpose();
                    local.JTr.noalias() += weight * jacobian * residual;
                }
                return local;
            },
            [](NormalEquations lhs, const NormalEquations &rhs) {
                lhs.JTJ += rhs.JTJ;
                lhs.JTr += rhs.JTr;
                return lhs;
            });

    bool is_success = false;
    Eigen::Vector6d pose;
    std::tie(is_success, pose) =
            utility::SolveLinearSystemPSD(equations.JTJ, -equations.JTr);
    return is_success ? TransformSymmetricPoseToMatrix4d(pose, source_mean,
                                                         target_mean)
                      : Eigen::Matrix4d::Identity();
}

std::tuple<std::shared_ptr<const geometry::PointCloud>,
           std::shared_ptr<const geometry::PointCloud>>
TransformationEstimationSymmetric::InitializePointCloudsForTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance) const {
    ValidateSymmetricICPNormals(source, target);
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
