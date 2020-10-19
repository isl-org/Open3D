// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/GeneralizedICP.h"

#include <Eigen/Dense>
#include <iostream>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/Eigen.h"

namespace open3d {
namespace pipelines {
namespace registration {

namespace {

/// Obatin the Rotation matrix that transform the basis vector e1 onto the
/// input vector x.
inline Eigen::Matrix3d GetRotationFromE1ToX(const Eigen::Vector3d &x) {
    const Eigen::Vector3d e1{1, 0, 0};
    const Eigen::Vector3d v = e1.cross(x);
    const double c = e1.dot(x);
    if (c < -0.99) {
        // Then means that x and e1 are in the same direction
        return Eigen::Matrix3d::Identity();
    }

    const Eigen::Matrix3d sv = utility::SkewMatrix(v);
    const double factor = 1 / (1 + c);
    return Eigen::Matrix3d::Identity() + sv + (sv * sv) * factor;
}

class PointCloudWithCovariance : public geometry::PointCloud {
public:
    std::vector<Eigen::Matrix3d> covariances_;
};

std::shared_ptr<PointCloudWithCovariance> InitializePointCloudForGeneralizedICP(
        const geometry::PointCloud &pcd) {
    utility::LogDebug("InitializePointCloudForGeneralizedICP");
    auto output = std::make_shared<PointCloudWithCovariance>();
    output->points_ = pcd.points_;
    output->normals_ = pcd.normals_;
    output->covariances_.resize(output->points_.size());

    const Eigen::Matrix3d C = Eigen::Vector3d(1, 1, 1e-3).asDiagonal();
#pragma omp parallel for
    for (int i = 0; i < (int)output->points_.size(); i++) {
        const auto Rx = GetRotationFromE1ToX(output->normals_[i]);
        output->covariances_[i] = Rx * C * Rx.transpose();
    }
    return output;
}

RegistrationResult GetRegistrationResultAndCorrespondences(
        const PointCloudWithCovariance &source,
        const geometry::PointCloud &target,
        const geometry::KDTreeFlann &target_kdtree,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    double error2 = 0.0;

#pragma omp parallel
    {
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#pragma omp for nowait
        for (int i = 0; i < (int)source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, max_correspondence_distance,
                                           1, indices, dists) > 0) {
                error2_private += dists[0];
                correspondence_set_private.push_back(
                        Eigen::Vector2i(i, indices[0]));
            }
        }
#pragma omp critical
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
    }

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (double)corres_number / (double)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)corres_number);
    }
    return result;
}
}  // namespace

double TransformationEstimationForGeneralizedICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    return ComputeRMSE(source, target, corres, Eigen::Matrix4d::Identity());
}

double TransformationEstimationForGeneralizedICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &T) const {
    if (corres.empty()) {
        return 0.0;
    }
    double err = 0.0;
    const auto &source_c = (const PointCloudWithCovariance &)source;
    const auto &target_c = (const PointCloudWithCovariance &)target;
    for (const auto &c : corres) {
        const Eigen::Vector3d &vs = source_c.points_[c[0]];
        const Eigen::Matrix3d &Cs = source_c.covariances_[c[0]];
        const Eigen::Vector3d &vt = target_c.points_[c[1]];
        const Eigen::Matrix3d &Ct = target_c.covariances_[c[1]];
        const Eigen::Vector3d d = vs - vt;
        const Eigen::Matrix3d &R = T.block<3, 3>(0, 0);
        const Eigen::Matrix3d M = Ct + R * Cs * R.transpose();
        err += d.transpose() * M * d;
    }
    return std::sqrt(err / (double)corres.size());
}

Eigen::Matrix4d
TransformationEstimationForGeneralizedICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    return ComputeTransformation(source, target, corres,
                                 Eigen::Matrix4d::Identity());
}

Eigen::Matrix4d
TransformationEstimationForGeneralizedICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4d &T) const {
    if (corres.empty() || !target.HasNormals() || !source.HasNormals()) {
        return Eigen::Matrix4d::Identity();
    }

    const auto &source_c = (const PointCloudWithCovariance &)source;
    const auto &target_c = (const PointCloudWithCovariance &)target;

    auto compute_jacobian_and_residual =
            [&](int i,
                std::vector<Eigen::Vector6d, utility::Vector6d_allocator> &J_r,
                std::vector<double> &r, std::vector<double> &w) {
                // source
                const Eigen::Vector3d &vs = source_c.points_[corres[i][0]];
                const Eigen::Matrix3d &Cs = source_c.covariances_[corres[i][0]];
                // target
                const Eigen::Vector3d &vt = target_c.points_[corres[i][1]];
                const Eigen::Matrix3d &Ct = target_c.covariances_[corres[i][1]];
                const Eigen::Vector3d d = vs - vt;
                const Eigen::Matrix3d &R = T.block<3, 3>(0, 0);
                const Eigen::Matrix3d M = Ct + R * Cs * R.transpose();

                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = -utility::SkewMatrix(vs);
                J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                J = M * J;

                constexpr int n_rows = 3;
                J_r.resize(n_rows);
                r.resize(n_rows);
                w.resize(n_rows);
                for (size_t i = 0; i < n_rows; ++i) {
                    r[i] = M.row(i).dot(d);
                    w[i] = kernel_->Weight(r[i]);
                    J_r[i] = J.row(i);
                }
            };

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
    double r2 = -1.0;
    std::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6d, Eigen::Vector6d>(
                    compute_jacobian_and_residual, (int)corres.size());

    bool is_success = false;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4d::Identity();
}

RegistrationResult PrivateRegistrationICP(
        const PointCloudWithCovariance &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init,
        const TransformationEstimationForGeneralizedICP &estimation,
        const ICPConvergenceCriteria &criteria) {
    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    PointCloudWithCovariance pcd = source;
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug(
                "GICP Iteration #{:d}: Correspondences {:d}, Fitness {:.4f}, "
                "RMSE {:.4f}",
                i, result.correspondence_set_.size(), result.fitness_,
                result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_, transformation);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}

RegistrationResult RegistrationGeneralizedICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForGeneralizedICP
                &estimation /* = TransformationEstimationForGeneralizedICP()*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogError(
                "GeneralizedICP require pre-computed normal vectors for target "
                "and source PointClouds.");
    }

    auto source_c = InitializePointCloudForGeneralizedICP(source);
    auto target_c = InitializePointCloudForGeneralizedICP(target);
    return PrivateRegistrationICP(*source_c, *target_c,
                                  max_correspondence_distance, init, estimation,
                                  criteria);
}
}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
