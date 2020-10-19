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

class PointCloudWithCovariance : public geometry::PointCloud {
public:
    std::vector<Eigen::Matrix3d> covariances_;
};

std::shared_ptr<PointCloudWithCovariance> InitializePointCloudForGeneralizedICP(
        const geometry::PointCloud &pcd,
        const geometry::KDTreeSearchParamKNN &search_param) {
    utility::LogDebug("InitializePointCloudForGeneralizedICP");
    auto output = std::make_shared<PointCloudWithCovariance>();
    output->points_ = pcd.points_;
    output->normals_ = pcd.normals_;
    output->covariances_.resize(output->points_.size());

    /// TODO: This literally duplicates the normal estimation. For now, just for
    /// the sake of experimenation, re-compute the covariance matrix at each
    /// point using just the 20 neighbors. As defined in the original papaer
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*output);
#pragma omp parallel for
    for (int i = 0; i < (int)output->points_.size(); i++) {
        auto &cov = output->covariances_[i];
        cov.setZero();

        Eigen::Vector3d mean;
        Eigen::Matrix3d covariance;
        std::vector<int> indices;
        std::vector<double> distance2;

        if (kdtree.Search(output->points_[i], search_param, indices,
                          distance2) >= 3) {
            std::tie(mean, covariance) =
                    utility::ComputeMeanAndCovariance(output->points_, indices);
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(
                    covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Vector3d values{1, 1, 1e-3};
            cov = svd.matrixU() * values.asDiagonal() *
                  svd.matrixV().transpose();
        }
    }
    return output;
}

RegistrationResult GetRegistrationResultAndCorrespondences(
        const geometry::PointCloud &source,
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
    if (corres.empty()) {
        return 0.0;
    }
    double err = 0.0;
    for (const auto &c : corres) {
        err += (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
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
                const Eigen::Vector3d &vs = source_c.points_[corres[i][0]];
                const Eigen::Matrix3d &Cs = source_c.covariances_[corres[i][1]];
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

    const int n_neighbors = 20;
    auto search_param = geometry::KDTreeSearchParamKNN(n_neighbors);
    auto pcd = *InitializePointCloudForGeneralizedICP(source, search_param);
    auto target_c =
            *InitializePointCloudForGeneralizedICP(target, search_param);

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target_c);
    if (!init.isIdentity()) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target_c, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug(
                "GICP Iteration #{:d}: Correspondences {:d}, Fitness {:.4f}, "
                "RMSE {:.4f}",
                i, result.correspondence_set_.size(), result.fitness_,
                result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target_c, result.correspondence_set_, transformation);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target_c, kdtree, max_correspondence_distance,
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

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
