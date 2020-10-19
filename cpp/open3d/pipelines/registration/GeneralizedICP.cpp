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
        const geometry::KDTreeSearchParamHybrid &search_param) {
    utility::LogDebug("InitializePointCloudForGeneralizedICP");
    (void)search_param;
    auto output = std::make_shared<PointCloudWithCovariance>();
    output->points_ = pcd.points_;
    output->normals_ = pcd.normals_;
    return output;
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
                (void)Cs;
                (void)Ct;
                const Eigen::Vector3d d = vt - vs;

                // const Eigen::Matrix3d M = Ct + T * Cs * T.transpose();
                const auto M = Eigen::Matrix3d::Identity();

                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                J.block<3, 3>(0, 3) = -utility::SkewMatrix(vs);
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
        double max_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimationForGeneralizedICP
                &estimation /* = TransformationEstimationForGeneralizedICP()*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogError(
                "GeneralizedICP require pre-computed normal vectors for target "
                "and source PointClouds.");
    }
    auto search_param =
            geometry::KDTreeSearchParamHybrid(max_distance * 2.0, 30);
    auto source_c = InitializePointCloudForGeneralizedICP(target, search_param);
    auto target_c = InitializePointCloudForGeneralizedICP(source, search_param);
    return RegistrationICP(*source_c, *target_c, max_distance, init, estimation,
                           criteria);
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
