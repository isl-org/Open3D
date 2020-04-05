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

#include "Open3D/Registration/Registration.h"

#include <cstdlib>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Registration/Feature.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {

namespace {
using namespace registration;

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

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        double error2_private = 0.0;
        CorrespondenceSet correspondence_set_private;
#ifdef _OPENMP
#pragma omp for nowait
#endif
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
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (int i = 0; i < (int)correspondence_set_private.size(); i++) {
                result.correspondence_set_.push_back(
                        correspondence_set_private[i]);
            }
            error2 += error2_private;
        }
#ifdef _OPENMP
    }
#endif

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

RegistrationResult EvaluateRANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    RegistrationResult result(transformation);
    double error2 = 0.0;
    int good = 0;
    double max_dis2 = max_correspondence_distance * max_correspondence_distance;
    for (const auto &c : corres) {
        double dis2 =
                (source.points_[c[0]] - target.points_[c[1]]).squaredNorm();
        if (dis2 < max_dis2) {
            good++;
            error2 += dis2;
            result.correspondence_set_.push_back(c);
        }
    }
    if (good == 0) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        result.fitness_ = (double)good / (double)corres.size();
        result.inlier_rmse_ = std::sqrt(error2 / (double)good);
    }
    return result;
}

}  // unnamed namespace

namespace registration {
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d
                &transformation /* = Eigen::Matrix4d::Identity()*/) {
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (transformation.isIdentity() == false) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
}

RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init /* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }
    if ((estimation.GetTransformationEstimationType() ==
                 TransformationEstimationType::PointToPlane ||
         estimation.GetTransformationEstimationType() ==
                 TransformationEstimationType::ColoredICP) &&
        (!source.HasNormals() || !target.HasNormals())) {
        utility::LogError(
                "TransformationEstimationPointToPlane and "
                "TransformationEstimationColoredICP "
                "require pre-computed normal vectors.");
    }

    Eigen::Matrix4d transformation = init;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(target);
    geometry::PointCloud pcd = source;
    if (init.isIdentity() == false) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4d update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
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

RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 6*/,
        const RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || (int)corres.size() < ransac_n ||
        max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }
    Eigen::Matrix4d transformation;
    CorrespondenceSet ransac_corres(ransac_n);
    RegistrationResult result;

    for (int itr = 0;
         itr < criteria.max_iteration_ && itr < criteria.max_validation_;
         itr++) {
        for (int j = 0; j < ransac_n; j++) {
            ransac_corres[j] = corres[utility::UniformRandInt(
                    0, static_cast<int>(corres.size()) - 1)];
        }
        transformation =
                estimation.ComputeTransformation(source, target, ransac_corres);
        geometry::PointCloud pcd = source;
        pcd.Transform(transformation);
        auto this_result = EvaluateRANSACBasedOnCorrespondence(
                pcd, target, corres, max_correspondence_distance,
                transformation);
        if (this_result.fitness_ > result.fitness_ ||
            (this_result.fitness_ == result.fitness_ &&
             this_result.inlier_rmse_ < result.inlier_rmse_)) {
            result = this_result;
        }
    }
    utility::LogDebug("RANSAC: Fitness {:e}, RMSE {:e}", result.fitness_,
                      result.inlier_rmse_);
    return result;
}

RegistrationResult RegistrationRANSACBasedOnFeatureMatching(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Feature &source_feature,
        const Feature &target_feature,
        double max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 4*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers /* = {}*/,
        const RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }

    RegistrationResult result;
    int total_validation = 0;
    bool finished_validation = false;
    int num_similar_features = 1;
    std::vector<std::vector<int>> similar_features(source.points_.size());

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        CorrespondenceSet ransac_corres(ransac_n);
        geometry::KDTreeFlann kdtree(target);
        geometry::KDTreeFlann kdtree_feature(target_feature);
        RegistrationResult result_private;

#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (!finished_validation) {
                std::vector<double> dists(num_similar_features);
                Eigen::Matrix4d transformation;
                for (int j = 0; j < ransac_n; j++) {
                    int source_sample_id = utility::UniformRandInt(
                            0, static_cast<int>(source.points_.size()) - 1);
                    if (similar_features[source_sample_id].empty()) {
                        std::vector<int> indices(num_similar_features);
                        kdtree_feature.SearchKNN(
                                Eigen::VectorXd(source_feature.data_.col(
                                        source_sample_id)),
                                num_similar_features, indices, dists);
#ifdef _OPENMP
#pragma omp critical
#endif
                        { similar_features[source_sample_id] = indices; }
                    }
                    ransac_corres[j](0) = source_sample_id;
                    if (num_similar_features == 1)
                        ransac_corres[j](1) =
                                similar_features[source_sample_id][0];
                    else {
                        ransac_corres[j](1) = similar_features
                                [source_sample_id][utility::UniformRandInt(
                                        0, num_similar_features - 1)];
                    }
                }
                bool check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == false &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                transformation = estimation.ComputeTransformation(
                        source, target, ransac_corres);
                check = true;
                for (const auto &checker : checkers) {
                    if (checker.get().require_pointcloud_alignment_ == true &&
                        checker.get().Check(source, target, ransac_corres,
                                            transformation) == false) {
                        check = false;
                        break;
                    }
                }
                if (check == false) continue;
                geometry::PointCloud pcd = source;
                pcd.Transform(transformation);
                auto this_result = GetRegistrationResultAndCorrespondences(
                        pcd, target, kdtree, max_correspondence_distance,
                        transformation);
                if (this_result.fitness_ > result_private.fitness_ ||
                    (this_result.fitness_ == result_private.fitness_ &&
                     this_result.inlier_rmse_ < result_private.inlier_rmse_)) {
                    result_private = this_result;
                }
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    total_validation = total_validation + 1;
                    if (total_validation >= criteria.max_validation_)
                        finished_validation = true;
                }
            }  // end of if statement
        }      // end of for-loop
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (result_private.fitness_ > result.fitness_ ||
                (result_private.fitness_ == result.fitness_ &&
                 result_private.inlier_rmse_ < result.inlier_rmse_)) {
                result = result_private;
            }
        }
#ifdef _OPENMP
    }
#endif
    utility::LogDebug("total_validation : {:d}", total_validation);
    utility::LogDebug("RANSAC: Fitness {:e}, RMSE {:e}", result.fitness_,
                      result.inlier_rmse_);
    return result;
}

Eigen::Matrix6d GetInformationMatrixFromPointClouds(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        double max_correspondence_distance,
        const Eigen::Matrix4d &transformation) {
    geometry::PointCloud pcd = source;
    if (transformation.isIdentity() == false) {
        pcd.Transform(transformation);
    }
    RegistrationResult result;
    geometry::KDTreeFlann target_kdtree(target);
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, target_kdtree, max_correspondence_distance,
            transformation);

    // write q^*
    // see http://redwood-data.org/indoor/registration.html
    // note: I comes first in this implementation
    Eigen::Matrix6d GTG = Eigen::Matrix6d::Zero();
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Zero();
        Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int c = 0; c < int(result.correspondence_set_.size()); c++) {
            int t = result.correspondence_set_[c](1);
            double x = target.points_[t](0);
            double y = target.points_[t](1);
            double z = target.points_[t](2);
            G_r_private.setZero();
            G_r_private(1) = z;
            G_r_private(2) = -y;
            G_r_private(3) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = -z;
            G_r_private(2) = x;
            G_r_private(4) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
            G_r_private.setZero();
            G_r_private(0) = y;
            G_r_private(1) = -x;
            G_r_private(5) = 1.0;
            GTG_private.noalias() += G_r_private * G_r_private.transpose();
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        { GTG += GTG_private; }
#ifdef _OPENMP
    }
#endif
    return GTG;
}

}  // namespace registration
}  // namespace open3d
