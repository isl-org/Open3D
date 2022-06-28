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

#include "open3d/pipelines/registration/FastGlobalRegistration.h"

#include <map>

#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Feature.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace pipelines {
namespace registration {

static std::vector<std::pair<int, int>> InitialMatching(
        const Feature& src_features, const Feature& dst_features) {
    geometry::KDTreeFlann src_feature_tree(src_features);
    geometry::KDTreeFlann dst_feature_tree(dst_features);
    std::map<int, int> corres_ij;
    std::vector<int> corres_ji(dst_features.data_.cols(), -1);

#pragma omp for nowait
    for (int j = 0; j < dst_features.data_.cols(); j++) {
        std::vector<int> corres_tmp(1);
        std::vector<double> dist_tmp(1);
        src_feature_tree.SearchKNN(Eigen::VectorXd(dst_features.data_.col(j)),
                                   1, corres_tmp, dist_tmp);
        int i = corres_tmp[0];
        corres_ji[j] = i;
        if (corres_ij.find(i) == corres_ij.end()) {
            // set a temp value to prevent other threads recomputing
            // corres_ij[i] until the following dst_feature_tree.SearchKNN()
            // call completes. There is still a race condition but the result
            // would be fine since both threads will compute the same result
            corres_ij[i] = -1;
            dst_feature_tree.SearchKNN(
                    Eigen::VectorXd(src_features.data_.col(i)), 1, corres_tmp,
                    dist_tmp);
            corres_ij[i] = corres_tmp[0];
        }
    }

    utility::LogDebug("\t[cross check] ");
    std::vector<std::pair<int, int>> corres_cross;
    for (const std::pair<const int, int>& ij : corres_ij) {
        if (corres_ji[ij.second] == ij.first) corres_cross.push_back(ij);
    }
    utility::LogDebug("Initial matchings : {}", corres_cross.size());
    return corres_cross;
}

static std::vector<std::pair<int, int>> AdvancedMatching(
        const geometry::PointCloud& src_point_cloud,
        const geometry::PointCloud& dst_point_cloud,
        const std::vector<std::pair<int, int>>& corres_cross,
        const FastGlobalRegistrationOption& option) {
    utility::LogDebug("\t[tuple constraint] ");
    int rand0, rand1, rand2, i, cnt = 0;
    int idi0, idi1, idi2, idj0, idj1, idj2;
    double scale = option.tuple_scale_;
    int ncorr = static_cast<int>(corres_cross.size());
    int number_of_trial = ncorr * 100;

    utility::random::UniformIntGenerator rand_generator(0, ncorr - 1);
    std::vector<std::pair<int, int>> corres_tuple;
    for (i = 0; i < number_of_trial; i++) {
        rand0 = rand_generator();
        rand1 = rand_generator();
        rand2 = rand_generator();
        idi0 = corres_cross[rand0].first;
        idj0 = corres_cross[rand0].second;
        idi1 = corres_cross[rand1].first;
        idj1 = corres_cross[rand1].second;
        idi2 = corres_cross[rand2].first;
        idj2 = corres_cross[rand2].second;

        // collect 3 points from source fragment
        Eigen::Vector3d pti0 = src_point_cloud.points_[idi0];
        Eigen::Vector3d pti1 = src_point_cloud.points_[idi1];
        Eigen::Vector3d pti2 = src_point_cloud.points_[idi2];
        double li0 = (pti0 - pti1).norm();
        double li1 = (pti1 - pti2).norm();
        double li2 = (pti2 - pti0).norm();

        // collect 3 points from dest fragment
        Eigen::Vector3d ptj0 = dst_point_cloud.points_[idj0];
        Eigen::Vector3d ptj1 = dst_point_cloud.points_[idj1];
        Eigen::Vector3d ptj2 = dst_point_cloud.points_[idj2];
        double lj0 = (ptj0 - ptj1).norm();
        double lj1 = (ptj1 - ptj2).norm();
        double lj2 = (ptj2 - ptj0).norm();

        // check tuple constraint
        if ((li0 * scale < lj0) && (lj0 < li0 / scale) && (li1 * scale < lj1) &&
            (lj1 < li1 / scale) && (li2 * scale < lj2) && (lj2 < li2 / scale)) {
            corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
            corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
            corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
            cnt++;
        }
        if (cnt >= option.maximum_tuple_count_) break;
    }
    utility::LogDebug("{:d} tuples ({:d} trial, {:d} actual).", cnt,
                      number_of_trial, i);

    utility::LogDebug("\t[final] matches {:d}.", (int)corres_tuple.size());
    return corres_tuple;
}

// Normalize scale of points. X' = (X-\mu)/scale
static std::tuple<std::vector<Eigen::Vector3d>, double, double>
NormalizePointCloud(std::vector<geometry::PointCloud>& point_cloud_vec,
                    const FastGlobalRegistrationOption& option) {
    int num = 2;
    double scale = 0;
    std::vector<Eigen::Vector3d> pcd_mean_vec;
    double scale_global, scale_start;

    for (int i = 0; i < num; ++i) {
        double max_scale = 0.0;
        Eigen::Vector3d mean;
        mean.setZero();

        int npti = static_cast<int>(point_cloud_vec[i].points_.size());
        for (int ii = 0; ii < npti; ++ii)
            mean = mean + point_cloud_vec[i].points_[ii];
        mean = mean / npti;
        pcd_mean_vec.push_back(mean);

        utility::LogDebug("normalize points :: mean = [{:f} {:f} {:f}]",
                          mean(0), mean(1), mean(2));
        for (int ii = 0; ii < npti; ++ii)
            point_cloud_vec[i].points_[ii] -= mean;

        for (int ii = 0; ii < npti; ++ii) {
            Eigen::Vector3d p(point_cloud_vec[i].points_[ii]);
            double temp = p.norm();
            if (temp > max_scale) max_scale = temp;
        }
        if (max_scale > scale) scale = max_scale;
    }

    if (option.use_absolute_scale_) {
        scale_global = 1.0;
        scale_start = scale;
    } else {
        scale_global = scale;
        scale_start = 1.0;
    }
    utility::LogDebug("normalize points :: global scale : {:f}", scale_global);
    if (scale_global <= 0) {
        utility::LogError("Invalid scale_global: {}, it must be > 0.",
                          scale_global);
    }

    for (int i = 0; i < num; ++i) {
        int npti = static_cast<int>(point_cloud_vec[i].points_.size());
        for (int ii = 0; ii < npti; ++ii) {
            point_cloud_vec[i].points_[ii] /= scale_global;
        }
    }
    return std::make_tuple(pcd_mean_vec, scale_global, scale_start);
}

static Eigen::Matrix4d OptimizePairwiseRegistration(
        const std::vector<geometry::PointCloud>& point_cloud_vec,
        const std::vector<std::pair<int, int>>& corres,
        double scale_start,
        const FastGlobalRegistrationOption& option) {
    utility::LogDebug("Pairwise rigid pose optimization");
    double par = scale_start;
    int numIter = option.iteration_number_;

    int i = 0, j = 1;
    geometry::PointCloud point_cloud_copy_j = point_cloud_vec[j];

    if (corres.size() < 10) return Eigen::Matrix4d::Identity();

    std::vector<double> s(corres.size(), 1.0);
    Eigen::Matrix4d trans;
    trans.setIdentity();

    for (int itr = 0; itr < numIter; itr++) {
        const int nvariable = 6;
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();
        double r = 0.0, r2 = 0.0;
        (void)r2;  // r2 is not used for now. Suppress clang warning.

        for (size_t c = 0; c < corres.size(); c++) {
            int ii = corres[c].first;
            int jj = corres[c].second;
            Eigen::Vector3d p, q;
            p = point_cloud_vec[i].points_[ii];
            q = point_cloud_copy_j.points_[jj];
            Eigen::Vector3d rpq = p - q;

            size_t c2 = c;
            double temp = par / (rpq.dot(rpq) + par);
            s[c2] = temp * temp;

            J.setZero();
            J(1) = -q(2);
            J(2) = q(1);
            J(3) = -1;
            r = rpq(0);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];

            J.setZero();
            J(2) = -q(0);
            J(0) = q(2);
            J(4) = -1;
            r = rpq(1);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];

            J.setZero();
            J(0) = -q(1);
            J(1) = q(0);
            J(5) = -1;
            r = rpq(2);
            JTJ += J * J.transpose() * s[c2];
            JTr += J * r * s[c2];
            r2 += r * r * s[c2];
            r2 += (par * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));
        }
        (void)r2;  // Fix warning in Clang.
        bool success;
        Eigen::VectorXd result;
        std::tie(success, result) = utility::SolveLinearSystemPSD(-JTJ, JTr);
        Eigen::Matrix4d delta = utility::TransformVector6dToMatrix4d(result);
        trans = delta * trans;
        point_cloud_copy_j.Transform(delta);

        // graduated non-convexity.
        if (option.decrease_mu_) {
            if (itr % 4 == 0 && par > option.maximum_correspondence_distance_) {
                par /= option.division_factor_;
            }
        }
    }
    return trans;
}

// Below line indicates how the transformation matrix aligns two point clouds
// e.g. T * point_cloud_vec[1] is aligned with point_cloud_vec[0].
static Eigen::Matrix4d GetTransformationOriginalScale(
        const Eigen::Matrix4d& transformation,
        const std::vector<Eigen::Vector3d>& pcd_mean_vec,
        const double scale_global) {
    Eigen::Matrix3d R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3d t = transformation.block<3, 1>(0, 3);
    Eigen::Matrix4d transtemp = Eigen::Matrix4d::Zero();
    transtemp.block<3, 3>(0, 0) = R;
    transtemp.block<3, 1>(0, 3) =
            -R * pcd_mean_vec[1] + t * scale_global + pcd_mean_vec[0];
    transtemp(3, 3) = 1;
    return transtemp;
}

RegistrationResult FastGlobalRegistrationBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const FastGlobalRegistrationOption &option /* =
                FastGlobalRegistrationOption()*/) {
    geometry::PointCloud source_orig = source;
    geometry::PointCloud target_orig = target;

    std::vector<geometry::PointCloud> point_cloud_vec;
    point_cloud_vec.push_back(source);
    point_cloud_vec.push_back(target);

    double scale_global, scale_start;
    std::vector<Eigen::Vector3d> pcd_mean_vec;
    std::tie(pcd_mean_vec, scale_global, scale_start) =
            NormalizePointCloud(point_cloud_vec, option);

    std::vector<std::pair<int, int>> corresvec;
    corresvec.reserve(corres.size());
    for (size_t i = 0; i < corres.size(); ++i) {
        corresvec.push_back({corres[i](0), corres[i](1)});
    }

    if (option.tuple_test_) {
        // for AdvancedMatching ensure the first point cloud is the larger one
        if (source.points_.size() > target.points_.size()) {
            corresvec = AdvancedMatching(source, target, corresvec, option);
        } else {
            corresvec = AdvancedMatching(target, source, corresvec, option);
            for (auto& p : corresvec) std::swap(p.first, p.second);
        }
    }

    Eigen::Matrix4d transformation;
    transformation = OptimizePairwiseRegistration(point_cloud_vec, corresvec,
                                                  scale_global, option);

    // as the original code T * point_cloud_vec[1] is aligned with
    // point_cloud_vec[0] matrix inverse is applied here.
    return EvaluateRegistration(
            source_orig, target_orig, option.maximum_correspondence_distance_,
            GetTransformationOriginalScale(transformation, pcd_mean_vec,
                                           scale_global)
                    .inverse());
}

RegistrationResult FastGlobalRegistrationBasedOnFeatureMatching(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const Feature& source_feature,
        const Feature& target_feature,
        const FastGlobalRegistrationOption& option /* =
        FastGlobalRegistrationOption()*/) {
    geometry::PointCloud source_orig = source;
    geometry::PointCloud target_orig = target;

    std::vector<geometry::PointCloud> point_cloud_vec;
    point_cloud_vec.push_back(source);
    point_cloud_vec.push_back(target);

    double scale_global, scale_start;
    std::vector<Eigen::Vector3d> pcd_mean_vec;
    std::tie(pcd_mean_vec, scale_global, scale_start) =
            NormalizePointCloud(point_cloud_vec, option);

    std::vector<std::pair<int, int>> corres;
    if (option.tuple_test_) {
        // for AdvancedMatching ensure the first point cloud is the larger one
        if (source.points_.size() > target.points_.size()) {
            corres = AdvancedMatching(
                    source, target,
                    InitialMatching(source_feature, target_feature), option);
        } else {
            corres = AdvancedMatching(
                    target, source,
                    InitialMatching(target_feature, source_feature), option);
            for (auto& p : corres) std::swap(p.first, p.second);
        }
    } else {
        corres = InitialMatching(source_feature, target_feature);
    }

    Eigen::Matrix4d transformation;
    transformation = OptimizePairwiseRegistration(point_cloud_vec, corres,
                                                  scale_global, option);

    // as the original code T * point_cloud_vec[1] is aligned with
    // point_cloud_vec[0] matrix inverse is applied here.
    return EvaluateRegistration(
            source_orig, target_orig, option.maximum_correspondence_distance_,
            GetTransformationOriginalScale(transformation, pcd_mean_vec,
                                           scale_global)
                    .inverse());
}

}  // namespace registration
}  // namespace pipelines
}  // namespace open3d
