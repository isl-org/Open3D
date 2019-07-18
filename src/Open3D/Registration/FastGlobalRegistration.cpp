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

#include "Open3D/Registration/FastGlobalRegistration.h"

#include <ctime>

#include "Open3D/Geometry/KDTreeFlann.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Registration/Feature.h"
#include "Open3D/Registration/Registration.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Eigen.h"

namespace open3d {

namespace {
using namespace registration;

std::vector<std::pair<int, int>> AdvancedMatching(
        const std::vector<geometry::PointCloud>& point_cloud_vec,
        const std::vector<Feature>& features_vec,
        const FastGlobalRegistrationOption& option) {
    // STEP 0) Swap source and target if necessary
    int fi = 0, fj = 1;
    utility::LogDebug("Advanced matching : [{:d} - {:d}]\n", fi, fj);
    bool swapped = false;
    if (point_cloud_vec[fj].points_.size() >
        point_cloud_vec[fi].points_.size()) {
        int temp = fi;
        fi = fj;
        fj = temp;
        swapped = true;
    }

    // STEP 1) Initial matching
    int nPti = int(point_cloud_vec[fi].points_.size());
    int nPtj = int(point_cloud_vec[fj].points_.size());
    geometry::KDTreeFlann feature_tree_i(features_vec[fi]);
    geometry::KDTreeFlann feature_tree_j(features_vec[fj]);
    std::vector<int> corresK;
    std::vector<double> dis;
    std::vector<std::pair<int, int>> corres;
    std::vector<std::pair<int, int>> corres_ij;
    std::vector<std::pair<int, int>> corres_ji;
    std::vector<int> i_to_j(nPti, -1);
    for (int j = 0; j < nPtj; j++) {
        feature_tree_i.SearchKNN(Eigen::VectorXd(features_vec[fj].data_.col(j)),
                                 1, corresK, dis);
        int i = corresK[0];
        if (i_to_j[i] == -1) {
            feature_tree_j.SearchKNN(
                    Eigen::VectorXd(features_vec[fi].data_.col(i)), 1, corresK,
                    dis);
            int ij = corresK[0];
            i_to_j[i] = ij;
        }
        corres_ji.push_back(std::pair<int, int>(i, j));
    }
    for (int i = 0; i < nPti; i++) {
        if (i_to_j[i] != -1)
            corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
    }
    int ncorres_ij = int(corres_ij.size());
    int ncorres_ji = int(corres_ji.size());
    for (int i = 0; i < ncorres_ij; ++i)
        corres.push_back(
                std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
    for (int j = 0; j < ncorres_ji; ++j)
        corres.push_back(
                std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));
    utility::LogDebug("points are remained : {:d}\n", (int)corres.size());

    // STEP 2) CROSS CHECK
    utility::LogDebug("\t[cross check] ");
    std::vector<std::pair<int, int>> corres_cross;
    std::vector<std::vector<int>> Mi(nPti), Mj(nPtj);
    int ci, cj;
    for (int i = 0; i < ncorres_ij; ++i) {
        ci = corres_ij[i].first;
        cj = corres_ij[i].second;
        Mi[ci].push_back(cj);
    }
    for (int j = 0; j < ncorres_ji; ++j) {
        ci = corres_ji[j].first;
        cj = corres_ji[j].second;
        Mj[cj].push_back(ci);
    }
    for (int i = 0; i < nPti; ++i) {
        for (size_t ii = 0; ii < Mi[i].size(); ++ii) {
            int j = Mi[i][ii];
            for (size_t jj = 0; jj < Mj[j].size(); ++jj) {
                if (Mj[j][jj] == i)
                    corres_cross.push_back(std::pair<int, int>(i, j));
            }
        }
    }
    utility::LogDebug("points are remained : %d\n", (int)corres_cross.size());

    // STEP 3) TUPLE CONSTRAINT
    utility::LogDebug("\t[tuple constraint] ");
    std::srand((unsigned int)std::time(0));
    int rand0, rand1, rand2, i, cnt = 0;
    int idi0, idi1, idi2, idj0, idj1, idj2;
    double scale = option.tuple_scale_;
    int ncorr = static_cast<int>(corres_cross.size());
    int number_of_trial = ncorr * 100;
    std::vector<std::pair<int, int>> corres_tuple;
    for (i = 0; i < number_of_trial; i++) {
        rand0 = rand() % ncorr;
        rand1 = rand() % ncorr;
        rand2 = rand() % ncorr;
        idi0 = corres_cross[rand0].first;
        idj0 = corres_cross[rand0].second;
        idi1 = corres_cross[rand1].first;
        idj1 = corres_cross[rand1].second;
        idi2 = corres_cross[rand2].first;
        idj2 = corres_cross[rand2].second;

        // collect 3 points from i-th fragment
        Eigen::Vector3d pti0 = point_cloud_vec[fi].points_[idi0];
        Eigen::Vector3d pti1 = point_cloud_vec[fi].points_[idi1];
        Eigen::Vector3d pti2 = point_cloud_vec[fi].points_[idi2];
        double li0 = (pti0 - pti1).norm();
        double li1 = (pti1 - pti2).norm();
        double li2 = (pti2 - pti0).norm();

        // collect 3 points from j-th fragment
        Eigen::Vector3d ptj0 = point_cloud_vec[fj].points_[idj0];
        Eigen::Vector3d ptj1 = point_cloud_vec[fj].points_[idj1];
        Eigen::Vector3d ptj2 = point_cloud_vec[fj].points_[idj2];
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
    utility::LogDebug("{:d} tuples ({:d} trial, {:d} actual).\n", cnt,
                      number_of_trial, i);

    if (swapped) {
        std::vector<std::pair<int, int>> temp;
        for (size_t i = 0; i < corres_tuple.size(); i++)
            temp.push_back(std::pair<int, int>(corres_tuple[i].second,
                                               corres_tuple[i].first));
        corres_tuple.clear();
        corres_tuple = temp;
    }
    utility::LogDebug("\t[final] matches {:d}.\n", (int)corres_tuple.size());
    return corres_tuple;
}

// Normalize scale of points. X' = (X-\mu)/scale
std::tuple<std::vector<Eigen::Vector3d>, double, double> NormalizePointCloud(
        std::vector<geometry::PointCloud>& point_cloud_vec,
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

        utility::LogDebug("normalize points :: mean = [{:f} {:f} {:f}]\n",
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
    utility::LogDebug("normalize points :: global scale : {:f}\n",
                      scale_global);

    for (int i = 0; i < num; ++i) {
        int npti = static_cast<int>(point_cloud_vec[i].points_.size());
        for (int ii = 0; ii < npti; ++ii) {
            point_cloud_vec[i].points_[ii] /= scale_global;
        }
    }
    return std::make_tuple(pcd_mean_vec, scale_global, scale_start);
}

Eigen::Matrix4d OptimizePairwiseRegistration(
        const std::vector<geometry::PointCloud>& point_cloud_vec,
        const std::vector<std::pair<int, int>>& corres,
        double scale_start,
        const FastGlobalRegistrationOption& option) {
    utility::LogDebug("Pairwise rigid pose optimization\n");
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
Eigen::Matrix4d GetTransformationOriginalScale(
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

}  // unnamed namespace

namespace registration {
RegistrationResult FastGlobalRegistration(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const Feature& source_feature,
        const Feature& target_feature,
        const FastGlobalRegistrationOption& option /* =
        FastGlobalRegistrationOption()*/) {
    std::vector<geometry::PointCloud> point_cloud_vec;
    point_cloud_vec.push_back(source);
    point_cloud_vec.push_back(target);

    std::vector<Feature> features_vec;
    features_vec.push_back(source_feature);
    features_vec.push_back(target_feature);

    double scale_global, scale_start;
    std::vector<Eigen::Vector3d> pcd_mean_vec;
    std::tie(pcd_mean_vec, scale_global, scale_start) =
            NormalizePointCloud(point_cloud_vec, option);
    std::vector<std::pair<int, int>> corres;
    corres = AdvancedMatching(point_cloud_vec, features_vec, option);
    Eigen::Matrix4d transformation;
    transformation = OptimizePairwiseRegistration(point_cloud_vec, corres,
                                                  scale_global, option);

    // as the original code T * point_cloud_vec[1] is aligned with
    // point_cloud_vec[0] matrix inverse is applied here.
    // clang-format off
    return RegistrationResult(GetTransformationOriginalScale(transformation,
                                                             pcd_mean_vec,
                                                             scale_global).inverse());
    // clang-format on
}

}  // namespace registration
}  // namespace open3d
