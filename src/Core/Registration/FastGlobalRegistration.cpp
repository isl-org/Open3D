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

#include "FastGlobalRegistration.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Registration/Registration.h>
#include <Core/Registration/Feature.h>
#include <Core/Geometry/KDTreeFlann.h>

#define DIV_FACTOR          1.4      // Division factor used for graduated non-convexity
#define USE_ABSOLUTE_SCALE  0        // Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
#define MAX_CORR_DIST       0.025    // Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
#define ITERATION_NUMBER    64       // Maximum number of iteration
#define TUPLE_SCALE         0.95     // Similarity measure used for tuples of feature points.
#define TUPLE_MAX_CNT       1000     // Maximum tuple numbers.

namespace three {

namespace {

std::vector<Eigen::Vector3d> Means;
double GlobalScale;
double StartScale;
std::vector<std::shared_ptr<PointCloud>> pointcloud_;
std::vector<std::shared_ptr<Feature>> features_;
std::vector<std::pair<int, int>> corres_;
Eigen::Matrix4d TransOutput_;

void AdvancedMatching()
{
    int fi = 0;
    int fj = 1;

    printf("Advanced matching : [%d - %d]\n", fi, fj);
    bool swapped = false;

    if (pointcloud_[fj]->points_.size() > pointcloud_[fi]->points_.size())
    {
        int temp = fi;
        fi = fj;
        fj = temp;
        swapped = true;
    }

    int nPti = pointcloud_[fi]->points_.size();
    int nPtj = pointcloud_[fj]->points_.size();

    ///////////////////////////
    /// BUILD FLANNTREE
    ///////////////////////////

    // build FLANNTree - fi
    KDTreeFlann feature_tree_i(*features_[fi]);
    KDTreeFlann feature_tree_j(*features_[fj]);

    bool crosscheck = true;
    bool tuple = true;

    std::vector<int> corres_K, corres_K2;
    std::vector<double> dis;
    std::vector<int> ind;

    std::vector<std::pair<int, int>> corres;
    std::vector<std::pair<int, int>> corres_cross;
    std::vector<std::pair<int, int>> corres_ij;
    std::vector<std::pair<int, int>> corres_ji;

    ///////////////////////////
    /// INITIAL MATCHING
    ///////////////////////////

    std::vector<int> i_to_j(nPti, -1);
    for (int j = 0; j < nPtj; j++)
    {
        feature_tree_i.SearchKNN(features_[fj]->data_.row(j), 1, corres_K, dis);
        int i = corres_K[0];
        if (i_to_j[i] == -1)
        {
            feature_tree_j.SearchKNN(features_[fi]->data_.row(i), 1, corres_K, dis);
            int ij = corres_K[0];
            i_to_j[i] = ij;
        }
        corres_ji.push_back(std::pair<int, int>(i, j));
    }

    for (int i = 0; i < nPti; i++)
    {
        if (i_to_j[i] != -1)
            corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
    }

    int ncorres_ij = corres_ij.size();
    int ncorres_ji = corres_ji.size();

    // corres = corres_ij + corres_ji;
    for (int i = 0; i < ncorres_ij; ++i)
        corres.push_back(std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
    for (int j = 0; j < ncorres_ji; ++j)
        corres.push_back(std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

    printf("points are remained : %d\n", (int)corres.size());

    ///////////////////////////
    /// CROSS CHECK
    /// input : corres_ij, corres_ji
    /// output : corres
    ///////////////////////////
    if (crosscheck)
    {
        printf("\t[cross check] ");

        // build data structure for cross check
        corres.clear();
        corres_cross.clear();
        std::vector<std::vector<int>> Mi(nPti);
        std::vector<std::vector<int>> Mj(nPtj);

        int ci, cj;
        for (int i = 0; i < ncorres_ij; ++i)
        {
            ci = corres_ij[i].first;
            cj = corres_ij[i].second;
            Mi[ci].push_back(cj);
        }
        for (int j = 0; j < ncorres_ji; ++j)
        {
            ci = corres_ji[j].first;
            cj = corres_ji[j].second;
            Mj[cj].push_back(ci);
        }

        // cross check
        for (int i = 0; i < nPti; ++i)
        {
            for (int ii = 0; ii < Mi[i].size(); ++ii)
            {
                int j = Mi[i][ii];
                for (int jj = 0; jj < Mj[j].size(); ++jj)
                {
                    if (Mj[j][jj] == i)
                    {
                        corres.push_back(std::pair<int, int>(i, j));
                        corres_cross.push_back(std::pair<int, int>(i, j));
                    }
                }
            }
        }
        printf("points are remained : %d\n", (int)corres.size());
    }

    ///////////////////////////
    /// TUPLE CONSTRAINT
    /// input : corres
    /// output : corres
    ///////////////////////////
    if (tuple)
    {
        srand(time(NULL));

        printf("\t[tuple constraint] ");
        int rand0, rand1, rand2;
        int idi0, idi1, idi2;
        int idj0, idj1, idj2;
        float scale = TUPLE_SCALE;
        int ncorr = corres.size();
        int number_of_trial = ncorr * 100;
        std::vector<std::pair<int, int>> corres_tuple;

        int cnt = 0;
        int i;
        for (i = 0; i < number_of_trial; i++)
        {
            rand0 = rand() % ncorr;
            rand1 = rand() % ncorr;
            rand2 = rand() % ncorr;

            idi0 = corres[rand0].first;
            idj0 = corres[rand0].second;
            idi1 = corres[rand1].first;
            idj1 = corres[rand1].second;
            idi2 = corres[rand2].first;
            idj2 = corres[rand2].second;

            // collect 3 points from i-th fragment
            Eigen::Vector3d pti0 = pointcloud_[fi]->points_[idi0];
            Eigen::Vector3d pti1 = pointcloud_[fi]->points_[idi1];
            Eigen::Vector3d pti2 = pointcloud_[fi]->points_[idi2];

            float li0 = (pti0 - pti1).norm();
            float li1 = (pti1 - pti2).norm();
            float li2 = (pti2 - pti0).norm();

            // collect 3 points from j-th fragment
            Eigen::Vector3d ptj0 = pointcloud_[fj]->points_[idj0];
            Eigen::Vector3d ptj1 = pointcloud_[fj]->points_[idj1];
            Eigen::Vector3d ptj2 = pointcloud_[fj]->points_[idj2];

            float lj0 = (ptj0 - ptj1).norm();
            float lj1 = (ptj1 - ptj2).norm();
            float lj2 = (ptj2 - ptj0).norm();

            if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
                (li1 * scale < lj1) && (lj1 < li1 / scale) &&
                (li2 * scale < lj2) && (lj2 < li2 / scale))
            {
                corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
                corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
                corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
                cnt++;
            }

            if (cnt >= TUPLE_MAX_CNT)
                break;
        }

        printf("%d tuples (%d trial, %d actual).\n", cnt, number_of_trial, i);
        corres.clear();

        for (int i = 0; i < corres_tuple.size(); ++i)
            corres.push_back(std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
    }

    if (swapped)
    {
        std::vector<std::pair<int, int>> temp;
        for (int i = 0; i < corres.size(); i++)
            temp.push_back(std::pair<int, int>(corres[i].second, corres[i].first));
        corres.clear();
        corres = temp;
    }

    printf("\t[final] matches %d.\n", (int)corres.size());
    corres_ = corres;
}


// Normalize scale of points.
// X' = (X-\mu)/scale
void NormalizePoints()
{
    int num = 2;
    float scale = 0;

    Means.clear();

    for (int i = 0; i < num; ++i)
    {
        float max_scale = 0;

        // compute mean
        Eigen::Vector3d mean;
        mean.setZero();

        int npti = pointcloud_[i]->points_.size();
        for (int ii = 0; ii < npti; ++ii)
        {
            mean = mean + pointcloud_[i]->points_[ii];
        }
        mean = mean / npti;
        Means.push_back(mean);

        printf("normalize points :: mean = [%f %f %f]\n", mean(0), mean(1), mean(2));

        for (int ii = 0; ii < npti; ++ii)
        {
            pointcloud_[i]->points_[ii] -= mean;
        }

        // compute scale
        for (int ii = 0; ii < npti; ++ii)
        {
            Eigen::Vector3d p(pointcloud_[i]->points_[ii]);
            float temp = p.norm(); // because we extract mean in the previous stage.
            if (temp > max_scale)
                max_scale = temp;
        }

        if (max_scale > scale)
            scale = max_scale;
    }

    // mean of the scale variation
    if (USE_ABSOLUTE_SCALE) {
        GlobalScale = 1.0f;
        StartScale = scale;
    } else {
        GlobalScale = scale; // second choice: we keep the maximum scale.
        StartScale = 1.0f;
    }
    printf("normalize points :: global scale : %f\n", GlobalScale);

    for (int i = 0; i < num; ++i)
    {
        int npti = pointcloud_[i]->points_.size();
        for (int ii = 0; ii < npti; ++ii)
        {
            pointcloud_[i]->points_[ii] /= GlobalScale;
        }
    }
}

double OptimizePairwise(bool decrease_mu_, int numIter_)
{
    printf("Pairwise rigid pose optimization\n");

    double par;
    int numIter = numIter_;
    TransOutput_ = Eigen::Matrix4d::Identity();

    par = StartScale;

    int i = 0;
    int j = 1;

    // make another copy of pointcloud_[j].
    std::vector<Eigen::Vector3d> pcj_copy;
    int npcj = pointcloud_[j]->points_.size();
    pcj_copy.resize(npcj);
    for (int cnt = 0; cnt < npcj; cnt++)
        pcj_copy[cnt] = pointcloud_[j]->points_[cnt];

    if (corres_.size() < 10)
        return -1;

    std::vector<double> s(corres_.size(), 1.0);

    Eigen::Matrix4d trans;
    trans.setIdentity();

    for (int itr = 0; itr < numIter; itr++) {

        // graduated non-convexity.
        if (decrease_mu_)
        {
            if (itr % 4 == 0 && par > MAX_CORR_DIST) {
                par /= DIV_FACTOR;
            }
        }

        const int nvariable = 6;    // 3 for rotation and 3 for translation
        Eigen::MatrixXd JTJ(nvariable, nvariable);
        Eigen::MatrixXd JTr(nvariable, 1);
        Eigen::MatrixXd J(nvariable, 1);
        JTJ.setZero();
        JTr.setZero();

        double r;
        double r2 = 0.0;

        for (int c = 0; c < corres_.size(); c++) {
            int ii = corres_[c].first;
            int jj = corres_[c].second;
            Eigen::Vector3d p, q;
            p = pointcloud_[i]->points_[ii];
            q = pcj_copy[jj];
            Eigen::Vector3d rpq = p - q;

            int c2 = c;

            float temp = par / (rpq.dot(rpq) + par);
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

        Eigen::MatrixXd result(nvariable, 1);
        result = -JTJ.llt().solve(JTr);

        Eigen::Affine3d aff_mat;
        aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(result(2), Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
        aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

        Eigen::Matrix4d delta = aff_mat.matrix().cast<double>();

        trans = delta * trans;

        // transform point clouds
        Eigen::Matrix3d R = delta.block<3, 3>(0, 0);
        Eigen::Vector3d t = delta.block<3, 1>(0, 3);
        for (int cnt = 0; cnt < npcj; cnt++)
            pcj_copy[cnt] = R * pcj_copy[cnt] + t;

    }

    TransOutput_ = trans * TransOutput_;
    return par;
}

// Below line indicates how the transformation matrix aligns two point clouds
// e.g. T * pointcloud_[1] is aligned with pointcloud_[0].
// '2' indicates that there are two point cloud fragments.
Eigen::Matrix4d GetTrans()
{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    R = TransOutput_.block<3, 3>(0, 0);
    t = TransOutput_.block<3, 1>(0, 3);

    Eigen::Matrix4d transtemp;
    transtemp.fill(0.0f);

    transtemp.block<3, 3>(0, 0) = R;
    transtemp.block<3, 1>(0, 3) = -R*Means[1] + t*GlobalScale + Means[0];
    transtemp(3, 3) = 1;

    return transtemp;
}

}    // unnamed namespace


RegistrationResult FastGlobalRegistration(
        const PointCloud &source, const PointCloud &target,
        const Feature &source_feature, const Feature &target_feature,
        double max_correspondence_distance,
        const Eigen::Matrix4d &init/* = Eigen::Matrix4d::Identity()*/,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria &criteria/* = ICPConvergenceCriteria()*/)
{
    std::shared_ptr<PointCloud> source_copy = std::make_shared<PointCloud>();
    std::shared_ptr<PointCloud> target_copy = std::make_shared<PointCloud>();
    *source_copy = source;
    *target_copy = target;
    pointcloud_.push_back(source_copy);
    pointcloud_.push_back(target_copy);

    std::shared_ptr<Feature> source_feature_copy = std::make_shared<Feature>();
    std::shared_ptr<Feature> target_feature_copy = std::make_shared<Feature>();
    *source_feature_copy = source_feature;
    *target_feature_copy = target_feature;
    features_.push_back(source_feature_copy);
    features_.push_back(target_feature_copy);

    NormalizePoints();
    AdvancedMatching();
    OptimizePairwise(true, ITERATION_NUMBER);

    // as the original code T * pointcloud_[1] is aligned with pointcloud_[0].
    // matrix inverse is applied here.
    return RegistrationResult(GetTrans().inverse());
}

}  // namespace three
