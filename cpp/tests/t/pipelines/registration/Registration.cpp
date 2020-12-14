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

#include "open3d/t/pipelines/registration/Registration.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/io/PointCloudIO.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class RegistrationPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Registration,
                         RegistrationPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class RegistrationPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Registration,
        RegistrationPermuteDevicePairs,
        testing::ValuesIn(RegistrationPermuteDevicePairs::TestCases()));

TEST_P(RegistrationPermuteDevices, EvaluateRegistration) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source_(device);
    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd",
                          source_, {"auto", false, false, true});
    core::Tensor source_points = source_.GetPoints().To(dtype).Copy(device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    t::geometry::PointCloud target_(device);
    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd",
                          target_, {"auto", false, false, true});
    core::Tensor target_points = target_.GetPoints().To(dtype).Copy(device);
    core::Tensor target_normals =
            target_.GetPointNormals().To(dtype).Copy(device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    // CASE 1: Identity Transformation
    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
    // // if using Float64, change float to double in the following vector

    double max_correspondence_dist = 0.02;

    t::pipelines::registration::RegistrationResult evaluation(init_trans);
    evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
            source_device, target_device, max_correspondence_dist, init_trans);
    // Compare with:
    /*
      [Correspondences]: 7552 [CPU] 7550 [GPU]
      Fitness: 0.0379812 [CPU] 0.0379712 [GPU]
      Inlier RMSE: 0.0119891 [CPU] 0.0119853 [GPU]
    */
    int num_corres = evaluation.correspondence_set_.GetShape()[0];
    bool isCorrespondencestrue = (num_corres > 7400);
    bool isFitnesstrue = (evaluation.fitness_ > 0.037);
    bool isInlierRMSEtrue = (evaluation.inlier_rmse_ < 0.02);

    EXPECT_TRUE(isCorrespondencestrue);
    EXPECT_TRUE(isFitnesstrue);
    EXPECT_TRUE(isInlierRMSEtrue);

    std::vector<float> trans_init_vec{
            0.862, 0.011, -0.507, 0.5,  -0.139, 0.967, -0.215, 0.7,
            0.487, 0.255, 0.835,  -1.4, 0.0,    0.0,   0.0,    1.0};
    // Creating Tensor from manual transformation vector
    core::Tensor init_trans_guess(trans_init_vec, {4, 4}, dtype, device);
    evaluation = open3d::t::pipelines::registration::EvaluateRegistration(
            source_device, target_device, max_correspondence_dist,
            init_trans_guess);
    // Compare with:
    /*
      [Correspondences]: 34741 [CPU] 34738 [GPU]
      Fitness: 0.174723 [CPU] 0.174708 [GPU]
      Inlier RMSE: 0.0117711 [CPU] 0.011769 [GPU]
    */

    num_corres = evaluation.correspondence_set_.GetShape()[0];
    isCorrespondencestrue = (num_corres > 34000);
    isFitnesstrue = (evaluation.fitness_ > 0.17);
    isInlierRMSEtrue = (evaluation.inlier_rmse_ < 0.02);

    EXPECT_TRUE(isCorrespondencestrue);
    EXPECT_TRUE(isFitnesstrue);
    EXPECT_TRUE(isInlierRMSEtrue);
}

TEST_P(RegistrationPermuteDevices, RegistrationICP) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud source_(device);
    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd",
                          source_, {"auto", false, false, true});
    core::Tensor source_points = source_.GetPoints().To(dtype).Copy(device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    t::geometry::PointCloud target_(device);
    t::io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd",
                          target_, {"auto", false, false, true});
    core::Tensor target_points = target_.GetPoints().To(dtype).Copy(device);
    core::Tensor target_normals =
            target_.GetPointNormals().To(dtype).Copy(device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    // Identity Transformation
    core::Tensor init_trans = core::Tensor::Eye(4, dtype, device);
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 5;
    double max_correspondence_dist = 0.02;

    // PointToPoint
    auto reg_p2p = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPoint(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));

    // Compare with following:
    /*
        Correspondences: 8381 [CPU]  8375 [GPU]
        Fitness: 0.0421505 [CPU] 0.0421204 [GPU]
        Inlier RMSE: 0.0118882 [CPU] 0.0118829  [GPU]
        Transformation [CPU]
                [[1.0 0.000816210 -0.00129755 0.00125516],
                [-0.000805086 0.999964 0.00854054 -0.00420071],
                [0.00130453 -0.00853954 0.999963 0.0161748],
                [0.0 0.0 0.0 1.0]]
        Transformation [GPU]
                [[0.999999 0.000869017 -0.00127612 0.000955089],
                [-0.000858009 0.999963 0.00857059 -0.00414474],
                [0.00128359 -0.00856941 0.999962 0.0162242],
                [0.0 0.0 0.0 1.0]]
    */
    int num_corres = reg_p2p.correspondence_set_.GetShape()[0];
    bool isCorrespondencestrue = (num_corres > 8300);
    bool isFitnesstrue = (reg_p2p.fitness_ > 0.04);
    bool isInlierRMSEtrue = (reg_p2p.inlier_rmse_ < 0.02);

    EXPECT_TRUE(isCorrespondencestrue);
    EXPECT_TRUE(isFitnesstrue);
    EXPECT_TRUE(isInlierRMSEtrue);

    // PointToPlane
    auto reg_p2plane = open3d::t::pipelines::registration::RegistrationICP(
            source_device, target_device, max_correspondence_dist, init_trans,
            open3d::t::pipelines::registration::
                    TransformationEstimationPointToPlane(),
            open3d::t::pipelines::registration::ICPConvergenceCriteria(
                    relative_fitness, relative_rmse, max_iterations));
    // Compare with following:
    /*
        Correspondences: 9293 [CPU]  9297 [GPU]
        Fitness: 0.0467372 [CPU] 0.0467574 [GPU]
        Inlier RMSE: 0.0117937 [CPU] 0.0117971  [GPU]
        Transformation [CPU]
                [[0.999987 0.00504691 -0.000450257 -0.000840760],
                [-0.00504024 0.999893 0.0137413 0.00628841],
                [0.000519560 -0.0137389 0.999906 0.0332873],
                [0.0 0.0 0.0 1.0]]
        Transformation [GPU]
                [[0.999987 0.00505141 -0.000483726 -0.000778346],
                [-0.00504429 0.999893 0.0137406 0.00630834],
                [0.000553084 -0.013738 0.999905 0.0332003],
    */

    num_corres = reg_p2plane.correspondence_set_.GetShape()[0];
    isCorrespondencestrue = (num_corres > 9200);
    isFitnesstrue = (reg_p2plane.fitness_ > 0.04);
    isInlierRMSEtrue = (reg_p2plane.inlier_rmse_ < 0.02);

    EXPECT_TRUE(isCorrespondencestrue);
    EXPECT_TRUE(isFitnesstrue);
    EXPECT_TRUE(isInlierRMSEtrue);
}

}  // namespace tests
}  // namespace open3d
