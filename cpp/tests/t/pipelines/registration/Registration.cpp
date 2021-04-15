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
#include "open3d/pipelines/registration/Registration.h"
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

TEST_P(RegistrationPermuteDevices, ICPConvergenceCriteriaConstructor) {
    // Constructor.
    t::pipelines::registration::ICPConvergenceCriteria convergence_criteria;
    // Default values.
    EXPECT_EQ(convergence_criteria.max_iteration_, 30);
    EXPECT_DOUBLE_EQ(convergence_criteria.relative_fitness_, 1e-6);
    EXPECT_DOUBLE_EQ(convergence_criteria.relative_rmse_, 1e-6);
}

TEST_P(RegistrationPermuteDevices, RegistrationResultConstructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float64;

    // Initial transformation input for tensor implementation.
    core::Tensor init_trans_t = core::Tensor::Eye(4, dtype, device);

    t::pipelines::registration::RegistrationResult reg_result(init_trans_t);

    EXPECT_DOUBLE_EQ(reg_result.inlier_rmse_, 0.0);
    EXPECT_DOUBLE_EQ(reg_result.fitness_, 0.0);
    EXPECT_TRUE(reg_result.transformation_.AllClose(init_trans_t));
}

TEST_P(RegistrationPermuteDevices, EvaluateRegistration) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);

    std::vector<float> target_normals_vec{
            -0.0085016, -0.22355,  -0.519574, 0.257463,   -0.0738755, -0.698319,
            0.0574301,  -0.484248, -0.409929, -0.0123503, -0.230172,  -0.52072,
            0.355904,   -0.142007, -0.720467, 0.0674038,  -0.418757,  -0.458602,
            0.226091,   0.258253,  -0.874024, 0.43979,    0.122441,   -0.574998,
            0.109144,   0.180992,  -0.762368, 0.273325,   0.292013,   -0.903111,
            0.385407,   -0.212348, -0.277818};
    core::Tensor target_normals(target_normals_vec, {11, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    open3d::geometry::PointCloud source_l_down =
            source_device.ToLegacyPointCloud();
    open3d::geometry::PointCloud target_l_down =
            target_device.ToLegacyPointCloud();

    // Initial transformation input for tensor implementation.
    core::Tensor init_trans_t =
            core::Tensor::Eye(4, core::Dtype::Float64, device);

    // Initial transformation input for legacy implementation.
    Eigen::Matrix4d init_trans_l = Eigen::Matrix4d::Identity();

    // Identity transformation.
    double max_correspondence_dist = 1.25;

    // Tensor evaluation.
    t::pipelines::registration::RegistrationResult evaluation_t =
            open3d::t::pipelines::registration::EvaluateRegistration(
                    source_device, target_device, max_correspondence_dist,
                    init_trans_t);

    // Legacy evaluation.
    open3d::pipelines::registration::RegistrationResult evaluation_l =
            open3d::pipelines::registration::EvaluateRegistration(
                    source_l_down, target_l_down, max_correspondence_dist,
                    init_trans_l);

    EXPECT_NEAR(evaluation_t.fitness_, evaluation_l.fitness_, 0.0005);
    EXPECT_NEAR(evaluation_t.inlier_rmse_, evaluation_l.inlier_rmse_, 0.0005);
}

TEST_P(RegistrationPermuteDevices, RegistrationICPPointToPoint) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);

    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);

    open3d::geometry::PointCloud source_l_down =
            source_device.ToLegacyPointCloud();
    open3d::geometry::PointCloud target_l_down =
            target_device.ToLegacyPointCloud();

    // Initial transformation input for tensor implementation.
    core::Tensor init_trans_t =
            core::Tensor::Eye(4, core::Dtype::Float64, device);

    // Initial transformation input for legacy implementation.
    Eigen::Matrix4d init_trans_l = Eigen::Matrix4d::Identity();

    double max_correspondence_dist = 1.25;
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 2;

    // PointToPoint - Tensor.
    t::pipelines::registration::RegistrationResult reg_p2p_t =
            open3d::t::pipelines::registration::RegistrationICP(
                    source_device, target_device, max_correspondence_dist,
                    init_trans_t,
                    open3d::t::pipelines::registration::
                            TransformationEstimationPointToPoint(),
                    open3d::t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));

    // PointToPoint - Legacy.
    pipelines::registration::RegistrationResult reg_p2p_l =
            open3d::pipelines::registration::RegistrationICP(
                    source_l_down, target_l_down, max_correspondence_dist,
                    init_trans_l,
                    open3d::pipelines::registration::
                            TransformationEstimationPointToPoint(),
                    open3d::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));

    EXPECT_NEAR(reg_p2p_t.fitness_, reg_p2p_l.fitness_, 0.0005);
    EXPECT_NEAR(reg_p2p_t.inlier_rmse_, reg_p2p_l.inlier_rmse_, 0.0005);
}

TEST_P(RegistrationPermuteDevices, RegistrationICPPointToPlane) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    std::vector<float> src_points_vec{
            1.15495,  2.40671, 1.15061,  1.81481,  2.06281, 1.71927, 0.888322,
            2.05068,  2.04879, 3.78842,  1.70788,  1.30246, 1.8437,  2.22894,
            0.986237, 2.95706, 2.2018,   0.987878, 1.72644, 1.24356, 1.93486,
            0.922024, 1.14872, 2.34317,  3.70293,  1.85134, 1.15357, 3.06505,
            1.30386,  1.55279, 0.634826, 1.04995,  2.47046, 1.40107, 1.37469,
            1.09687,  2.93002, 1.96242,  1.48532,  3.74384, 1.30258, 1.30244};
    core::Tensor source_points(src_points_vec, {14, 3}, dtype, device);
    t::geometry::PointCloud source_device(device);
    source_device.SetPoints(source_points);

    std::vector<float> target_points_vec{
            2.41766, 2.05397, 1.74994, 1.37848, 2.19793, 1.66553, 2.24325,
            2.27183, 1.33708, 3.09898, 1.98482, 1.77401, 1.81615, 1.48337,
            1.49697, 3.01758, 2.20312, 1.51502, 2.38836, 1.39096, 1.74914,
            1.30911, 1.4252,  1.37429, 3.16847, 1.39194, 1.90959, 1.59412,
            1.53304, 1.5804,  1.34342, 2.19027, 1.30075};
    core::Tensor target_points(target_points_vec, {11, 3}, dtype, device);

    std::vector<float> target_normals_vec{
            -0.0085016, -0.22355,  -0.519574, 0.257463,   -0.0738755, -0.698319,
            0.0574301,  -0.484248, -0.409929, -0.0123503, -0.230172,  -0.52072,
            0.355904,   -0.142007, -0.720467, 0.0674038,  -0.418757,  -0.458602,
            0.226091,   0.258253,  -0.874024, 0.43979,    0.122441,   -0.574998,
            0.109144,   0.180992,  -0.762368, 0.273325,   0.292013,   -0.903111,
            0.385407,   -0.212348, -0.277818};
    core::Tensor target_normals(target_normals_vec, {11, 3}, dtype, device);
    t::geometry::PointCloud target_device(device);
    target_device.SetPoints(target_points);
    target_device.SetPointNormals(target_normals);

    open3d::geometry::PointCloud source_l_down =
            source_device.ToLegacyPointCloud();
    open3d::geometry::PointCloud target_l_down =
            target_device.ToLegacyPointCloud();

    // Initial transformation input for tensor implementation.
    core::Tensor init_trans_t =
            core::Tensor::Eye(4, core::Dtype::Float64, device);

    // Initial transformation input for legacy implementation.
    Eigen::Matrix4d init_trans_l = Eigen::Matrix4d::Identity();

    double max_correspondence_dist = 2.0;
    double relative_fitness = 1e-6;
    double relative_rmse = 1e-6;
    int max_iterations = 2;

    // PointToPlane - Tensor.
    t::pipelines::registration::RegistrationResult reg_p2plane_t =
            open3d::t::pipelines::registration::RegistrationICP(
                    source_device, target_device, max_correspondence_dist,
                    init_trans_t,
                    open3d::t::pipelines::registration::
                            TransformationEstimationPointToPlane(),
                    open3d::t::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));

    // PointToPlane - Legacy.
    pipelines::registration::RegistrationResult reg_p2plane_l =
            open3d::pipelines::registration::RegistrationICP(
                    source_l_down, target_l_down, max_correspondence_dist,
                    init_trans_l,
                    open3d::pipelines::registration::
                            TransformationEstimationPointToPlane(),
                    open3d::pipelines::registration::ICPConvergenceCriteria(
                            relative_fitness, relative_rmse, max_iterations));

    EXPECT_NEAR(reg_p2plane_t.fitness_, reg_p2plane_l.fitness_, 0.0005);
    EXPECT_NEAR(reg_p2plane_t.inlier_rmse_, reg_p2plane_l.inlier_rmse_, 0.0005);
}

}  // namespace tests
}  // namespace open3d
