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

#include "open3d/t/pipelines/registration/Registration.h"

#include "core/CoreTest.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"
#include "tests/UnitTest.h"

namespace t_reg = open3d::t::pipelines::registration;
namespace l_reg = open3d::pipelines::registration;
static const std::string source_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_0.pcd";
static const std::string target_pointcloud_filename =
        std::string(TEST_DATA_DIR) + "/ICP/cloud_bin_1.pcd";

namespace open3d {
namespace tests {

class RegistrationPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Registration,
                         RegistrationPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(RegistrationPermuteDevices, ICPConvergenceCriteriaConstructor) {
    // Constructor.
    t_reg::ICPConvergenceCriteria convergence_criteria;
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

    t_reg::RegistrationResult reg_result(init_trans_t);

    EXPECT_DOUBLE_EQ(reg_result.inlier_rmse_, 0.0);
    EXPECT_DOUBLE_EQ(reg_result.fitness_, 0.0);
    EXPECT_TRUE(reg_result.transformation_.AllClose(init_trans_t));
}

static std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
GetTestPointClouds(const core::Dtype& dtype, const core::Device& device) {
    t::geometry::PointCloud source, target;

    t::io::ReadPointCloud(source_pointcloud_filename, source,
                          {"auto", false, false, true});
    t::io::ReadPointCloud(target_pointcloud_filename, target,
                          {"auto", false, false, true});

    source = source.To(device, false).VoxelDownSample(0.08);
    source.SetPoints(source.GetPoints().To(dtype, false));
    source.SetPointNormals(source.GetPointNormals().To(dtype, false));

    target = target.To(device, false).VoxelDownSample(0.08);
    target.SetPoints(target.GetPoints().To(dtype, false));
    target.SetPointNormals(target.GetPointNormals().To(dtype, false));

    return std::make_tuple(source, target);
}

TEST_P(RegistrationPermuteDevices, EvaluateRegistration) {
    core::Device device = GetParam();

    for (auto dtype : {core::Dtype::Float32, core::Dtype::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd =
                source_tpcd.ToLegacyPointCloud();
        open3d::geometry::PointCloud target_lpcd =
                target_tpcd.ToLegacyPointCloud();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t = core::Tensor::Eye(
                4, core::Dtype::Float64, core::Device("CPU:0"));

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l = Eigen::Matrix4d::Identity();

        // Identity transformation.
        double max_correspondence_dist = 1.5;

        // Tensor evaluation.
        t_reg::RegistrationResult evaluation_t = t_reg::EvaluateRegistration(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t);

        // Legacy evaluation.
        l_reg::RegistrationResult evaluation_l = l_reg::EvaluateRegistration(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l);
        Eigen::Matrix4d::Identity();
        EXPECT_NEAR(evaluation_t.fitness_, evaluation_l.fitness_, 0.0005);
        EXPECT_NEAR(evaluation_t.inlier_rmse_, evaluation_l.inlier_rmse_,
                    0.0005);
    }
}

TEST_P(RegistrationPermuteDevices, RegistrationICPPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Dtype::Float32, core::Dtype::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd =
                source_tpcd.ToLegacyPointCloud();
        open3d::geometry::PointCloud target_lpcd =
                target_tpcd.ToLegacyPointCloud();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t =
                core::Tensor::Init<double>({{0.862, 0.011, -0.507, 0.5},
                                            {-0.139, 0.967, -0.215, 0.7},
                                            {0.487, 0.255, 0.835, -1.4},
                                            {0.0, 0.0, 0.0, 1.0}},
                                           core::Device("CPU:0"));

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

        double max_correspondence_dist = 1.5;
        double relative_fitness = 1e-6;
        double relative_rmse = 1e-6;
        int max_iterations = 2;

        // PointToPoint - Tensor.
        t_reg::RegistrationResult reg_p2p_t = t_reg::RegistrationICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationPointToPoint(),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        // PointToPoint - Legacy.
        l_reg::RegistrationResult reg_p2p_l = l_reg::RegistrationICP(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l,
                l_reg::TransformationEstimationPointToPoint(),
                l_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        EXPECT_NEAR(reg_p2p_t.fitness_, reg_p2p_l.fitness_, 0.0005);
        EXPECT_NEAR(reg_p2p_t.inlier_rmse_, reg_p2p_l.inlier_rmse_, 0.0005);
    }
}

TEST_P(RegistrationPermuteDevices, RegistrationICPPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Dtype::Float32, core::Dtype::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd =
                source_tpcd.ToLegacyPointCloud();
        open3d::geometry::PointCloud target_lpcd =
                target_tpcd.ToLegacyPointCloud();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t =
                core::Tensor::Init<double>({{0.862, 0.011, -0.507, 0.5},
                                            {-0.139, 0.967, -0.215, 0.7},
                                            {0.487, 0.255, 0.835, -1.4},
                                            {0.0, 0.0, 0.0, 1.0}},
                                           core::Device("CPU:0"));

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

        double max_correspondence_dist = 1.5;
        double relative_fitness = 1e-6;
        double relative_rmse = 1e-6;
        int max_iterations = 2;

        // L1Loss Method:

        // PointToPlane - Tensor.
        t_reg::RegistrationResult reg_p2plane_t = t_reg::RegistrationICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationPointToPlane(
                        t_reg::RobustKernel(t_reg::RobustKernelMethod::L1Loss,
                                            /*scale parameter =*/1.0,
                                            /*shape parameter =*/1.0)),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        // PointToPlane - Legacy.
        l_reg::RegistrationResult reg_p2plane_l = l_reg::RegistrationICP(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l,
                l_reg::TransformationEstimationPointToPlane(
                        std::make_shared<l_reg::L1Loss>()),
                l_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        EXPECT_NEAR(reg_p2plane_t.fitness_, reg_p2plane_l.fitness_, 0.0005);
        EXPECT_NEAR(reg_p2plane_t.inlier_rmse_, reg_p2plane_l.inlier_rmse_,
                    0.0005);
    }
}

TEST_P(RegistrationPermuteDevices, RobustKernel) {
    double scaling_parameter = 1.0;
    double shape_parameter = 1.0;

    std::unordered_map<int, double> expected_output = {
            {0, 1.0},         // L2Loss [1.0]
            {1, 1.0204},      // L1Loss [1.0 / abs(residual)]
            {2, 1.0},         // HuberLoss [scale / max(abs(residual), scale)]
            {3, 0.5101},      // CauchyLoss [1 / (1 + sq(residual / scale))]
            {4, 0.260202},    // GMLoss [scale / sq(scale + sq(residual))]
            {5, 0.00156816},  // TukeyLoss [sq(1 - sq(min(1, abs(r) / scale)))]
            {6, 0.714213}     // GeneralizedLoss
    };

    for (auto dtype : {core::Dtype::Float32, core::Dtype::Float64}) {
        for (auto LossMethod : {t_reg::RobustKernelMethod::L2Loss,
                                t_reg::RobustKernelMethod::L1Loss,
                                t_reg::RobustKernelMethod::HuberLoss,
                                t_reg::RobustKernelMethod::CauchyLoss,
                                t_reg::RobustKernelMethod::GMLoss,
                                t_reg::RobustKernelMethod::TukeyLoss,
                                t_reg::RobustKernelMethod::GeneralizedLoss}) {
            DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
                DISPATCH_ROBUST_KERNEL_FUNCTION(
                        LossMethod, scalar_t, scaling_parameter,
                        shape_parameter, [&]() {
                            auto weight = GetWeightFromRobustKernel(0.98);
                            EXPECT_NEAR(weight,
                                        expected_output[(int)LossMethod], 1e-3);
                        });
            });
        }

        // GeneralizedLoss can behave as other loss methods by changing the
        // shape_parameter (and adjusting the scaling_parameter).
        // For shape_parameter = 2 : L2Loss.
        // For shape_parameter = 0 : Cauchy or Lorentzian Loss.
        // For shape_parameter = -2 : German-McClure or GM Loss.
        // For shape_parameter = 1 : Charbonnier Loss or Pseudo-Huber loss or
        // smoothened form of L1 Loss.
        //
        // Refer:
        // @article{BarronCVPR2019,
        //   Author = {Jonathan T. Barron},
        //   Title = {A General and Adaptive Robust Loss Function},
        //   Journal = {CVPR},
        //   Year = {2019}
        // }
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            DISPATCH_ROBUST_KERNEL_FUNCTION(
                    t_reg::RobustKernelMethod::GeneralizedLoss, scalar_t,
                    scaling_parameter, 2.0, [&]() {
                        auto weight = GetWeightFromRobustKernel(0.98);
                        EXPECT_NEAR(weight, 1.0, 1e-3);
                    });

            DISPATCH_ROBUST_KERNEL_FUNCTION(
                    t_reg::RobustKernelMethod::GeneralizedLoss, scalar_t,
                    scaling_parameter, 0.0, [&]() {
                        auto weight = GetWeightFromRobustKernel(0.98);
                        EXPECT_NEAR(weight, 0.675584, 1e-3);
                    });

            DISPATCH_ROBUST_KERNEL_FUNCTION(
                    t_reg::RobustKernelMethod::GeneralizedLoss, scalar_t,
                    scaling_parameter, -2.0, [&]() {
                        auto weight = GetWeightFromRobustKernel(0.98);
                        EXPECT_NEAR(weight, 0.650259, 1e-3);
                    });

            DISPATCH_ROBUST_KERNEL_FUNCTION(
                    t_reg::RobustKernelMethod::GeneralizedLoss, scalar_t,
                    scaling_parameter, 1.0, [&]() {
                        auto weight = GetWeightFromRobustKernel(0.98);
                        EXPECT_NEAR(weight, 0.714213, 1e-3);
                    });
        });
    }
}

}  // namespace tests
}  // namespace open3d
