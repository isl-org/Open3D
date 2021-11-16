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
#include "open3d/pipelines/registration/ColoredICP.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/pipelines/registration/RobustKernel.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"
#include "open3d/t/pipelines/registration/RobustKernelImpl.h"
#include "tests/Tests.h"

namespace t_reg = open3d::t::pipelines::registration;
namespace l_reg = open3d::pipelines::registration;

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
    core::Dtype dtype = core::Float64;

    // Initial transformation input for tensor implementation.
    core::Tensor init_trans_t = core::Tensor::Eye(4, dtype, device);

    t_reg::RegistrationResult reg_result(init_trans_t);

    EXPECT_DOUBLE_EQ(reg_result.inlier_rmse_, 0.0);
    EXPECT_DOUBLE_EQ(reg_result.fitness_, 0.0);
    EXPECT_TRUE(reg_result.transformation_.AllClose(init_trans_t));
}

static std::tuple<t::geometry::PointCloud, t::geometry::PointCloud>
GetTestPointClouds(const core::Dtype& dtype, const core::Device& device) {
    t::geometry::PointCloud source(device);
    t::geometry::PointCloud target(device);

    core::Tensor source_points =
            core::Tensor::Init<float>(
                    {{1.0, 0.5, 2.0}, {0.5, 0.5, 2.0}, {0.5, 0.5, 2.5},
                     {3.0, 1.0, 1.5}, {3.5, 1.0, 1.0}, {0.5, 1.0, 2.0},
                     {1.5, 1.0, 1.5}, {2.0, 1.0, 1.5}, {1.0, 1.0, 2.0},
                     {2.5, 1.0, 1.5}, {3.0, 1.0, 1.0}, {0.5, 1.0, 2.5},
                     {1.0, 1.0, 1.0}, {1.5, 1.0, 1.0}, {1.0, 1.5, 1.0},
                     {3.0, 1.5, 1.0}, {3.5, 1.5, 1.0}, {3.0, 1.5, 1.5},
                     {0.5, 1.5, 1.5}, {0.5, 1.5, 2.0}, {1.0, 1.5, 2.0},
                     {2.5, 1.5, 1.5}, {1.5, 1.5, 1.0}, {1.5, 1.5, 1.5},
                     {2.0, 1.5, 1.5}, {3.0, 1.5, 0.5}, {2.5, 1.5, 1.0},
                     {2.0, 1.5, 1.0}, {3.0, 2.0, 0.5}, {0.5, 2.0, 1.5},
                     {3.0, 2.0, 1.0}, {1.0, 2.0, 1.0}, {2.0, 1.5, 0.5},
                     {0.5, 2.0, 2.0}, {2.5, 2.0, 1.0}, {2.5, 2.0, 0.5},
                     {2.0, 2.0, 0.5}, {2.5, 1.5, 0.5}, {3.0, 2.0, 1.5},
                     {2.0, 2.0, 1.0}, {1.0, 2.0, 2.0}, {1.5, 2.0, 1.0},
                     {1.5, 2.0, 1.5}, {2.5, 2.0, 1.5}, {2.0, 2.0, 1.5},
                     {1.0, 2.0, 0.5}, {0.5, 2.0, 1.0}, {1.5, 2.0, 0.5},
                     {1.0, 2.0, 1.5}},
                    device)
                    .To(dtype);

    core::Tensor target_points =
            core::Tensor::Init<float>(
                    {{1.5, 1.0, 1.5}, {2.5, 1.0, 1.5}, {1.5, 1.0, 1.0},
                     {1.0, 1.0, 1.0}, {2.0, 1.0, 1.5}, {3.0, 1.0, 1.5},
                     {1.0, 1.0, 0.5}, {1.0, 1.5, 1.0}, {1.0, 1.5, 0.5},
                     {1.0, 1.0, 1.5}, {3.0, 1.0, 2.0}, {3.0, 1.5, 2.0},
                     {3.0, 1.5, 1.5}, {1.0, 1.5, 1.5}, {1.5, 1.5, 1.5},
                     {2.5, 1.5, 1.5}, {2.0, 1.5, 1.5}, {1.5, 1.5, 1.0},
                     {2.5, 1.5, 2.0}, {1.0, 2.0, 1.0}, {1.0, 2.0, 0.5},
                     {2.5, 1.5, 1.0}, {3.0, 2.0, 1.5}, {2.5, 2.0, 1.0},
                     {2.5, 2.0, 1.5}, {1.5, 2.0, 1.0}, {2.0, 1.5, 1.0},
                     {1.0, 2.0, 1.5}, {2.0, 2.0, 1.0}, {1.5, 2.0, 1.5},
                     {1.5, 2.0, 0.5}, {2.0, 2.0, 1.5}, {2.0, 2.0, 0.5},
                     {1.5, 2.5, 1.0}, {1.0, 2.5, 1.0}, {3.0, 2.0, 1.0},
                     {2.0, 2.5, 1.0}, {2.5, 2.5, 1.0}},
                    device)
                    .To(dtype);

    core::Tensor target_normals =
            core::Tensor::Init<float>({{0.15597, -0.0463812, -0.986672},
                                       {-0.213545, 0.887963, 0.407334},
                                       {0.423193, -0.121977, -0.897792},
                                       {0.202251, 0.27611, -0.939605},
                                       {0.275452, 0.207216, -0.938716},
                                       {0.326146, 0.0385317, -0.944534},
                                       {0.983129, -0.174668, -0.0543011},
                                       {0.898665, -0.0602029, 0.434485},
                                       {0.711325, 0.193223, -0.675783},
                                       {0.346158, 0.198724, -0.916888},
                                       {0.302085, 0.28938, -0.908297},
                                       {0.341044, 0.414138, -0.843907},
                                       {0.212191, 0.213068, -0.953717},
                                       {0.239759, 0.313187, -0.918929},
                                       {0.302290, 0.27265, -0.913391},
                                       {0.209796, 0.402747, -0.890944},
                                       {0.267025, 0.218226, -0.938656},
                                       {0.00126928, -0.976587, -0.21512},
                                       {0.321912, 0.194736, -0.926526},
                                       {0.831227, 0.236675, -0.503037},
                                       {0.987006, -0.155324, 0.0411639},
                                       {0.103384, -0.808796, -0.57893},
                                       {0.181245, 0.66226, -0.727023},
                                       {0.235471, 0.525053, -0.817846},
                                       {0.231954, 0.446165, -0.864369},
                                       {-0.261931, -0.725542, -0.636381},
                                       {0.120953, -0.864985, -0.487003},
                                       {0.858345, -0.227847, 0.459706},
                                       {-0.416259, -0.367408, -0.831709},
                                       {-0.476652, 0.206048, -0.854604},
                                       {-0.211959, -0.523378, -0.825317},
                                       {-0.964914, 0.0541031, -0.256931},
                                       {-0.0653566, -0.913961, -0.400504},
                                       {-0.846868, -0.170805, -0.503628},
                                       {0.0366971, 0.515834, -0.855902},
                                       {-0.0714554, -0.855019, -0.513651},
                                       {-0.0217377, -0.957744, -0.286799},
                                       {-0.0345231, -0.947096, -0.319088}},
                                      device)
                    .To(dtype);

    source.SetPointPositions(source_points.To(dtype, false));

    target.SetPointPositions(target_points.To(dtype, false));
    target.SetPointNormals(target_normals.To(dtype, false));

    return std::make_tuple(source, target);
}

TEST_P(RegistrationPermuteDevices, EvaluateRegistration) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l = Eigen::Matrix4d::Identity();

        // Identity transformation.
        double max_correspondence_dist = 3.0;

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

TEST_P(RegistrationPermuteDevices, ICPPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

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

        double max_correspondence_dist = 3.0;
        double relative_fitness = 1e-6;
        double relative_rmse = 1e-6;
        int max_iterations = 2;

        // PointToPoint - Tensor.
        t_reg::RegistrationResult reg_p2p_t = t_reg::ICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationPointToPoint(),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations),
                -1.0);

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

TEST_P(RegistrationPermuteDevices, ICPPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

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
        t_reg::RegistrationResult reg_p2plane_t = t_reg::ICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationPointToPlane(
                        t_reg::RobustKernel(t_reg::RobustKernelMethod::L1Loss,
                                            /*scale parameter =*/1.0,
                                            /*shape parameter =*/1.0)),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations),
                -1.0);

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

TEST_P(RegistrationPermuteDevices, RegistrationColoredICP) {
    core::Device device = GetParam();

    t::geometry::PointCloud source_tpcd, target_tpcd;
    t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) + "/ColoredICP/frag_115.ply",
            source_tpcd);
    t::io::ReadPointCloud(
            std::string(TEST_DATA_DIR) + "/ColoredICP/frag_116.ply",
            target_tpcd);
    source_tpcd = source_tpcd.To(device);
    target_tpcd = target_tpcd.To(device);

    for (auto dtype : {core::Float32, core::Float64}) {
        for (auto& kv : source_tpcd.GetPointAttr()) {
            if (kv.first == "colors") {
                source_tpcd.SetPointAttr(
                        kv.first, kv.second.To(device, dtype).Div(255.0));
            } else {
                source_tpcd.SetPointAttr(kv.first, kv.second.To(device, dtype));
            }
        }
        for (auto& kv : target_tpcd.GetPointAttr()) {
            if (kv.first == "colors") {
                target_tpcd.SetPointAttr(
                        kv.first, kv.second.To(device, dtype).Div(255.0));
            } else {
                target_tpcd.SetPointAttr(kv.first, kv.second.To(device, dtype));
            }
        }

        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t = core::Tensor::Eye(4, dtype, device);

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

        double max_correspondence_dist = 0.01;
        double relative_fitness = 1e-6;
        double relative_rmse = 1e-6;
        int max_iterations = 2;

        // ColoredICP - Tensor.
        t_reg::RegistrationResult reg_p2plane_t = t_reg::ICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationForColoredICP(),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations),
                -1.0);

        // ColoredICP - Legacy.
        l_reg::RegistrationResult reg_p2plane_l = l_reg::RegistrationColoredICP(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l,
                l_reg::TransformationEstimationForColoredICP(),
                l_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        EXPECT_NEAR(reg_p2plane_t.fitness_, reg_p2plane_l.fitness_, 0.02);
        EXPECT_NEAR(reg_p2plane_t.inlier_rmse_, reg_p2plane_l.inlier_rmse_,
                    0.02);
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

    for (auto dtype : {core::Float32, core::Float64}) {
        for (auto loss_method : {t_reg::RobustKernelMethod::L2Loss,
                                 t_reg::RobustKernelMethod::L1Loss,
                                 t_reg::RobustKernelMethod::HuberLoss,
                                 t_reg::RobustKernelMethod::CauchyLoss,
                                 t_reg::RobustKernelMethod::GMLoss,
                                 t_reg::RobustKernelMethod::TukeyLoss,
                                 t_reg::RobustKernelMethod::GeneralizedLoss}) {
            DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
                DISPATCH_ROBUST_KERNEL_FUNCTION(
                        loss_method, scalar_t, scaling_parameter,
                        shape_parameter, [&]() {
                            auto weight = GetWeightFromRobustKernel(0.98);
                            EXPECT_NEAR(weight,
                                        expected_output[(int)loss_method],
                                        1e-3);
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

TEST_P(RegistrationPermuteDevices, GetInformationMatrixFromPointCloud) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        t::geometry::PointCloud source_tpcd(device), target_tpcd(device);
        std::tie(source_tpcd, target_tpcd) = GetTestPointClouds(dtype, device);

        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t =
                core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

        // Initial transformation input for legacy implementation.
        Eigen::Matrix4d initial_transform_l = Eigen::Matrix4d::Identity();

        // Identity transformation.
        double max_correspondence_dist = 3.0;

        // Tensor information matrix.
        core::Tensor information_matrix_t = t_reg::GetInformationMatrix(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t);

        // Legacy evaluation.
        Eigen::Matrix6d information_matrix_l =
                l_reg::GetInformationMatrixFromPointClouds(
                        source_lpcd, target_lpcd, max_correspondence_dist,
                        initial_transform_l);

        core::Tensor information_matrix_from_legacy =
                core::eigen_converter::EigenMatrixToTensor(
                        information_matrix_l);

        EXPECT_TRUE(information_matrix_t.AllClose(
                information_matrix_from_legacy, 1e-1, 1e-1));
    }
}

}  // namespace tests
}  // namespace open3d
