// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/registration/Registration.h"

#include "core/CoreTest.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
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

static std::tuple<t::geometry::PointCloud,
                  t::geometry::PointCloud,
                  core::Tensor,
                  double>
GetRegistrationTestData(core::Dtype& dtype, core::Device& device) {
    t::geometry::PointCloud source_tpcd, target_tpcd;
    data::DemoICPPointClouds pcd_fragments;
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[0], source_tpcd);
    t::io::ReadPointCloud(pcd_fragments.GetPaths()[1], target_tpcd);
    source_tpcd = source_tpcd.To(device).VoxelDownSample(0.02);
    target_tpcd = target_tpcd.To(device).VoxelDownSample(0.02);

    // Convert color to float values.
    for (auto& kv : source_tpcd.GetPointAttr()) {
        if (kv.first == "colors" && kv.second.GetDtype() == core::UInt8) {
            source_tpcd.SetPointAttr(kv.first,
                                     kv.second.To(device, dtype).Div(255.0));
        } else {
            source_tpcd.SetPointAttr(kv.first, kv.second.To(device, dtype));
        }
    }
    for (auto& kv : target_tpcd.GetPointAttr()) {
        if (kv.first == "colors" && kv.second.GetDtype() == core::UInt8) {
            target_tpcd.SetPointAttr(kv.first,
                                     kv.second.To(device, dtype).Div(255.0));
        } else {
            target_tpcd.SetPointAttr(kv.first, kv.second.To(device, dtype));
        }
    }

    // Initial transformation input.
    const core::Tensor initial_transform_t =
            core::Tensor::Init<double>({{0.862, 0.011, -0.507, 0.5},
                                        {-0.139, 0.967, -0.215, 0.7},
                                        {0.487, 0.255, 0.835, -1.4},
                                        {0.0, 0.0, 0.0, 1.0}},
                                       core::Device("CPU:0"));

    const double max_correspondence_dist = 0.7;

    return std::make_tuple(source_tpcd, target_tpcd, initial_transform_t,
                           max_correspondence_dist);
}

TEST_P(RegistrationPermuteDevices, EvaluateRegistration) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // 1. Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t;
        // Search radius.
        double max_correspondence_dist;
        std::tie(source_tpcd, target_tpcd, initial_transform_t,
                 max_correspondence_dist) =
                GetRegistrationTestData(dtype, device);
        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for legacy implementation.
        const Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

        // Tensor evaluation.
        t_reg::RegistrationResult evaluation_t = t_reg::EvaluateRegistration(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t);

        // Legacy evaluation.
        l_reg::RegistrationResult evaluation_l = l_reg::EvaluateRegistration(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l);
        Eigen::Matrix4d::Identity();
        EXPECT_NEAR(evaluation_t.fitness_, evaluation_l.fitness_, 0.005);
        EXPECT_NEAR(evaluation_t.inlier_rmse_, evaluation_l.inlier_rmse_,
                    0.005);
    }
}

TEST_P(RegistrationPermuteDevices, ICPPointToPoint) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // 1. Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t;
        // Search radius.
        double max_correspondence_dist;
        std::tie(source_tpcd, target_tpcd, initial_transform_t,
                 max_correspondence_dist) =
                GetRegistrationTestData(dtype, device);
        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for legacy implementation.
        const Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

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

        EXPECT_NEAR(reg_p2p_t.fitness_, reg_p2p_l.fitness_, 0.005);
        EXPECT_NEAR(reg_p2p_t.inlier_rmse_, reg_p2p_l.inlier_rmse_, 0.005);
    }
}

TEST_P(RegistrationPermuteDevices, ICPPointToPlane) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // 1. Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t;
        // Search radius.
        double max_correspondence_dist;
        std::tie(source_tpcd, target_tpcd, initial_transform_t,
                 max_correspondence_dist) =
                GetRegistrationTestData(dtype, device);
        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for legacy implementation.
        const Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

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

        EXPECT_NEAR(reg_p2plane_t.fitness_, reg_p2plane_l.fitness_, 0.005);
        EXPECT_NEAR(reg_p2plane_t.inlier_rmse_, reg_p2plane_l.inlier_rmse_,
                    0.005);
    }
}

TEST_P(RegistrationPermuteDevices, ICPColored) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // 1. Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t;
        // Search radius.
        double max_correspondence_dist;
        std::tie(source_tpcd, target_tpcd, initial_transform_t,
                 max_correspondence_dist) =
                GetRegistrationTestData(dtype, device);
        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for legacy implementation.
        const Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

        double relative_fitness = 1e-6;
        double relative_rmse = 1e-6;
        int max_iterations = 2;

        // ColoredICP - Tensor.
        t_reg::RegistrationResult reg_colored_t = t_reg::ICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                initial_transform_t,
                t_reg::TransformationEstimationForColoredICP(),
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations),
                -1.0);

        // ColoredICP - Legacy.
        l_reg::RegistrationResult reg_colored_l = l_reg::RegistrationColoredICP(
                source_lpcd, target_lpcd, max_correspondence_dist,
                initial_transform_l,
                l_reg::TransformationEstimationForColoredICP(),
                l_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations));

        EXPECT_NEAR(reg_colored_t.fitness_, reg_colored_l.fitness_, 0.05);
        EXPECT_NEAR(reg_colored_t.inlier_rmse_, reg_colored_l.inlier_rmse_,
                    0.02);
    }
}

core::Tensor ComputeDirectionVectors(const core::Tensor& positions) {
    core::Tensor directions = core::Tensor::Empty(
            positions.GetShape(), positions.GetDtype(), positions.GetDevice());
    for (int64_t i = 0; i < positions.GetLength(); ++i) {
        // Compute the norm of the position vector.
        core::Tensor norm = (positions[i][0] * positions[i][0] +
                             positions[i][1] * positions[i][1] +
                             positions[i][2] * positions[i][2])
                                    .Sqrt();

        // If the norm is zero, set the direction vector to zero.
        if (norm.Item<float>() == 0.0) {
            directions[i].Fill(0.0);
        } else {
            // Otherwise, compute the direction vector by dividing the position
            // vector by its norm.
            directions[i] = positions[i] / norm;
        }
    }
    return directions;
}

static std::tuple<t::geometry::PointCloud,
                  t::geometry::PointCloud,
                  core::Tensor,
                  core::Tensor,
                  double,
                  double>
GetDopplerICPRegistrationTestData(core::Dtype& dtype, core::Device& device) {
    t::geometry::PointCloud source_tpcd, target_tpcd;
    data::DemoDopplerICPSequence demo_sequence;
    t::io::ReadPointCloud(demo_sequence.GetPath(0), source_tpcd);
    t::io::ReadPointCloud(demo_sequence.GetPath(1), target_tpcd);

    source_tpcd.SetPointAttr(
            "directions",
            ComputeDirectionVectors(source_tpcd.GetPointPositions()));

    source_tpcd = source_tpcd.To(device).UniformDownSample(5);
    target_tpcd = target_tpcd.To(device).UniformDownSample(5);

    Eigen::Matrix4d calibration{Eigen::Matrix4d::Identity()};
    double period{0.0};
    demo_sequence.GetCalibration(calibration, period);

    // Calibration transformation input.
    const core::Tensor calibration_t =
            core::eigen_converter::EigenMatrixToTensor(calibration)
                    .To(device, dtype);

    // Get the ground truth pose for the pair<0, 1> (on CPU:0).
    auto trajectory = demo_sequence.GetTrajectory();
    const core::Tensor pose_t =
            core::eigen_converter::EigenMatrixToTensor(trajectory[1].second);

    const double max_correspondence_dist = 0.3;
    const double normals_search_radius = 10.0;
    const int normals_max_neighbors = 30;

    target_tpcd.EstimateNormals(normals_search_radius, normals_max_neighbors);

    return std::make_tuple(source_tpcd, target_tpcd, calibration_t, pose_t,
                           period, max_correspondence_dist);
}

TEST_P(RegistrationPermuteDevices, ICPDoppler) {
    core::Device device = GetParam();

    for (auto dtype : {core::Float32, core::Float64}) {
        // Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Calibration transformation input.
        core::Tensor calibration_t;
        // Ground truth pose.
        core::Tensor pose_t;
        // Time period between each point cloud scan.
        double period{0.0};
        // Search radius.
        double max_correspondence_dist{0.0};
        std::tie(source_tpcd, target_tpcd, calibration_t, pose_t, period,
                 max_correspondence_dist) =
                GetDopplerICPRegistrationTestData(dtype, device);

        const double relative_fitness = 1e-6;
        const double relative_rmse = 1e-6;
        const int max_iterations = 20;

        t_reg::TransformationEstimationForDopplerICP estimation_dicp;
        estimation_dicp.period_ = period;
        estimation_dicp.transform_vehicle_to_sensor_ = calibration_t;

        // DopplerICP - Tensor.
        t_reg::RegistrationResult reg_doppler_t = t_reg::ICP(
                source_tpcd, target_tpcd, max_correspondence_dist,
                core::Tensor::Eye(4, dtype, device), estimation_dicp,
                t_reg::ICPConvergenceCriteria(relative_fitness, relative_rmse,
                                              max_iterations),
                -1.0);

        core::Tensor estimated_pose =
                t::pipelines::kernel::TransformationToPose(
                        reg_doppler_t.transformation_.Inverse());
        core::Tensor expected_pose =
                t::pipelines::kernel::TransformationToPose(pose_t);

        const double pose_diff =
                (expected_pose - estimated_pose).Abs().Sum({0}).Item<double>();

        EXPECT_NEAR(reg_doppler_t.fitness_ - 0.9, 0.0, 0.05);
        EXPECT_NEAR(pose_diff - 0.017, 0.0, 0.005);
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
        // 1. Get data and parameters.
        t::geometry::PointCloud source_tpcd, target_tpcd;
        // Initial transformation input for tensor implementation.
        core::Tensor initial_transform_t;
        // Search radius.
        double max_correspondence_dist;
        std::tie(source_tpcd, target_tpcd, initial_transform_t,
                 max_correspondence_dist) =
                GetRegistrationTestData(dtype, device);
        open3d::geometry::PointCloud source_lpcd = source_tpcd.ToLegacy();
        open3d::geometry::PointCloud target_lpcd = target_tpcd.ToLegacy();

        // Initial transformation input for legacy implementation.
        const Eigen::Matrix4d initial_transform_l =
                core::eigen_converter::TensorToEigenMatrixXd(
                        initial_transform_t);

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
