// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/NormalDistributionsTransform.h"

#include <array>
#include <cmath>
#include <limits>

#include "open3d/geometry/PointCloud.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

geometry::PointCloud MakeStructuredPointCloud() {
    geometry::PointCloud pcd;
    const std::array<Eigen::Vector3d, 5> centers = {
            Eigen::Vector3d(-1.2, -0.8, -0.4), Eigen::Vector3d(-0.2, 0.7, 0.3),
            Eigen::Vector3d(0.9, -0.1, 0.8), Eigen::Vector3d(1.5, 1.0, -0.2),
            Eigen::Vector3d(-1.5, 1.1, 0.9)};

    for (int cluster = 0; cluster < static_cast<int>(centers.size());
         ++cluster) {
        const double angle = 0.35 * static_cast<double>(cluster);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        rotation.block<2, 2>(0, 0) << c, -s, s, c;

        for (int i = 0; i < 160; ++i) {
            const double a = (static_cast<double>(i % 16) - 7.5) / 7.5;
            const double b = (static_cast<double>((i / 16) % 10) - 4.5) / 4.5;
            const double h = std::sin(0.37 * static_cast<double>(i) +
                                      static_cast<double>(cluster));
            const Eigen::Vector3d offset(0.20 * a + 0.04 * b,
                                         0.13 * b + 0.03 * h,
                                         0.08 * h + 0.025 * a * b);
            pcd.points_.push_back(centers[cluster] + rotation * offset);
        }
    }
    return pcd;
}

geometry::PointCloud MakeRoundIndexedPointCloud() {
    geometry::PointCloud pcd;
    pcd.points_ = {
            Eigen::Vector3d(0.60, 0.08, 0.12),
            Eigen::Vector3d(0.75, 0.32, 0.18),
            Eigen::Vector3d(0.90, 0.16, 0.42),
            Eigen::Vector3d(1.10, 0.38, 0.08),
            Eigen::Vector3d(1.25, 0.10, 0.34),
            Eigen::Vector3d(1.40, 0.28, 0.26),
    };
    return pcd;
}

Eigen::Matrix4d MakeSmallInitialTransformation() {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();
    transformation.block<3, 1>(0, 3) = Eigen::Vector3d(0.01, -0.01, 0.01);
    return transformation;
}

Eigen::Matrix4d MakeTransformation() {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ())
                    .toRotationMatrix();
    transformation.block<3, 1>(0, 3) = Eigen::Vector3d(0.24, -0.17, 0.11);
    return transformation;
}

}  // namespace

TEST(NormalDistributionsTransform, OptionRejectsInvalidValues) {
    using pipelines::registration::NormalDistributionsTransformOption;

    EXPECT_THROW(NormalDistributionsTransformOption(-1.0), std::runtime_error);
    EXPECT_THROW(NormalDistributionsTransformOption(1.0, 2),
                 std::runtime_error);
    EXPECT_THROW(NormalDistributionsTransformOption(1.0, 4, 0.01, 0.01, 0.0),
                 std::runtime_error);
    EXPECT_THROW(NormalDistributionsTransformOption(1.0, 4, 0.01, 0.01, 1e-6,
                                                    30, 0.0),
                 std::runtime_error);
    EXPECT_THROW(NormalDistributionsTransformOption(1.0, 4, 0.01, 0.01, 1e-6,
                                                    30, 9.0, 2),
                 std::runtime_error);
}

TEST(NormalDistributionsTransform, RegistrationRejectsMutatedInvalidOption) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    const geometry::PointCloud pcd = MakeStructuredPointCloud();
    NormalDistributionsTransformOption option;
    option.voxel_size_ = 0.0;

    EXPECT_THROW(RegistrationNDT(pcd, pcd, option), std::runtime_error);

    option = NormalDistributionsTransformOption();
    option.relative_objective_ = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(RegistrationNDT(pcd, pcd, option), std::runtime_error);

    option = NormalDistributionsTransformOption();
    option.outlier_threshold_ = std::numeric_limits<double>::infinity();
    EXPECT_THROW(RegistrationNDT(pcd, pcd, option), std::runtime_error);
}

TEST(NormalDistributionsTransform, VoxelIndexUsesRound) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    const geometry::PointCloud pcd = MakeRoundIndexedPointCloud();
    const Eigen::Matrix4d init = MakeSmallInitialTransformation();
    const auto result =
            RegistrationNDT(pcd, pcd,
                            NormalDistributionsTransformOption(
                                    1.0, 4, 1e-3, 1e-6, 1e-6, 1, 9.0, 0),
                            init);

    EXPECT_FALSE(result.transformation_.isApprox(init, 1e-12));
    EXPECT_LT((result.transformation_ - Eigen::Matrix4d::Identity()).norm(),
              (init - Eigen::Matrix4d::Identity()).norm());
}

TEST(NormalDistributionsTransform, InvalidTargetCoordinatesThrow) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    const geometry::PointCloud source = MakeStructuredPointCloud();
    const std::array<double, 3> invalid_coordinates = {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(), 1e30};
    for (const double coordinate : invalid_coordinates) {
        geometry::PointCloud target = source;
        target.points_.emplace_back(coordinate, 0.0, 0.0);
        EXPECT_THROW(RegistrationNDT(source, target,
                                     NormalDistributionsTransformOption()),
                     std::runtime_error)
                << "coordinate: " << coordinate;
    }
}

TEST(NormalDistributionsTransform, InvalidSourceCoordinatesThrow) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    const geometry::PointCloud target = MakeStructuredPointCloud();
    const std::array<double, 3> invalid_coordinates = {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(), 1e30};
    for (const double coordinate : invalid_coordinates) {
        geometry::PointCloud source = target;
        source.points_.emplace_back(coordinate, 0.0, 0.0);
        EXPECT_THROW(RegistrationNDT(source, target,
                                     NormalDistributionsTransformOption()),
                     std::runtime_error)
                << "coordinate: " << coordinate;
    }
}

TEST(NormalDistributionsTransform, VoxelIndexSupportsCoordinatesAboveInt32) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    geometry::PointCloud target;
    const std::array<Eigen::Vector3d, 6> offsets = {
            Eigen::Vector3d(0.125, 0.0, 0.0),
            Eigen::Vector3d(-0.125, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.125, 0.0),
            Eigen::Vector3d(0.0, -0.125, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.125),
            Eigen::Vector3d(0.0, 0.0, -0.125)};
    for (const double center_x : {3e9, -3e9}) {
        for (const Eigen::Vector3d &offset : offsets) {
            target.points_.push_back(Eigen::Vector3d(center_x, 0.0, 0.0) +
                                     offset);
        }
    }

    geometry::PointCloud source;
    source.points_.assign(6, Eigen::Vector3d(3e9, 0.0, 0.0));

    const auto result =
            RegistrationNDT(source, target,
                            NormalDistributionsTransformOption(
                                    1.0, 6, 1e-3, 1e-6, 1e-6, 1, 0.1, 0));

    EXPECT_EQ(result.correspondence_set_.size(), source.points_.size());
    EXPECT_DOUBLE_EQ(result.fitness_, 1.0);
}

TEST(NormalDistributionsTransform, ResultRMSEMatchesCorrespondenceSet) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    const geometry::PointCloud pcd = MakeRoundIndexedPointCloud();
    const Eigen::Matrix4d init = MakeSmallInitialTransformation();
    const auto result =
            RegistrationNDT(pcd, pcd,
                            NormalDistributionsTransformOption(
                                    1.0, 4, 1e-3, 1e-6, 1e-6, 1, 9.0, 0),
                            init);

    EXPECT_FALSE(result.transformation_.isApprox(init, 1e-12));
    EXPECT_LT((result.transformation_ - Eigen::Matrix4d::Identity()).norm(),
              (init - Eigen::Matrix4d::Identity()).norm());
    ASSERT_EQ(result.correspondence_set_.size(), pcd.points_.size());

    geometry::PointCloud source_transformed = pcd;
    source_transformed.Transform(result.transformation_);
    double error2 = 0.0;
    for (const Eigen::Vector2i &correspondence : result.correspondence_set_) {
        error2 += (source_transformed.points_[correspondence[0]] -
                   pcd.points_[correspondence[1]])
                          .squaredNorm();
    }
    const double expected_rmse =
            std::sqrt(error2 / result.correspondence_set_.size());
    EXPECT_NEAR(result.inlier_rmse_, expected_rmse, 1e-12);
}

TEST(NormalDistributionsTransform, RegistrationNDTRecoversKnownTransform) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    geometry::PointCloud source = MakeStructuredPointCloud();
    geometry::PointCloud target = source;
    const Eigen::Matrix4d expected = MakeTransformation();
    target.Transform(expected);

    const auto result =
            RegistrationNDT(source, target,
                            NormalDistributionsTransformOption(
                                    0.8, 4, 1e-3, 0.01, 1e-7, 40, 9.0, 1),
                            Eigen::Matrix4d::Identity());

    EXPECT_GT(result.fitness_, 0.80);
    EXPECT_LT(result.inlier_rmse_, 0.16);
    EXPECT_TRUE(result.transformation_.isApprox(expected, 5e-2))
            << "expected:\n"
            << expected << "\nactual:\n"
            << result.transformation_;
}

TEST(NormalDistributionsTransform, RegistrationNDTImprovesInitialAlignment) {
    using pipelines::registration::EvaluateRegistration;
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    geometry::PointCloud source = MakeStructuredPointCloud();
    geometry::PointCloud target = source;
    const Eigen::Matrix4d expected = MakeTransformation();
    target.Transform(expected);

    const auto initial = EvaluateRegistration(source, target, 0.35);
    const auto result =
            RegistrationNDT(source, target,
                            NormalDistributionsTransformOption(
                                    0.8, 4, 1e-3, 0.01, 1e-7, 40, 9.0, 1),
                            Eigen::Matrix4d::Identity());
    const auto refined =
            EvaluateRegistration(source, target, 0.35, result.transformation_);

    EXPECT_GT(refined.fitness_, initial.fitness_);
    EXPECT_LT(refined.inlier_rmse_, initial.inlier_rmse_);
}

TEST(NormalDistributionsTransform,
     RankDeficientHessianPreservesInitialTransformation) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    geometry::PointCloud source;
    source.points_.assign(10, Eigen::Vector3d::Zero());

    geometry::PointCloud target;
    target.points_ = {
            Eigen::Vector3d(0.35, 0.25, 0.25),
            Eigen::Vector3d(0.25, 0.35, 0.25),
            Eigen::Vector3d(0.25, 0.25, 0.35),
            Eigen::Vector3d(0.15, 0.15, 0.15),
    };

    Eigen::Matrix4d init = Eigen::Matrix4d::Identity();
    init.block<3, 1>(0, 3) = Eigen::Vector3d(0.2, 0.2, 0.2);
    const auto result =
            RegistrationNDT(source, target,
                            NormalDistributionsTransformOption(
                                    1.0, 4, 1e-3, 1e-6, 1e-6, 1, 9.0, 0),
                            init);

    EXPECT_TRUE(result.transformation_.isApprox(init, 1e-12))
            << "expected:\n"
            << init << "\nactual:\n"
            << result.transformation_;
}

TEST(NormalDistributionsTransform, RelativeObjectiveStopsBeforeSecondUpdate) {
    using pipelines::registration::NormalDistributionsTransformOption;
    using pipelines::registration::RegistrationNDT;

    geometry::PointCloud source = MakeStructuredPointCloud();
    geometry::PointCloud target = source;
    target.Transform(MakeTransformation());

    NormalDistributionsTransformOption one_iteration_option(0.8, 4, 1e-3, 1e-15,
                                                            1e-12, 1, 9.0, 1);
    NormalDistributionsTransformOption converged_option(0.8, 4, 1e-3, 1e-15,
                                                        1e9, 40, 9.0, 1);
    EXPECT_DOUBLE_EQ(converged_option.relative_objective_, 1e9);

    const auto one_iteration =
            RegistrationNDT(source, target, one_iteration_option);
    const auto converged = RegistrationNDT(source, target, converged_option);
    EXPECT_TRUE(converged.transformation_.isApprox(
            one_iteration.transformation_, 1e-12));
}

}  // namespace tests
}  // namespace open3d
