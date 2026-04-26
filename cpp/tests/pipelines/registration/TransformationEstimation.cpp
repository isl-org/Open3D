// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/TransformationEstimation.h"

#include <array>
#include <cmath>
#include <string>

#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/utility/Eigen.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {
namespace {

geometry::PointCloud MakeBoxTarget() {
    geometry::PointCloud target;
    auto add = [&](const Eigen::Vector3d &point,
                   const Eigen::Vector3d &normal) {
        target.points_.push_back(point);
        target.normals_.push_back(normal);
    };

    const std::array<double, 2> kCoord = {0.0, 1.0};
    for (double y : kCoord) {
        for (double z : kCoord) {
            add(Eigen::Vector3d(0.0, y, z), Eigen::Vector3d(-1.0, 0.0, 0.0));
            add(Eigen::Vector3d(1.0, y, z), Eigen::Vector3d(1.0, 0.0, 0.0));
        }
    }
    for (double x : kCoord) {
        for (double z : kCoord) {
            add(Eigen::Vector3d(x, 0.0, z), Eigen::Vector3d(0.0, -1.0, 0.0));
            add(Eigen::Vector3d(x, 1.0, z), Eigen::Vector3d(0.0, 1.0, 0.0));
        }
    }
    for (double x : kCoord) {
        for (double y : kCoord) {
            add(Eigen::Vector3d(x, y, 0.0), Eigen::Vector3d(0.0, 0.0, -1.0));
            add(Eigen::Vector3d(x, y, 1.0), Eigen::Vector3d(0.0, 0.0, 1.0));
        }
    }
    return target;
}

geometry::PointCloud MakeDegeneratePlaneTarget() {
    geometry::PointCloud target;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            target.points_.push_back(Eigen::Vector3d(x, y, 0.0));
            target.normals_.push_back(Eigen::Vector3d(0.0, 0.0, 1.0));
        }
    }
    return target;
}

geometry::PointCloud MakeOffsetPlaneTarget() {
    geometry::PointCloud target;
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            target.points_.push_back(
                    Eigen::Vector3d(0.2 * x + 0.5, 0.2 * y + 0.5, 1.0));
        }
    }
    return target;
}

geometry::PointCloud MakeCylinderSideTarget() {
    geometry::PointCloud target;
    constexpr int kThetaCount = 24;
    constexpr int kZCount = 7;
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kZMin = -0.6;
    constexpr double kZMax = 0.6;

    for (int theta_index = 0; theta_index < kThetaCount; ++theta_index) {
        const double theta = 2.0 * kPi * theta_index / kThetaCount;
        const Eigen::Vector3d normal(std::cos(theta), std::sin(theta), 0.0);
        for (int z_index = 0; z_index < kZCount; ++z_index) {
            const double alpha = static_cast<double>(z_index) / (kZCount - 1);
            const double z = kZMin + alpha * (kZMax - kZMin);
            target.points_.push_back(Eigen::Vector3d(normal(0), normal(1), z));
            target.normals_.push_back(normal);
        }
    }
    return target;
}

geometry::PointCloud MakeAsymmetricBoxTarget() {
    geometry::PointCloud target;
    auto add = [&](const Eigen::Vector3d &point,
                   const Eigen::Vector3d &normal) {
        target.points_.push_back(point);
        target.normals_.push_back(normal);
    };

    const std::array<double, 5> kX = {0.0, 0.23, 0.51, 0.88, 1.30};
    const std::array<double, 5> kY = {0.0, 0.17, 0.39, 0.64, 0.80};
    const std::array<double, 5> kZ = {0.0, 0.19, 0.46, 0.73, 1.10};
    for (double y : kY) {
        for (double z : kZ) {
            add(Eigen::Vector3d(0.0, y, z), Eigen::Vector3d(-1.0, 0.0, 0.0));
            add(Eigen::Vector3d(1.3, y, z), Eigen::Vector3d(1.0, 0.0, 0.0));
        }
    }
    for (double x : kX) {
        for (double z : kZ) {
            add(Eigen::Vector3d(x, 0.0, z), Eigen::Vector3d(0.0, -1.0, 0.0));
            add(Eigen::Vector3d(x, 0.8, z), Eigen::Vector3d(0.0, 1.0, 0.0));
        }
    }
    for (double x : kX) {
        for (double y : kY) {
            add(Eigen::Vector3d(x, y, 0.0), Eigen::Vector3d(0.0, 0.0, -1.0));
            add(Eigen::Vector3d(x, y, 1.1), Eigen::Vector3d(0.0, 0.0, 1.0));
        }
    }
    return target;
}

pipelines::registration::CorrespondenceSet MakeIdentityCorrespondences(
        size_t size) {
    pipelines::registration::CorrespondenceSet corres;
    corres.reserve(size);
    for (int i = 0; i < static_cast<int>(size); ++i) {
        corres.emplace_back(i, i);
    }
    return corres;
}

Eigen::Matrix4d MakeSmallTransform() {
    Eigen::Vector6d update;
    update << 0.01, -0.02, 0.015, 0.03, -0.02, 0.04;
    return utility::TransformVector6dToMatrix4d(update);
}

Eigen::Matrix4d MakeSmallRegistrationTransform() {
    Eigen::Vector6d update;
    update << 0.02, -0.015, 0.01, 0.035, -0.025, 0.03;
    return utility::TransformVector6dToMatrix4d(update);
}

}  // namespace

TEST(TransformationEstimation, DCRegOptionConstructor) {
    pipelines::registration::DCRegOption option;
    EXPECT_DOUBLE_EQ(option.degeneracy_condition_threshold_, 10.0);
    EXPECT_DOUBLE_EQ(option.kappa_target_, 10.0);
    EXPECT_DOUBLE_EQ(option.pcg_tolerance_, 1e-6);
    EXPECT_EQ(option.pcg_max_iteration_, 10);
    EXPECT_EQ(option.local_plane_knn_, 5);
    EXPECT_DOUBLE_EQ(option.local_plane_max_thickness_, 0.2);
    EXPECT_DOUBLE_EQ(option.local_plane_weight_slope_, 0.9);
    EXPECT_DOUBLE_EQ(option.local_plane_min_weight_, 0.1);
    EXPECT_TRUE(option.local_plane_use_weight_derivative_);
    EXPECT_DOUBLE_EQ(option.local_frame_convergence_rotation_, 1e-5);
    EXPECT_DOUBLE_EQ(option.local_frame_convergence_translation_, 1e-3);

    pipelines::registration::DCRegOption custom(20.0, 15.0, 1e-8, 6, 7, 0.15,
                                                0.7, 0.05, false, 1e-4, 2e-3);
    EXPECT_DOUBLE_EQ(custom.degeneracy_condition_threshold_, 20.0);
    EXPECT_DOUBLE_EQ(custom.kappa_target_, 15.0);
    EXPECT_DOUBLE_EQ(custom.pcg_tolerance_, 1e-8);
    EXPECT_EQ(custom.pcg_max_iteration_, 6);
    EXPECT_EQ(custom.local_plane_knn_, 7);
    EXPECT_DOUBLE_EQ(custom.local_plane_max_thickness_, 0.15);
    EXPECT_DOUBLE_EQ(custom.local_plane_weight_slope_, 0.7);
    EXPECT_DOUBLE_EQ(custom.local_plane_min_weight_, 0.05);
    EXPECT_FALSE(custom.local_plane_use_weight_derivative_);
    EXPECT_DOUBLE_EQ(custom.local_frame_convergence_rotation_, 1e-4);
    EXPECT_DOUBLE_EQ(custom.local_frame_convergence_translation_, 2e-3);
}

TEST(TransformationEstimation, PointToPlaneDCRegEmptyCorrespondence) {
    const geometry::PointCloud target = MakeBoxTarget();
    const geometry::PointCloud source = target;
    pipelines::registration::TransformationEstimationPointToPlaneDCReg
            estimation;

    const Eigen::Matrix4d update =
            estimation.ComputeTransformation(source, target, {});
    EXPECT_TRUE(update.isIdentity(0.0));
}

TEST(TransformationEstimation, PointToPlaneDCRegRequiresTargetNormals) {
    const geometry::PointCloud source = MakeBoxTarget();
    geometry::PointCloud target = source;
    target.normals_.clear();
    pipelines::registration::TransformationEstimationPointToPlaneDCReg
            estimation;

    EXPECT_ANY_THROW(estimation.InitializePointCloudsForTransformation(
            source, target, 1.0));
    const Eigen::Matrix4d update = estimation.ComputeTransformation(
            source, target, MakeIdentityCorrespondences(source.points_.size()));
    EXPECT_TRUE(update.isIdentity(0.0));
}

TEST(TransformationEstimation, PointToPlaneDCRegMatchesPointToPlane) {
    const geometry::PointCloud target = MakeBoxTarget();
    geometry::PointCloud source = target;
    source.Transform(MakeSmallTransform().inverse());
    const auto corres = MakeIdentityCorrespondences(source.points_.size());

    pipelines::registration::TransformationEstimationPointToPlane baseline;
    pipelines::registration::TransformationEstimationPointToPlaneDCReg dcreg;
    const Eigen::Matrix4d baseline_update =
            baseline.ComputeTransformation(source, target, corres);
    const Eigen::Matrix4d dcreg_update =
            dcreg.ComputeTransformation(source, target, corres);

    EXPECT_TRUE(baseline_update.allFinite());
    EXPECT_TRUE(dcreg_update.allFinite());
    EXPECT_LT((baseline_update - dcreg_update).norm(), 1e-8);
}

TEST(TransformationEstimation, PointToPlaneDCRegDegeneratePlaneIsFinite) {
    const geometry::PointCloud target = MakeDegeneratePlaneTarget();
    geometry::PointCloud source = target;
    source.Translate(Eigen::Vector3d(0.5, -0.25, -0.1));
    const auto corres = MakeIdentityCorrespondences(source.points_.size());

    pipelines::registration::TransformationEstimationPointToPlaneDCReg dcreg;
    const Eigen::Matrix4d update =
            dcreg.ComputeTransformation(source, target, corres);

    EXPECT_TRUE(update.allFinite());
    const double translation_norm = update.block<3, 1>(0, 3).norm();
    EXPECT_LT(translation_norm, 1.0);
    EXPECT_NEAR(update(2, 3), 0.1, 1e-8);
}

TEST(TransformationEstimation, DCRegDegeneracyAnalysisWellConditioned) {
    const geometry::PointCloud target = MakeAsymmetricBoxTarget();
    geometry::PointCloud source = target;
    source.Transform(MakeSmallRegistrationTransform().inverse());
    const auto corres = MakeIdentityCorrespondences(source.points_.size());

    const pipelines::registration::DCRegDegeneracyAnalysis analysis =
            pipelines::registration::ComputeDCRegDegeneracyAnalysis(
                    source, target, corres);

    EXPECT_TRUE(analysis.has_correspondence_);
    EXPECT_TRUE(analysis.has_target_normals_);
    EXPECT_FALSE(analysis.is_rank_deficient_);
    EXPECT_TRUE(analysis.schur_factorization_ok_);
    EXPECT_TRUE(std::isfinite(analysis.condition_number_full_));
    EXPECT_TRUE(std::isfinite(analysis.condition_number_rotation_));
    EXPECT_TRUE(std::isfinite(analysis.condition_number_translation_));
    EXPECT_TRUE(analysis.schur_eigenvalues_rotation_.allFinite());
    EXPECT_TRUE(analysis.schur_eigenvalues_translation_.allFinite());
    EXPECT_NE(analysis.degeneracy_description_.find("target/world"),
              std::string::npos);
    EXPECT_NE(analysis.coordinate_frame_.find("left-multiplied SE(3)"),
              std::string::npos);
    EXPECT_NE(analysis.solver_type_, "invalid");
}

TEST(TransformationEstimation, PointToPlaneDCRegCylinderDegeneracyIsStable) {
    const geometry::PointCloud target = MakeCylinderSideTarget();
    geometry::PointCloud source = target;
    source.Translate(Eigen::Vector3d(-0.12, 0.05, -0.35));
    const auto corres = MakeIdentityCorrespondences(source.points_.size());

    pipelines::registration::TransformationEstimationPointToPlaneDCReg dcreg;
    const double initial_rmse = dcreg.ComputeRMSE(source, target, corres);
    const Eigen::Matrix4d dcreg_update =
            dcreg.ComputeTransformation(source, target, corres);
    geometry::PointCloud dcreg_aligned = source;
    dcreg_aligned.Transform(dcreg_update);

    EXPECT_TRUE(dcreg_update.allFinite());
    EXPECT_GT(initial_rmse, 0.05);
    EXPECT_LT(dcreg.ComputeRMSE(dcreg_aligned, target, corres), 1e-8);
    EXPECT_NEAR(dcreg_update(0, 3), 0.12, 1e-8);
    EXPECT_NEAR(dcreg_update(1, 3), -0.05, 1e-8);
    EXPECT_NEAR(dcreg_update(2, 3), 0.0, 1e-8);
    EXPECT_LT((dcreg_update.block<3, 3>(0, 0) - Eigen::Matrix3d::Identity())
                      .norm(),
              1e-8);
}

TEST(TransformationEstimation, DCRegDegeneracyAnalysisCylinder) {
    const geometry::PointCloud target = MakeCylinderSideTarget();
    geometry::PointCloud source = target;
    source.Translate(Eigen::Vector3d(-0.12, 0.05, -0.35));
    const auto corres = MakeIdentityCorrespondences(source.points_.size());

    const pipelines::registration::DCRegDegeneracyAnalysis analysis =
            pipelines::registration::ComputeDCRegDegeneracyAnalysis(
                    source, target, corres);

    EXPECT_TRUE(analysis.has_correspondence_);
    EXPECT_TRUE(analysis.has_target_normals_);
    EXPECT_TRUE(analysis.is_rank_deficient_);
    EXPECT_TRUE(analysis.is_degenerate_);
    EXPECT_FALSE(analysis.schur_factorization_ok_);
    EXPECT_TRUE(analysis.schur_eigenvalues_rotation_.allFinite());
    EXPECT_TRUE(analysis.schur_eigenvalues_translation_.allFinite());
    EXPECT_TRUE(analysis.axis_aligned_eigenvalues_rotation_.allFinite());
    EXPECT_TRUE(analysis.axis_aligned_eigenvalues_translation_.allFinite());
    EXPECT_TRUE(std::isfinite(analysis.condition_number_full_));
    EXPECT_TRUE(std::isfinite(analysis.condition_number_rotation_));
    EXPECT_TRUE(std::isfinite(analysis.condition_number_translation_));
    EXPECT_EQ(analysis.weak_rotation_axes_(0), 0);
    EXPECT_EQ(analysis.weak_rotation_axes_(1), 0);
    EXPECT_EQ(analysis.weak_rotation_axes_(2), 1);
    EXPECT_EQ(analysis.weak_translation_axes_(0), 0);
    EXPECT_EQ(analysis.weak_translation_axes_(1), 0);
    EXPECT_EQ(analysis.weak_translation_axes_(2), 1);
    EXPECT_NEAR(analysis.axis_aligned_eigenvalues_translation_(2), 0.0, 1e-10);
    EXPECT_EQ(analysis.weak_rotation_axes_description_, "z");
    EXPECT_EQ(analysis.weak_translation_axes_description_, "z");
    EXPECT_NE(analysis.degeneracy_description_.find("target/world"),
              std::string::npos);
    EXPECT_NE(analysis.coordinate_frame_.find("left-multiplied SE(3)"),
              std::string::npos);
    EXPECT_EQ(analysis.solver_type_, "minimum_norm");
}

TEST(TransformationEstimation, DCRegLocalRegistrationUsesLocalPlane) {
    const geometry::PointCloud target = MakeOffsetPlaneTarget();
    geometry::PointCloud source = target;
    source.Translate(Eigen::Vector3d(0.0, 0.0, -0.1));

    pipelines::registration::DCRegOption option;
    option.local_frame_convergence_rotation_ = 1e-8;
    option.local_frame_convergence_translation_ = 1e-8;
    const pipelines::registration::ICPConvergenceCriteria criteria(1e-9, 1e-9,
                                                                   10);
    const pipelines::registration::RegistrationResult result =
            pipelines::registration::RegistrationICPDCRegLocal(
                    source, target, 0.5, Eigen::Matrix4d::Identity(), option,
                    criteria);

    EXPECT_TRUE(result.transformation_.allFinite());
    EXPECT_NEAR(result.transformation_(0, 3), 0.0, 1e-8);
    EXPECT_NEAR(result.transformation_(1, 3), 0.0, 1e-8);
    EXPECT_NEAR(result.transformation_(2, 3), 0.1, 1e-8);
    EXPECT_NEAR(result.fitness_, 1.0, 1e-12);
    EXPECT_NEAR(result.inlier_rmse_, 0.0, 1e-12);
}

TEST(TransformationEstimation, DCRegLocalDegeneracyAnalysisUsesLocalFrame) {
    const geometry::PointCloud target = MakeOffsetPlaneTarget();
    geometry::PointCloud source = target;
    source.Translate(Eigen::Vector3d(0.0, 0.0, -0.1));

    const pipelines::registration::DCRegDegeneracyAnalysis analysis =
            pipelines::registration::ComputeDCRegLocalDegeneracyAnalysis(
                    source, target, 0.5, Eigen::Matrix4d::Identity());

    EXPECT_TRUE(analysis.has_correspondence_);
    EXPECT_TRUE(analysis.has_target_normals_);
    EXPECT_TRUE(analysis.is_degenerate_);
    EXPECT_EQ(analysis.weak_rotation_axes_description_, "x, y, z");
    EXPECT_EQ(analysis.weak_translation_axes_description_, "x, y, z");
    EXPECT_NE(analysis.coordinate_frame_.find("local body frame"),
              std::string::npos);
    EXPECT_NE(analysis.degeneracy_description_.find("local-plane"),
              std::string::npos);
    EXPECT_EQ(analysis.solver_type_, "qr_fallback");
}

TEST(TransformationEstimation,
     PointToPlaneDCRegRegistrationICPMatchesBaseline) {
    const geometry::PointCloud target = MakeAsymmetricBoxTarget();
    const Eigen::Matrix4d expected = MakeSmallRegistrationTransform();
    geometry::PointCloud source = target;
    source.Transform(expected.inverse());

    const pipelines::registration::ICPConvergenceCriteria criteria(1e-9, 1e-9,
                                                                   20);
    const pipelines::registration::RegistrationResult baseline =
            pipelines::registration::RegistrationICP(
                    source, target, 0.2, Eigen::Matrix4d::Identity(),
                    pipelines::registration::
                            TransformationEstimationPointToPlane(),
                    criteria);
    const pipelines::registration::RegistrationResult dcreg =
            pipelines::registration::RegistrationICP(
                    source, target, 0.2, Eigen::Matrix4d::Identity(),
                    pipelines::registration::
                            TransformationEstimationPointToPlaneDCReg(),
                    criteria);

    EXPECT_TRUE(baseline.transformation_.allFinite());
    EXPECT_TRUE(dcreg.transformation_.allFinite());
    EXPECT_NEAR(baseline.fitness_, dcreg.fitness_, 1e-12);
    EXPECT_NEAR(baseline.inlier_rmse_, dcreg.inlier_rmse_, 1e-12);
    EXPECT_LT((baseline.transformation_ - dcreg.transformation_).norm(), 1e-8);
    EXPECT_LT((dcreg.transformation_ - expected).norm(), 1e-8);
}

TEST(TransformationEstimation, DISABLED_Constructor) { NotImplemented(); }

TEST(TransformationEstimation, DISABLED_Destructor) { NotImplemented(); }

TEST(TransformationEstimation, DISABLED_MemberData) { NotImplemented(); }

TEST(TransformationEstimation, DISABLED_GetTransformationEstimationType) {
    NotImplemented();
}

TEST(TransformationEstimation, DISABLED_ComputeRMSE) { NotImplemented(); }

TEST(TransformationEstimation, DISABLED_ComputeTransformation) {
    NotImplemented();
}

TEST(TransformationEstimation, DISABLED_TransformationEstimationPointToPoint) {
    NotImplemented();
}

TEST(TransformationEstimation, DISABLED_TransformationEstimationPointToPlane) {
    NotImplemented();
}

}  // namespace tests
}  // namespace open3d
