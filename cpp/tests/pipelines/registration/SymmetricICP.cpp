// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/SymmetricICP.h"

#include <Eigen/Geometry>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "open3d/geometry/PointCloud.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

using namespace open3d::pipelines;

TEST(SymmetricICP, ComputeRMSEUsesSymmetricObjectiveAndAlignsNormals) {
    geometry::PointCloud source;
    geometry::PointCloud target;
    source.points_ = {{1.0, 0.0, 0.0}};
    source.normals_ = {{1.0, 0.0, 0.0}};
    target.points_ = {{0.5, std::sqrt(3.0) / 2.0, 0.0}};
    target.normals_ = {{0.5, std::sqrt(3.0) / 2.0, 0.0}};

    registration::CorrespondenceSet corres = {{0, 0}};
    registration::TransformationEstimationSymmetric estimation;

    EXPECT_NEAR(estimation.ComputeRMSE(source, target, corres), 0.0, 1e-12);
    target.normals_[0] *= -1.0;
    EXPECT_NEAR(estimation.ComputeRMSE(source, target, corres), 0.0, 1e-12);

    target.points_[0] = {0.0, 0.0, 0.0};
    target.normals_[0] = {1.0, 0.0, 0.0};
    EXPECT_NEAR(estimation.ComputeRMSE(source, target, corres), 2.0, 1e-12);
    target.normals_[0] *= -1.0;
    EXPECT_NEAR(estimation.ComputeRMSE(source, target, corres), 2.0, 1e-12);
}

TEST(SymmetricICP, ComputeTransformationRecoversFullRankRigidMotion) {
    geometry::PointCloud source;
    source.points_ = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                      {0.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}};
    source.normals_ = {{1.0, 2.0, 3.0},  {2.0, -1.0, 1.0},  {-1.0, 3.0, 2.0},
                       {3.0, 1.0, -2.0}, {-2.0, -1.0, 3.0}, {1.0, -3.0, 2.0}};
    for (Eigen::Vector3d &normal : source.normals_) {
        normal.normalize();
    }

    Eigen::Matrix4d expected = Eigen::Matrix4d::Identity();
    expected.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())
                    .toRotationMatrix();
    expected.block<3, 1>(0, 3) = Eigen::Vector3d(0.2, -0.1, 0.15);

    geometry::PointCloud target = source;
    target.Transform(expected);
    registration::CorrespondenceSet corres = {{0, 0}, {1, 1}, {2, 2},
                                              {3, 3}, {4, 4}, {5, 5}};
    registration::TransformationEstimationSymmetric estimation;

    const Eigen::Matrix4d actual =
            estimation.ComputeTransformation(source, target, corres);
    EXPECT_TRUE(actual.isApprox(expected, 1e-10));

    for (std::size_t i = 0; i < target.normals_.size(); i += 2) {
        target.normals_[i] *= -1.0;
    }
    const Eigen::Matrix4d actual_with_flipped_normals =
            estimation.ComputeTransformation(source, target, corres);
    EXPECT_TRUE(actual_with_flipped_normals.isApprox(expected, 1e-10));
}

TEST(SymmetricICP, TransformationEstimationType) {
    registration::TransformationEstimationSymmetric estimation;
    EXPECT_EQ(estimation.GetTransformationEstimationType(),
              registration::TransformationEstimationType::SymmetricICP);
    EXPECT_NE(registration::TransformationEstimationType::SymmetricICP,
              registration::TransformationEstimationType::PointToPlane);
}

TEST(SymmetricICP, InputContracts) {
    geometry::PointCloud source;
    geometry::PointCloud target;
    source.points_ = {{0.0, 0.0, 0.0}};
    source.normals_ = {{0.0, 0.0, 1.0}};
    target.points_ = {{0.0, 0.0, 0.0}};
    target.normals_ = {{0.0, 0.0, 1.0}};

    registration::CorrespondenceSet corres;
    registration::TransformationEstimationSymmetric estimation;

    EXPECT_EQ(estimation.ComputeRMSE(source, target, corres), 0.0);
    EXPECT_TRUE(estimation.ComputeTransformation(source, target, corres)
                        .isApprox(Eigen::Matrix4d::Identity()));

    corres = {{0, 0}};
    source.normals_.clear();
    EXPECT_THROW(estimation.ComputeRMSE(source, target, corres),
                 std::runtime_error);
    EXPECT_THROW(estimation.ComputeTransformation(source, target, corres),
                 std::runtime_error);

    source.normals_ = {{0.0, 0.0, 1.0}};
    target.normals_.clear();
    EXPECT_THROW(estimation.ComputeRMSE(source, target, corres),
                 std::runtime_error);
    EXPECT_THROW(estimation.ComputeTransformation(source, target, corres),
                 std::runtime_error);

    target.normals_ = {{0.0, 0.0, 1.0}};
    const auto expect_invalid_correspondence =
            [&](const registration::CorrespondenceSet &invalid_corres) {
                EXPECT_THROW(
                        estimation.ComputeRMSE(source, target, invalid_corres),
                        std::runtime_error);
                EXPECT_THROW(estimation.ComputeTransformation(source, target,
                                                              invalid_corres),
                             std::runtime_error);
            };
    expect_invalid_correspondence({{-1, 0}});
    expect_invalid_correspondence({{0, -1}});
    expect_invalid_correspondence({{1, 0}});
    expect_invalid_correspondence({{0, 1}});
}

}  // namespace tests
}  // namespace open3d
