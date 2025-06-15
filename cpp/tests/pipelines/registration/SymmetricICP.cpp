// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/pipelines/registration/SymmetricICP.h"

#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/registration/Registration.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Random.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

using namespace open3d::pipelines;

TEST(SymmetricICP, TransformationEstimationSymmetricConstructor) {
    registration::TransformationEstimationSymmetric estimation;
    EXPECT_EQ(estimation.GetTransformationEstimationType(),
              registration::TransformationEstimationType::PointToPlane);
}

TEST(SymmetricICP, TransformationEstimationSymmetricComputeRMSE) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    source.points_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    source.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    target.points_ = {{0.1, 0.1, 0.1}, {1.1, 0.1, 0.1}, {0.1, 1.1, 0.1}};
    target.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    registration::CorrespondenceSet corres = {{0, 0}, {1, 1}, {2, 2}};
    registration::TransformationEstimationSymmetric estimation;

    double rmse = estimation.ComputeRMSE(source, target, corres);
    EXPECT_GT(rmse, 0.0);
}

TEST(SymmetricICP, TransformationEstimationSymmetricComputeRMSEEmptyCorres) {
    geometry::PointCloud source;
    geometry::PointCloud target;
    registration::CorrespondenceSet corres;
    registration::TransformationEstimationSymmetric estimation;

    double rmse = estimation.ComputeRMSE(source, target, corres);
    EXPECT_EQ(rmse, 0.0);
}

TEST(SymmetricICP, TransformationEstimationSymmetricComputeRMSENoNormals) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    source.points_ = {{0, 0, 0}, {1, 0, 0}};
    target.points_ = {{0.1, 0.1, 0.1}, {1.1, 0.1, 0.1}};

    registration::CorrespondenceSet corres = {{0, 0}, {1, 1}};
    registration::TransformationEstimationSymmetric estimation;

    double rmse = estimation.ComputeRMSE(source, target, corres);
    EXPECT_EQ(rmse, 0.0);
}

TEST(SymmetricICP, TransformationEstimationSymmetricComputeTransformation) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    // Create test point clouds with normals
    source.points_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    source.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    target.points_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    target.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    registration::CorrespondenceSet corres = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    registration::TransformationEstimationSymmetric estimation;

    Eigen::Matrix4d transformation =
            estimation.ComputeTransformation(source, target, corres);

    // Should be close to identity for perfect correspondence
    EXPECT_TRUE(transformation.isApprox(Eigen::Matrix4d::Identity(), 1e-3));
}

TEST(SymmetricICP,
     TransformationEstimationSymmetricComputeTransformationEmptyCorres) {
    geometry::PointCloud source;
    geometry::PointCloud target;
    registration::CorrespondenceSet corres;
    registration::TransformationEstimationSymmetric estimation;

    Eigen::Matrix4d transformation =
            estimation.ComputeTransformation(source, target, corres);

    EXPECT_TRUE(transformation.isApprox(Eigen::Matrix4d::Identity()));
}

TEST(SymmetricICP,
     TransformationEstimationSymmetricComputeTransformationNoNormals) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    source.points_ = {{0, 0, 0}, {1, 0, 0}};
    target.points_ = {{0.1, 0.1, 0.1}, {1.1, 0.1, 0.1}};
    // No normals

    registration::CorrespondenceSet corres = {{0, 0}, {1, 1}};
    registration::TransformationEstimationSymmetric estimation;

    Eigen::Matrix4d transformation =
            estimation.ComputeTransformation(source, target, corres);

    EXPECT_TRUE(transformation.isApprox(Eigen::Matrix4d::Identity()));
}

TEST(SymmetricICP, RegistrationSymmetricICP) {
    geometry::PointCloud source;
    geometry::PointCloud target;

    // Create test point clouds with normals
    source.points_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    source.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    // Target is slightly translated
    target.points_ = {
            {0.05, 0.05, 0.05}, {1.05, 0.05, 0.05}, {0.05, 1.05, 0.05}};
    target.normals_ = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};

    registration::TransformationEstimationSymmetric estimation;
    registration::ICPConvergenceCriteria criteria;

    registration::RegistrationResult result =
            registration::RegistrationSymmetricICP(source, target, 0.1,
                                                   Eigen::Matrix4d::Identity(),
                                                   estimation, criteria);

    EXPECT_GT(result.correspondence_set_.size(), 0);
    EXPECT_GT(result.fitness_, 0.0);
    EXPECT_GE(result.inlier_rmse_, 0.0);
}

TEST(SymmetricICP, RegistrationSymmetricICPConvergence) {
    utility::random::Seed(42);

    // Create a more complex test case
    geometry::PointCloud source;
    geometry::PointCloud target;

    // Generate random points with normals
    const int num_points = 50;
    utility::random::UniformRealGenerator<double> rand_gen(0.0, 10.0);
    for (int i = 0; i < num_points; ++i) {
        double x = rand_gen();
        double y = rand_gen();
        double z = rand_gen();

        source.points_.push_back({x, y, z});
        source.normals_.push_back({0, 0, 1});  // Simple normal for testing
    }

    // Create target by transforming source with known transformation
    Eigen::Matrix4d true_transformation = Eigen::Matrix4d::Identity();
    true_transformation(0, 3) = 0.1;   // Small translation in x
    true_transformation(1, 3) = 0.05;  // Small translation in y

    target = source;
    target.Transform(true_transformation);

    registration::TransformationEstimationSymmetric estimation;
    registration::ICPConvergenceCriteria criteria(1e-6, 1e-6, 30);

    registration::RegistrationResult result =
            registration::RegistrationSymmetricICP(source, target, 0.5,
                                                   Eigen::Matrix4d::Identity(),
                                                   estimation, criteria);

    // Check that registration converged to reasonable result
    EXPECT_GT(result.fitness_, 0.5);
    EXPECT_LT(result.inlier_rmse_, 1.0);
}

}  // namespace tests
}  // namespace open3d
