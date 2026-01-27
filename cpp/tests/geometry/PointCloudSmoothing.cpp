// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <random>

#include "Eigen/Eigenvalues"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

// A point cloud that is a plane with some noise.
static geometry::PointCloud CreateNoisyPlane(size_t n_points = 100,
                                             double noise_std = 0.01) {
    geometry::PointCloud pcd;
    pcd.points_.resize(n_points);

    std::mt19937 rng(0);  // Fixed seed so that the test is deterministic
    std::uniform_real_distribution<double> dist_xy(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_z(-noise_std, noise_std);

    for (size_t i = 0; i < n_points; ++i) {
        pcd.points_[i] =
                Eigen::Vector3d(dist_xy(rng), dist_xy(rng), dist_z(rng));
    }
    return pcd;
}

// Computes the average absolute distance to the Z=0 plane.
static double AveragePlaneDistance(const geometry::PointCloud& pcd) {
    if (pcd.IsEmpty()) {
        return 0.0;
    }
    double total_dist = 0.0;
    for (const auto& point : pcd.points_) {
        total_dist += std::abs(point.z());
    }
    return total_dist / pcd.points_.size();
}

namespace {
// Helper functions copied from PointCloudSmoothing.cpp for testing.
Eigen::Vector3d ComputeCentroid(const std::vector<Eigen::Vector3d>& points,
                                const std::vector<int>& indices) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if (indices.empty()) {
        return centroid;
    }
    int valid_count = 0;
    for (int idx : indices) {
        if (idx < 0 || idx >= static_cast<int>(points.size())) {
            continue;
        }
        centroid += points[idx];
        ++valid_count;
    }
    if (valid_count > 0) {
        centroid /= static_cast<double>(valid_count);
    }
    return centroid;
}

Eigen::Vector3d ComputeWeightedCentroid(const open3d::geometry::PointCloud& pcd,
                                        const std::vector<int>& indices,
                                        const std::vector<double>& weights) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();

    if (indices.size() != weights.size() || indices.empty()) {
        return centroid;
    }

    double sum_w = 0.0;
    const size_t n_points = pcd.points_.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0 || idx >= static_cast<int>(n_points)) {
            continue;
        }

        const double w = weights[i];
        centroid += w * pcd.points_[idx];
        sum_w += w;
    }

    if (sum_w > 0.0) {
        centroid /= sum_w;
    }

    return centroid;
}

Eigen::Matrix3d ComputeWeightedCovariance(
        const open3d::geometry::PointCloud& pcd,
        const std::vector<int>& indices,
        const std::vector<double>& weights,
        const Eigen::Vector3d& centroid) {
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    if (indices.size() != weights.size() || indices.empty()) {
        return C;
    }
    const size_t n_points = pcd.points_.size();
    for (size_t i = 0; i < indices.size(); ++i) {
        const int idx = indices[i];
        if (idx < 0 || idx >= static_cast<int>(n_points)) {
            continue;
        }
        const Eigen::Vector3d diff = pcd.points_[idx] - centroid;
        C.noalias() += weights[i] * diff * diff.transpose();
    }
    return C;
}

Eigen::Vector3d ProjectOntoPlane(const Eigen::Vector3d& p,
                                 const Eigen::Vector3d& centroid,
                                 const Eigen::Vector3d& normal) {
    return p - normal * ((p - centroid).dot(normal));
}
}  // namespace

TEST(PointCloudSmoothingHelpers, ComputeCentroid) {
    std::vector<Eigen::Vector3d> points = {
            {0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {-1, -1, -1}};
    std::vector<int> indices = {0, 1, 2};
    Eigen::Vector3d centroid = ComputeCentroid(points, indices);
    ExpectEQ(centroid, Eigen::Vector3d(1, 1, 1));
}

TEST(PointCloudSmoothingHelpers, ComputeCentroid_EmptyIndices) {
    std::vector<Eigen::Vector3d> points = {{0, 0, 0}};
    std::vector<int> indices;
    Eigen::Vector3d centroid = ComputeCentroid(points, indices);
    ExpectEQ(centroid, Eigen::Vector3d::Zero().eval());
}

TEST(PointCloudSmoothingHelpers, ComputeCentroid_InvalidIndices) {
    std::vector<Eigen::Vector3d> points = {{1, 1, 1}};
    std::vector<int> indices = {0, -1, 100};
    // It should only average the valid index 0
    Eigen::Vector3d centroid = ComputeCentroid(points, indices);
    ExpectEQ(centroid, Eigen::Vector3d(1, 1, 1));
}

TEST(PointCloudSmoothingHelpers, ComputeWeightedCentroid) {
    geometry::PointCloud pcd;
    pcd.points_ = {{0, 0, 0}, {2, 2, 2}, {4, 4, 4}};
    std::vector<int> indices = {0, 1, 2};
    std::vector<double> weights = {1.0, 1.0, 1.0};
    Eigen::Vector3d centroid = ComputeWeightedCentroid(pcd, indices, weights);
    ExpectEQ(centroid, Eigen::Vector3d(2, 2, 2));

    weights = {1.0, 0.5, 0.0};
    centroid = ComputeWeightedCentroid(pcd, indices, weights);
    ExpectEQ(centroid, ((Eigen::Vector3d(0, 0, 0) * 1.0 +
                         Eigen::Vector3d(2, 2, 2) * 0.5) /
                        1.5)
                               .eval());
}

TEST(PointCloudSmoothingHelpers, ComputeWeightedCovariance) {
    geometry::PointCloud pcd;
    pcd.points_ = {{1, 0, 0}, {-1, 0, 0}};
    std::vector<int> indices = {0, 1};
    std::vector<double> weights = {1.0, 1.0};
    Eigen::Vector3d centroid = {0, 0, 0};
    Eigen::Matrix3d C =
            ComputeWeightedCovariance(pcd, indices, weights, centroid);

    Eigen::Matrix3d expected;
    expected << 2, 0, 0, 0, 0, 0, 0, 0, 0;
    ExpectEQ(C, expected);
}

TEST(PointCloudSmoothingHelpers, ProjectOntoPlane) {
    Eigen::Vector3d p(1, 1, 1);
    Eigen::Vector3d centroid(0, 0, 0);
    Eigen::Vector3d normal(0, 0, 1);
    Eigen::Vector3d projected = ProjectOntoPlane(p, centroid, normal);
    ExpectEQ(projected, Eigen::Vector3d(1, 1, 0));
}

// A point cloud with a step edge.
static geometry::PointCloud CreateStepEdge(size_t n_points_per_side = 50,
                                           double step_height = 0.5,
                                           double noise_std = 0.01) {
    geometry::PointCloud pcd;
    pcd.points_.resize(n_points_per_side * 2);

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist_x1(0.0, 1.0);
    std::uniform_real_distribution<double> dist_x2(-1.0, 0.0);
    std::uniform_real_distribution<double> dist_y(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_z(-noise_std, noise_std);

    // Side 1
    for (size_t i = 0; i < n_points_per_side; ++i) {
        pcd.points_[i] =
                Eigen::Vector3d(dist_x1(rng), dist_y(rng), dist_z(rng));
    }

    // Side 2
    for (size_t i = 0; i < n_points_per_side; ++i) {
        pcd.points_[n_points_per_side + i] = Eigen::Vector3d(
                dist_x2(rng), dist_y(rng), step_height + dist_z(rng));
    }
    return pcd;
}

// A point cloud with isolated points.
static geometry::PointCloud CreateCloudWithOutliers(size_t n_points = 100,
                                                    size_t n_outliers = 5,
                                                    double noise_std = 0.01) {
    geometry::PointCloud pcd = CreateNoisyPlane(n_points, noise_std);
    for (size_t i = 0; i < n_outliers; ++i) {
        pcd.points_.push_back(Eigen::Vector3d(10.0 + i, 10.0, 10.0));
    }
    return pcd;
}

TEST(PointCloudSmoothing, SmoothLaplacian) {
    auto pcd = CreateNoisyPlane();
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed = pcd.SmoothLaplacian(10, 0.5);
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothLaplacianEmpty) {
    geometry::PointCloud pcd;
    auto pcd_smoothed = pcd.SmoothLaplacian(1, 0.5);
    EXPECT_TRUE(pcd_smoothed.IsEmpty());
}

TEST(PointCloudSmoothing, SmoothLaplacianZeroIterations) {
    auto pcd = CreateNoisyPlane();
    auto pcd_smoothed = pcd.SmoothLaplacian(0, 0.5);
    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    ExpectEQ(pcd.points_, pcd_smoothed.points_);
}

TEST(PointCloudSmoothing, SmoothTaubin) {
    auto pcd = CreateNoisyPlane();
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed = pcd.SmoothTaubin(10, 0.5, -0.5);
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothTaubinEmpty) {
    geometry::PointCloud pcd;
    auto pcd_smoothed = pcd.SmoothTaubin(1, 0.5, -0.5);
    EXPECT_TRUE(pcd_smoothed.IsEmpty());
}

TEST(PointCloudSmoothing, SmoothTaubinZeroIterations) {
    auto pcd = CreateNoisyPlane();
    auto pcd_smoothed = pcd.SmoothTaubin(0, 0.5, -0.5);
    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    ExpectEQ(pcd.points_, pcd_smoothed.points_);
}

TEST(PointCloudSmoothing, SmoothMLS_KNN) {
    auto pcd = CreateNoisyPlane(100, 0.1);
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamKNN(20));
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothMLS_Radius) {
    auto pcd = CreateNoisyPlane(200, 0.1);
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamRadius(0.3));
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothMLS_Hybrid) {
    auto pcd = CreateNoisyPlane(200, 0.1);
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed =
            pcd.SmoothMLS(geometry::KDTreeSearchParamHybrid(0.3, 30));
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothMLS_Empty) {
    geometry::PointCloud pcd;
    auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamKNN(10));
    EXPECT_TRUE(pcd_smoothed.IsEmpty());
}

TEST(PointCloudSmoothing, SmoothMLS_InvalidParams) {
    auto pcd = CreateNoisyPlane();
    // Both k and radius are invalid
    auto pcd_smoothed =
            pcd.SmoothMLS(geometry::KDTreeSearchParamHybrid(-1, -1));
    ExpectEQ(pcd.points_, pcd_smoothed.points_);
}

TEST(PointCloudSmoothing, SmoothMLS_HandlesOutliers) {
    size_t n_points = 100, n_outliers = 5;
    auto pcd = CreateCloudWithOutliers(n_points, n_outliers, 0.01);

    // The outliers are far away, so they won't have enough neighbors
    auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamRadius(1.0));

    // Check that outliers were not moved
    for (size_t i = 0; i < n_outliers; ++i) {
        size_t index = n_points + i;
        ExpectEQ(pcd.points_[index], pcd_smoothed.points_[index]);
    }

    // Check that the plane part was smoothed
    double initial_noise = 0.0;
    for (size_t i = 0; i < n_points; ++i)
        initial_noise += std::abs(pcd.points_[i].z());
    initial_noise /= n_points;

    double final_noise = 0.0;
    for (size_t i = 0; i < n_points; ++i)
        final_noise += std::abs(pcd_smoothed.points_[i].z());
    final_noise /= n_points;

    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothBilateral) {
    auto pcd = CreateNoisyPlane(200, 0.1);
    double initial_noise = AveragePlaneDistance(pcd);

    // The filter needs normals.
    pcd.EstimateNormals();

    auto pcd_smoothed = pcd.SmoothBilateral(
            geometry::KDTreeSearchParamHybrid(0.2, 30), 0.1, 0.1);
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothBilateral_PreservesEdges) {
    auto pcd = CreateStepEdge(100, 0.5, 0.02);
    pcd.EstimateNormals();

    auto pcd_smoothed = pcd.SmoothBilateral(
            geometry::KDTreeSearchParamRadius(0.2), 0.1, 0.1);

    // Calculate average height of each side
    double avg_z1_before = 0, avg_z2_before = 0;
    double avg_z1_after = 0, avg_z2_after = 0;
    for (size_t i = 0; i < 100; ++i) avg_z1_before += pcd.points_[i].z();
    for (size_t i = 100; i < 200; ++i) avg_z2_before += pcd.points_[i].z();
    for (size_t i = 0; i < 100; ++i)
        avg_z1_after += pcd_smoothed.points_[i].z();
    for (size_t i = 100; i < 200; ++i)
        avg_z2_after += pcd_smoothed.points_[i].z();

    avg_z1_before /= 100;
    avg_z2_before /= 100;
    avg_z1_after /= 100;
    avg_z2_after /= 100;

    // The step height should be preserved
    double step_before = std::abs(avg_z2_before - avg_z1_before);
    double step_after = std::abs(avg_z2_after - avg_z1_after);
    EXPECT_NEAR(step_before, step_after, 0.1);

    // Noise on each plane should be reduced
    auto get_plane_noise = [](const geometry::PointCloud& pc, size_t start,
                              size_t end, double plane_z) {
        double noise = 0;
        for (size_t i = start; i < end; ++i)
            noise += std::abs(pc.points_[i].z() - plane_z);
        return noise / (end - start);
    };

    double noise1_before = get_plane_noise(pcd, 0, 100, avg_z1_before);
    double noise2_before = get_plane_noise(pcd, 100, 200, avg_z2_before);
    double noise1_after = get_plane_noise(pcd_smoothed, 0, 100, avg_z1_after);
    double noise2_after = get_plane_noise(pcd_smoothed, 100, 200, avg_z2_after);

    EXPECT_LT(noise1_after, noise1_before);
    EXPECT_LT(noise2_after, noise2_before);
}

TEST(PointCloudSmoothing, SmoothBilateral_NoNormals) {
    auto pcd = CreateNoisyPlane(200, 0.1);
    double initial_noise = AveragePlaneDistance(pcd);

    // This should work and estimate normals internally
    auto pcd_smoothed = pcd.SmoothBilateral(
            geometry::KDTreeSearchParamHybrid(0.2, 30), 0.1, 0.1);
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothBilateral_Empty) {
    geometry::PointCloud pcd;
    auto pcd_smoothed =
            pcd.SmoothBilateral(geometry::KDTreeSearchParamKNN(10), 0.1, 0.1);
    EXPECT_TRUE(pcd_smoothed.IsEmpty());
}

TEST(PointCloudSmoothing, SmoothBilateral_InvalidSigma) {
    auto pcd = CreateNoisyPlane();

    EXPECT_THROW(
            pcd.SmoothBilateral(geometry::KDTreeSearchParamKNN(10), -1.0, 0.1),
            std::runtime_error);
}

}  // namespace tests
}  // namespace open3d
