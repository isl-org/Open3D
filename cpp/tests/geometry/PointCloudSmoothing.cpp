// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// cppcheck-suppress missingIncludeSystem
#include <random>

#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/geometry/PointCloud.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

// Inverse-distance Laplacian / Taubin on meshes is regression-tested with fixed
// reference vertices in TriangleMesh.FilterSmoothLaplacian and
// TriangleMesh.FilterSmoothTaubin (same ApplyIndexedLaplacianUpdate core).

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

static geometry::PointCloud CreateTwoPointLine() {
    geometry::PointCloud pcd;
    pcd.points_ = {{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};
    return pcd;
}

// Unit-cube corner order: 000, 100, 010, 110, 001, 101, 011, 111.
static geometry::PointCloud CreateUnitCubeCorners() {
    geometry::PointCloud pcd;
    pcd.points_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                   {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    return pcd;
}

// This is a unit cube with a single displaced corner. The displacement makes
// its full-neighborhood MLS covariance have a unique smallest eigenvector.
static geometry::PointCloud CreatePerturbedUnitCubeCorners() {
    auto pcd = CreateUnitCubeCorners();
    pcd.points_[7].z() = 1.25;
    return pcd;
}

static geometry::PointCloud CreateNoisyPlaneWithPerPointAttributes(
        size_t n_points = 100, double noise_std = 0.01) {
    geometry::PointCloud pcd = CreateNoisyPlane(n_points, noise_std);
    pcd.colors_.resize(n_points);
    pcd.covariances_.resize(n_points);
    for (size_t i = 0; i < n_points; ++i) {
        const double value = static_cast<double>(i);
        pcd.colors_[i] = Eigen::Vector3d(value, value + 1.0, value + 2.0);
        pcd.covariances_[i] = Eigen::Matrix3d::Identity() * (value + 1.0);
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

TEST(PointCloudSmoothing, SmoothLaplacianExcludesSelfNeighbor) {
    auto pcd = CreateTwoPointLine();
    auto pcd_smoothed = pcd.SmoothLaplacian(1, 0.5, 1);

    std::vector<Eigen::Vector3d> expected_points = {
            {5.0, 0.0, 0.0},
            {5.0, 0.0, 0.0},
    };
    ExpectEQ(pcd_smoothed.points_, expected_points);
}

TEST(PointCloudSmoothing, SmoothLaplacian_UnitCubeReference) {
    auto pcd = CreateUnitCubeCorners();
    const auto pcd_smoothed = pcd.SmoothLaplacian(1, 0.5, 7, true);

    // Every corner uses the other seven corners. For 000, their mean is
    // (4/7, 4/7, 4/7), therefore 000 -> (2/7, 2/7, 2/7). The remaining
    // values follow by cube symmetry.
    const std::vector<Eigen::Vector3d> ref = {
            {2.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0},
            {5.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0},
            {2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0},
            {5.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0},
            {2.0 / 7.0, 2.0 / 7.0, 5.0 / 7.0},
            {5.0 / 7.0, 2.0 / 7.0, 5.0 / 7.0},
            {2.0 / 7.0, 5.0 / 7.0, 5.0 / 7.0},
            {5.0 / 7.0, 5.0 / 7.0, 5.0 / 7.0},
    };
    EXPECT_EQ(pcd_smoothed.points_.size(), pcd.points_.size());
    ExpectEQ(pcd_smoothed.points_, ref, 1e-12);
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

TEST(PointCloudSmoothing, SmoothLaplacianAndTaubinPreservePerPointAttributes) {
    auto pcd = CreateNoisyPlaneWithPerPointAttributes(80, 0.05);
    const auto laplacian = pcd.SmoothLaplacian(5, 0.5, 20, true);
    const auto laplacian_again = pcd.SmoothLaplacian(5, 0.5, 20, true);
    const auto taubin = pcd.SmoothTaubin(5, 0.5, -0.53, 20, true);
    const auto taubin_again = pcd.SmoothTaubin(5, 0.5, -0.53, 20, true);

    for (const auto* smoothed : {&laplacian, &taubin}) {
        EXPECT_EQ(pcd.points_.size(), smoothed->points_.size());
        ExpectEQ(pcd.colors_, smoothed->colors_);
        ExpectEQ(pcd.covariances_, smoothed->covariances_);
    }
    ExpectEQ(laplacian.points_, laplacian_again.points_);
    ExpectEQ(taubin.points_, taubin_again.points_);
}

TEST(PointCloudSmoothing, SmoothTaubin) {
    auto pcd = CreateNoisyPlane();
    double initial_noise = AveragePlaneDistance(pcd);

    auto pcd_smoothed = pcd.SmoothTaubin(10, 0.5, -0.5);
    double final_noise = AveragePlaneDistance(pcd_smoothed);

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_LT(final_noise, initial_noise);
}

TEST(PointCloudSmoothing, SmoothTaubinExcludesSelfNeighbor) {
    auto pcd = CreateTwoPointLine();
    auto pcd_smoothed = pcd.SmoothTaubin(1, 0.5, -0.5, 1);

    std::vector<Eigen::Vector3d> expected_points = {
            {5.0, 0.0, 0.0},
            {5.0, 0.0, 0.0},
    };
    ExpectEQ(pcd_smoothed.points_, expected_points);
}

TEST(PointCloudSmoothing, SmoothTaubin_UnitCubeReference) {
    auto pcd = CreateUnitCubeCorners();
    const auto pcd_smoothed = pcd.SmoothTaubin(1, 0.5, -0.53, 7, true);

    // A full-cube uniform-neighborhood pass scales displacement from the
    // center by (1 - 8 * factor / 7). The lambda and mu passes thus scale by
    // (3/7) * (1 + 4.24/7) = 0.688163265306..., giving these coordinates.
    const std::vector<Eigen::Vector3d> ref = {
            {0.155918367347, 0.155918367347, 0.155918367347},
            {0.844081632653, 0.155918367347, 0.155918367347},
            {0.155918367347, 0.844081632653, 0.155918367347},
            {0.844081632653, 0.844081632653, 0.155918367347},
            {0.155918367347, 0.155918367347, 0.844081632653},
            {0.844081632653, 0.155918367347, 0.844081632653},
            {0.155918367347, 0.844081632653, 0.844081632653},
            {0.844081632653, 0.844081632653, 0.844081632653},
    };
    EXPECT_EQ(pcd_smoothed.points_.size(), pcd.points_.size());
    ExpectEQ(pcd_smoothed.points_, ref, 1e-12);
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

TEST(PointCloudSmoothing, SmoothMLS_PerturbedUnitCubeReference) {
    auto pcd = CreatePerturbedUnitCubeCorners();
    const auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamKNN(8));

    // KNN-only MLS uses uniform weights. The displaced 111 corner makes the
    // covariance non-degenerate; weighted PCA followed by plane projection
    // gives the following hand-derived tangent-plane projections.
    const std::vector<Eigen::Vector3d> ref = {
            {0.270923615903, 0.270923615903, -0.175601750345},
            {0.857719411863, -0.142280588137, 0.092220533208},
            {-0.142280588137, 0.857719411863, 0.092220533208},
            {0.444515207823, 0.444515207823, 0.360042816760},
            {0.538745899455, 0.538745899455, 0.650806657735},
            {1.125541695415, 0.125541695415, 0.918628941287},
            {0.125541695415, 1.125541695415, 0.918628941287},
            {0.779293062264, 0.779293062264, 1.393053326860},
    };
    EXPECT_EQ(pcd_smoothed.points_.size(), pcd.points_.size());
    ExpectEQ(pcd_smoothed.points_, ref, 1e-9);
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

TEST(PointCloudSmoothing, SmoothMLS_PreservesPerPointAttributes) {
    auto pcd = CreateNoisyPlane(100, 0.1);
    pcd.colors_.resize(pcd.points_.size(), Eigen::Vector3d(0.1, 0.2, 0.3));
    pcd.covariances_.resize(pcd.points_.size(), Eigen::Matrix3d::Identity());

    auto pcd_smoothed = pcd.SmoothMLS(geometry::KDTreeSearchParamKNN(20));

    EXPECT_EQ(pcd.points_.size(), pcd_smoothed.points_.size());
    EXPECT_FALSE(pcd_smoothed.HasNormals());
    EXPECT_TRUE(pcd_smoothed.normals_.empty());
    ExpectEQ(pcd.colors_, pcd_smoothed.colors_);
    ExpectEQ(pcd.covariances_, pcd_smoothed.covariances_);
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

TEST(PointCloudSmoothing, SmoothBilateral_UnitCubeReference) {
    auto pcd = CreateUnitCubeCorners();
    pcd.normals_ = {
            {-1, -1, -1}, {1, -1, -1}, {-1, 1, -1}, {1, 1, -1},
            {-1, -1, 1},  {1, -1, 1},  {-1, 1, 1},  {1, 1, 1},
    };

    const auto pcd_smoothed =
            pcd.SmoothBilateral(geometry::KDTreeSearchParamKNN(8), 1.0, 1.0);

    // At 000, neighbors at Hamming distance h have aggregate weight
    // C(3,h) exp(-h/2 - h^2/6). Their weighted coordinate is
    // 0.298085265092; cube symmetry supplies the other corners.
    constexpr double low = 0.298085265092;
    constexpr double high = 1.0 - low;
    const std::vector<Eigen::Vector3d> ref = {
            {low, low, low},   {high, low, low},   {low, high, low},
            {high, high, low}, {low, low, high},   {high, low, high},
            {low, high, high}, {high, high, high},
    };
    EXPECT_EQ(pcd_smoothed.points_.size(), pcd.points_.size());
    ExpectEQ(pcd_smoothed.points_, ref, 1e-9);
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

TEST(PointCloudSmoothing, SmoothBilateral_NormalScaleInvariant) {
    auto pcd = CreateNoisyPlane(200, 0.1);
    pcd.EstimateNormals();

    auto scaled_normal_pcd = pcd;
    for (auto& normal : scaled_normal_pcd.normals_) {
        normal *= 7.0;
    }

    auto pcd_smoothed = pcd.SmoothBilateral(
            geometry::KDTreeSearchParamHybrid(0.2, 30), 0.1, 0.1);
    auto scaled_pcd_smoothed = scaled_normal_pcd.SmoothBilateral(
            geometry::KDTreeSearchParamHybrid(0.2, 30), 0.1, 0.1);

    ExpectEQ(pcd_smoothed.points_, scaled_pcd_smoothed.points_, 1e-9);
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
