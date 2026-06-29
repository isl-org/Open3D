// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/NearestNeighborSearch.h"

#include <cmath>
#include <limits>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class NNSPermuteDevices : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        NearestNeighborSearch,
        NNSPermuteDevices,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

TEST_P(NNSPermuteDevices, KnnSearch) {
    // Define test data.
    core::Device device = GetParam();
    core::Tensor dataset_points = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                             {0.0, 0.0, 0.1},
                                                             {0.0, 0.0, 0.2},
                                                             {0.0, 0.1, 0.0},
                                                             {0.0, 0.1, 0.1},
                                                             {0.0, 0.1, 0.2},
                                                             {0.0, 0.2, 0.0},
                                                             {0.0, 0.2, 0.1},
                                                             {0.0, 0.2, 0.2},
                                                             {0.1, 0.0, 0.0},
                                                             {0.1, 0.0, 0.1},
                                                             {0.1, 0.1, 0.0}},
                                                            device);
    core::Tensor query_points =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, device);
    core::Tensor gt_indices, gt_indices64, gt_distances;

    // int32 & int64
    // Set up index.
    core::nns::NearestNeighborSearch nns32(dataset_points, core::Int32);
    core::nns::NearestNeighborSearch nns64(dataset_points, core::Int64);
    nns32.KnnIndex();
    nns64.KnnIndex();

    // If k <= 0.
    EXPECT_THROW(nns32.KnnSearch(query_points, -1), std::runtime_error);
    EXPECT_THROW(nns32.KnnSearch(query_points, 0), std::runtime_error);
    EXPECT_THROW(nns64.KnnSearch(query_points, -1), std::runtime_error);
    EXPECT_THROW(nns64.KnnSearch(query_points, 0), std::runtime_error);

    // If k == 3.
    core::Tensor indices, indices64, distances, distances64;
    core::SizeVector shape{1, 3};
    gt_indices = core::Tensor::Init<int32_t>({{10, 1, 4}}, device);
    gt_indices64 = core::Tensor::Init<int64_t>({{10, 1, 4}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.00332258, 0.00626358, 0.00747938}}, device);

    std::tie(indices, distances) = nns32.KnnSearch(query_points, 3);
    std::tie(indices64, distances64) = nns64.KnnSearch(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(indices64.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(distances64.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(indices64.AllClose(gt_indices64));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(distances64.AllClose(gt_distances));

    // If k > size.
    shape = core::SizeVector{1, 12};
    gt_indices = core::Tensor::Init<int32_t>(
            {{10, 1, 4, 9, 11, 0, 3, 2, 5, 7, 6, 8}}, device);
    gt_indices64 = core::Tensor::Init<int64_t>(
            {{10, 1, 4, 9, 11, 0, 3, 2, 5, 7, 6, 8}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.00332258, 0.00626358, 0.00747938, 0.0108912, 0.0121070,
              0.0138322, 0.015048, 0.018695, 0.0199108, 0.0286952, 0.0362638,
              0.0411266}},
            device);

    std::tie(indices, distances) = nns32.KnnSearch(query_points, 14);
    std::tie(indices64, distances64) = nns64.KnnSearch(query_points, 14);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(indices64.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(distances64.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(indices64.AllClose(gt_indices64));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(distances64.AllClose(gt_distances));

    // Multiple points.
    query_points = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843}, {0.064705, 0.043921, 0.087843}},
            device);
    shape = core::SizeVector{2, 3};
    gt_indices = core::Tensor::Init<int32_t>({{10, 1, 4}, {10, 1, 4}}, device);
    gt_indices64 =
            core::Tensor::Init<int64_t>({{10, 1, 4}, {10, 1, 4}}, device);
    gt_distances =
            core::Tensor::Init<float>({{0.00332258, 0.00626358, 0.00747938},
                                       {0.00332258, 0.00626358, 0.00747938}},
                                      device);

    std::tie(indices, distances) = nns32.KnnSearch(query_points, 3);
    std::tie(indices64, distances64) = nns64.KnnSearch(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(indices64.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(distances64.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(indices64.AllClose(gt_indices64));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(distances64.AllClose(gt_distances));

    // Dimension > 3.
    dataset_points = dataset_points.Reshape({9, 4});
    core::nns::NearestNeighborSearch nns_new32(dataset_points, core::Int32);
    core::nns::NearestNeighborSearch nns_new64(dataset_points, core::Int64);
    nns_new32.KnnIndex();
    nns_new64.KnnIndex();

    core::Tensor query_points_new = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843, 0.0}}, device);
    shape = core::SizeVector{1, 3};
    gt_indices = core::Tensor::Init<int32_t>({{8, 7, 3}}, device);
    gt_indices64 = core::Tensor::Init<int64_t>({{8, 7, 3}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.00453838, 0.00626358, 0.00747938}}, device);

    std::tie(indices, distances) = nns_new32.KnnSearch(query_points_new, 3);
    std::tie(indices64, distances64) = nns_new64.KnnSearch(query_points_new, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(indices64.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(distances64.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(indices64.AllClose(gt_indices64));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(distances64.AllClose(gt_distances));
}

TEST_P(NNSPermuteDevices, FixedRadiusSearch) {
    // Define test data.
    core::Device device = GetParam();
    core::Tensor dataset_points = core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.1},
                                                              {0.0, 0.0, 0.2},
                                                              {0.0, 0.1, 0.0},
                                                              {0.0, 0.1, 0.1},
                                                              {0.0, 0.1, 0.2},
                                                              {0.0, 0.2, 0.0},
                                                              {0.0, 0.2, 0.1},
                                                              {0.0, 0.2, 0.2},
                                                              {0.1, 0.0, 0.0}},
                                                             device);
    core::Tensor query_points = core::Tensor::Init<double>(
            {{0.064705, 0.043921, 0.087843}}, device);
    core::Tensor gt_indices, gt_distances;

    // int32
    // Set up index.
    core::nns::NearestNeighborSearch nns32(dataset_points, core::Int32);

    // If radius <= 0.
    if (device.IsCUDA() || device.IsSYCL()) {
        EXPECT_THROW(nns32.FixedRadiusIndex(-1.0), std::runtime_error);
        EXPECT_THROW(nns32.FixedRadiusIndex(0.0), std::runtime_error);
    } else {
        nns32.FixedRadiusIndex();
        EXPECT_THROW(nns32.FixedRadiusSearch(query_points, -1.0),
                     std::runtime_error);
        EXPECT_THROW(nns32.FixedRadiusSearch(query_points, 0.0),
                     std::runtime_error);
    }

    // If radius == 0.1.
    nns32.FixedRadiusIndex(0.1);
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result;
    core::SizeVector shape{2};
    gt_indices = core::Tensor::Init<int32_t>({1, 4}, device);
    gt_distances = core::Tensor::Init<double>({0.00626358, 0.00747938}, device);

    result = nns32.FixedRadiusSearch(query_points, 0.1);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // int64
    // Set up index.
    core::nns::NearestNeighborSearch nns64(dataset_points, core::Int64);

    // If radius <= 0.
    if (device.IsCUDA() || device.IsSYCL()) {
        EXPECT_THROW(nns64.FixedRadiusIndex(-1.0), std::runtime_error);
        EXPECT_THROW(nns64.FixedRadiusIndex(0.0), std::runtime_error);
    } else {
        nns64.FixedRadiusIndex();
        EXPECT_THROW(nns64.FixedRadiusSearch(query_points, -1.0),
                     std::runtime_error);
        EXPECT_THROW(nns64.FixedRadiusSearch(query_points, 0.0),
                     std::runtime_error);
    }

    // If radius == 0.1.
    nns64.FixedRadiusIndex(0.1);
    gt_indices = core::Tensor::Init<int64_t>({1, 4}, device);
    gt_distances = core::Tensor::Init<double>({0.00626358, 0.00747938}, device);

    result = nns64.FixedRadiusSearch(query_points, 0.1);
    indices = std::get<0>(result);
    distances = std::get<1>(result);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
}

TEST(NearestNeighborSearch, MultiRadiusSearch) {
    // Define test data.
    core::Tensor dataset_points = core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                                              {0.0, 0.0, 0.1},
                                                              {0.0, 0.0, 0.2},
                                                              {0.0, 0.1, 0.0},
                                                              {0.0, 0.1, 0.1},
                                                              {0.0, 0.1, 0.2},
                                                              {0.0, 0.2, 0.0},
                                                              {0.0, 0.2, 0.1},
                                                              {0.0, 0.2, 0.2},
                                                              {0.1, 0.0, 0.0}});
    core::Tensor query_points = core::Tensor::Init<double>(
            {{0.064705, 0.043921, 0.087843}, {0.064705, 0.043921, 0.087843}});
    core::Tensor radius;
    core::Tensor gt_indices, gt_distances;

    // int32
    // Set up index.
    core::nns::NearestNeighborSearch nns32(dataset_points, core::Int32);
    nns32.MultiRadiusIndex();

    // If radius <= 0.
    radius = core::Tensor::Init<double>({1.0, 0.0});
    EXPECT_THROW(nns32.MultiRadiusSearch(query_points, radius),
                 std::runtime_error);
    EXPECT_THROW(nns32.MultiRadiusSearch(query_points, radius),
                 std::runtime_error);

    // If radius == 0.1.
    radius = core::Tensor::Init<double>({0.1, 0.1});
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result;
    core::SizeVector shape{4};
    gt_indices = core::Tensor::Init<int32_t>({1, 4, 1, 4});
    gt_distances = core::Tensor::Init<double>(
            {0.00626358, 0.00747938, 0.00626358, 0.00747938});

    result = nns32.MultiRadiusSearch(query_points, radius);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // int64
    // Set up index.
    core::nns::NearestNeighborSearch nns64(dataset_points, core::Int64);
    nns64.MultiRadiusIndex();

    // If radius <= 0.
    radius = core::Tensor::Init<double>({1.0, 0.0});
    EXPECT_THROW(nns64.MultiRadiusSearch(query_points, radius),
                 std::runtime_error);
    EXPECT_THROW(nns64.MultiRadiusSearch(query_points, radius),
                 std::runtime_error);

    // If radius == 0.1.
    radius = core::Tensor::Init<double>({0.1, 0.1});
    gt_indices = core::Tensor::Init<int64_t>({1, 4, 1, 4});
    gt_distances = core::Tensor::Init<double>(
            {0.00626358, 0.00747938, 0.00626358, 0.00747938});

    result = nns64.MultiRadiusSearch(query_points, radius);
    indices = std::get<0>(result);
    distances = std::get<1>(result);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
}

TEST_P(NNSPermuteDevices, HybridSearch) {
    // Define test data.
    core::Device device = GetParam();
    core::Tensor dataset_points = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                             {0.0, 0.0, 0.1},
                                                             {0.0, 0.0, 0.2},
                                                             {0.0, 0.1, 0.0},
                                                             {0.0, 0.1, 0.1},
                                                             {0.0, 0.1, 0.2},
                                                             {0.0, 0.2, 0.0},
                                                             {0.0, 0.2, 0.1},
                                                             {0.0, 0.2, 0.2},
                                                             {0.1, 0.0, 0.0}},
                                                            device);
    core::Tensor query_points =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, device);
    core::Tensor gt_indices, gt_distances, gt_counts;

    // int32
    // Set up index.
    core::nns::NearestNeighborSearch nns32(dataset_points, core::Int32);
    double radius = 0.1;
    int max_knn = 3;
    nns32.HybridIndex(radius);

    // test.
    core::Tensor indices, distances, counts;
    core::SizeVector shape{1, 3};
    core::SizeVector shape_counts{1};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, -1}}, device);
    gt_distances =
            core::Tensor::Init<float>({{0.00626358, 0.00747938, 0}}, device);
    gt_counts = core::Tensor::Init<int32_t>({2}, device);
    std::tie(indices, distances, counts) =
            nns32.HybridSearch(query_points, radius, max_knn);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(counts.GetShape(), shape_counts);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(counts.AllClose(gt_counts));

    // int64
    // Set up index.
    core::nns::NearestNeighborSearch nns64(dataset_points, core::Int64);
    nns64.HybridIndex(radius);

    // test.
    gt_indices = core::Tensor::Init<int64_t>({{1, 4, -1}}, device);
    gt_distances =
            core::Tensor::Init<float>({{0.00626358, 0.00747938, 0}}, device);
    gt_counts = core::Tensor::Init<int64_t>({2}, device);
    std::tie(indices, distances, counts) =
            nns64.HybridSearch(query_points, radius, max_knn);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(counts.GetShape(), shape_counts);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(counts.AllClose(gt_counts));
}

#ifdef BUILD_SYCL_MODULE
// ── SYCL-specific correctness and regression tests ────────────────────────

// Fixture helper: skip if no SYCL device is available.
struct SYCLNNSTest : public ::testing::Test {
    void SetUp() override {
        if (!core::sy::IsAvailable()) GTEST_SKIP() << "No SYCL device.";
    }
    const core::Device cpu{"CPU:0"};
    const core::Device sycl{"SYCL:0"};
};

// Shared 12-point dataset used by several tests below.
static core::Tensor MakeSYCLTestDataset(const core::Device& device) {
    return core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                      {0.0, 0.0, 0.1},
                                      {0.0, 0.0, 0.2},
                                      {0.0, 0.1, 0.0},
                                      {0.0, 0.1, 0.1},
                                      {0.0, 0.1, 0.2},
                                      {0.0, 0.2, 0.0},
                                      {0.0, 0.2, 0.1},
                                      {0.0, 0.2, 0.2},
                                      {0.1, 0.0, 0.0},
                                      {0.1, 0.0, 0.1},
                                      {0.1, 0.1, 0.0}},
                                     device);
}

// Parity test (regression): KNN SYCL vs CPU must agree on indices & distances.
TEST_F(SYCLNNSTest, KnnSearchMatchesCPU) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, cpu);

    core::nns::NearestNeighborSearch nns_cpu(dataset, core::Int32);
    nns_cpu.KnnIndex();
    core::Tensor indices_cpu, distances_cpu;
    std::tie(indices_cpu, distances_cpu) = nns_cpu.KnnSearch(query, 3);

    core::nns::NearestNeighborSearch nns_sycl(dataset.To(sycl), core::Int32);
    nns_sycl.KnnIndex();
    core::Tensor indices_sycl, distances_sycl;
    std::tie(indices_sycl, distances_sycl) =
            nns_sycl.KnnSearch(query.To(sycl), 3);

    EXPECT_TRUE(indices_sycl.To(cpu).AllClose(indices_cpu));
    EXPECT_TRUE(distances_sycl.To(cpu).AllClose(distances_cpu, 1e-5, 1e-5));
}

// C1: Query on a coincident point (distance = 0) must not produce a negative
// distance due to floating-point cancellation in −2qp + |p|².
TEST_F(SYCLNNSTest, KnnSearchCoincidentPoint_C1) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu).To(sycl);
    // Query exactly at dataset point 4 = (0.0, 0.1, 0.1).
    core::Tensor query = core::Tensor::Init<float>({{0.0, 0.1, 0.1}}, sycl);
    core::nns::NearestNeighborSearch nns(dataset, core::Int32);
    nns.KnnIndex();
    core::Tensor indices, distances;
    std::tie(indices, distances) = nns.KnnSearch(query, 3);
    // The nearest neighbor (point 4) must have distance exactly 0.0, not < 0.
    auto dists_cpu = distances.To(cpu);
    EXPECT_GE(dists_cpu[0][0].Item<float>(), 0.f);
    EXPECT_NEAR(dists_cpu[0][0].Item<float>(), 0.f, 1e-6f);
}

// C4: Equidistant neighbors must be returned with a consistent tie-break
// (smaller global index wins).
TEST_F(SYCLNNSTest, KnnSearchEquidistantTieBreak_C4) {
    // Three points equidistant from the origin: (1,0,0), (0,1,0), (0,0,1).
    core::Tensor dataset = core::Tensor::Init<float>(
            {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}, sycl);
    core::Tensor query = core::Tensor::Init<float>({{0.f, 0.f, 0.f}}, sycl);
    core::nns::NearestNeighborSearch nns(dataset, core::Int32);
    nns.KnnIndex();
    core::Tensor indices, distances;
    std::tie(indices, distances) = nns.KnnSearch(query, 3);
    auto idx = indices.To(cpu);
    // All distances equal 1.0; tie-break by index → expect 0, 1, 2.
    EXPECT_EQ(idx[0][0].Item<int32_t>(), 0);
    EXPECT_EQ(idx[0][1].Item<int32_t>(), 1);
    EXPECT_EQ(idx[0][2].Item<int32_t>(), 2);
    auto dists = distances.To(cpu);
    EXPECT_NEAR(dists[0][0].Item<float>(), 1.f, 1e-5f);
    EXPECT_NEAR(dists[0][1].Item<float>(), 1.f, 1e-5f);
    EXPECT_NEAR(dists[0][2].Item<float>(), 1.f, 1e-5f);
}

// C5: knn > num_points must be clamped (not crash, not return garbage).
TEST_F(SYCLNNSTest, KnnSearchMoreThanNumPoints_C5) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu).To(sycl);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, sycl);
    core::nns::NearestNeighborSearch nns(dataset, core::Int32);
    nns.KnnIndex();
    core::Tensor indices, distances;
    // Ask for 20 neighbors but only 12 exist.
    std::tie(indices, distances) = nns.KnnSearch(query, 20);
    EXPECT_EQ(indices.GetShape()[1], 12);  // clamped to num_points
    EXPECT_EQ(distances.GetShape()[1], 12);
    // All indices must be valid (0..11) – no -1 placeholder.
    auto idx_cpu = indices.To(cpu);
    for (int i = 0; i < 12; ++i) {
        EXPECT_GE(idx_cpu[0][i].Item<int32_t>(), 0);
        EXPECT_LT(idx_cpu[0][i].Item<int32_t>(), 12);
    }
}

// K-bucket path coverage: verify parity with CPU for k values that hit each
// dispatch bucket (1, 2, 4, 8, 16, 32 GRF path, 64 scratch path).
// Both queries are chosen to avoid equidistant neighbors so tie-breaking
// does not cause spurious CPU vs. SYCL index mismatches.
TEST_F(SYCLNNSTest, KnnSearchKBucketParityCPU) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    // Two off-lattice queries to avoid equidistant tie situations.
    core::Tensor query = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843}, {0.051, 0.031, 0.071}}, cpu);

    for (int k : {1, 2, 3, 4, 5, 8, 9, 16, 17, 32}) {
        core::nns::NearestNeighborSearch nns_cpu(dataset, core::Int32);
        nns_cpu.KnnIndex();
        core::Tensor idx_cpu, dist_cpu;
        std::tie(idx_cpu, dist_cpu) = nns_cpu.KnnSearch(query, k);

        core::nns::NearestNeighborSearch nns_sycl(dataset.To(sycl),
                                                  core::Int32);
        nns_sycl.KnnIndex();
        core::Tensor idx_sycl, dist_sycl;
        std::tie(idx_sycl, dist_sycl) = nns_sycl.KnnSearch(query.To(sycl), k);

        EXPECT_TRUE(idx_sycl.To(cpu).AllClose(idx_cpu))
                << "k=" << k << ": index mismatch";
        EXPECT_TRUE(dist_sycl.To(cpu).AllClose(dist_cpu, 1e-5f, 1e-5f))
                << "k=" << k << ": distance mismatch";
    }
}

// tile_bytes override: a smaller tile should produce the same result as the
// default (correctness across tile boundaries).
TEST_F(SYCLNNSTest, KnnSearchNonDefaultTileBytes) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu).To(sycl);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, sycl);

    core::nns::NearestNeighborSearch nns_default(dataset, core::Int32);
    nns_default.KnnIndex();
    core::Tensor idx_default, dist_default;
    std::tie(idx_default, dist_default) = nns_default.KnnSearch(query, 5);

    // Use a tiny tile (256 bytes) to force many tile iterations.
    core::nns::KnnIndex knn_tiny(dataset, core::Int32,
                                 /*tile_bytes=*/256LL);
    core::Tensor idx_tiny, dist_tiny;
    std::tie(idx_tiny, dist_tiny) = knn_tiny.SearchKnn(query, 5);

    EXPECT_TRUE(idx_tiny.To(cpu).AllClose(idx_default.To(cpu)));
    EXPECT_TRUE(dist_tiny.To(cpu).AllClose(dist_default.To(cpu), 1e-5f, 1e-5f));
}

// Radius search parity: SYCL vs CPU.
TEST_F(SYCLNNSTest, FixedRadiusSearchMatchesCPU) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, cpu);
    const double radius = 0.1;

    core::nns::NearestNeighborSearch nns_cpu(dataset, core::Int32);
    nns_cpu.FixedRadiusIndex(radius);
    core::Tensor idx_cpu, dist_cpu, splits_cpu;
    std::tie(idx_cpu, dist_cpu, splits_cpu) =
            nns_cpu.FixedRadiusSearch(query, radius);

    core::nns::NearestNeighborSearch nns_sycl(dataset.To(sycl), core::Int32);
    nns_sycl.FixedRadiusIndex(radius);
    core::Tensor idx_sycl, dist_sycl, splits_sycl;
    std::tie(idx_sycl, dist_sycl, splits_sycl) =
            nns_sycl.FixedRadiusSearch(query.To(sycl), radius);

    // row_splits dtype matches index_dtype (Int32); compare as int32.
    EXPECT_EQ(splits_sycl.To(cpu)[1].Item<int32_t>(),
              splits_cpu[1].Item<int32_t>());
    EXPECT_TRUE(idx_sycl.To(cpu).AllClose(idx_cpu));
    EXPECT_TRUE(dist_sycl.To(cpu).AllClose(dist_cpu, 1e-5f, 1e-5f));
}

// Radius search C1: coincident query must have distance exactly 0.
TEST_F(SYCLNNSTest, FixedRadiusSearchCoincidentPoint_C1) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu).To(sycl);
    // Query at point 4 = (0.0, 0.1, 0.1).
    core::Tensor query = core::Tensor::Init<float>({{0.0, 0.1, 0.1}}, sycl);
    const double radius = 0.05;
    core::nns::NearestNeighborSearch nns(dataset, core::Int32);
    nns.FixedRadiusIndex(radius);
    core::Tensor idx, dist, splits;
    std::tie(idx, dist, splits) = nns.FixedRadiusSearch(query, radius);
    // The coincident point itself must appear with distance >= 0.
    auto dist_cpu = dist.To(cpu);
    for (int64_t i = 0; i < dist_cpu.GetShape(0); ++i)
        EXPECT_GE(dist_cpu[i].Item<float>(), 0.f);
}

// Hybrid search parity: SYCL vs CPU.
TEST_F(SYCLNNSTest, HybridSearchMatchesCPU) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, cpu);
    const double radius = 0.1;
    const int max_knn = 3;

    core::nns::NearestNeighborSearch nns_cpu(dataset, core::Int32);
    nns_cpu.HybridIndex(radius);
    core::Tensor idx_cpu, dist_cpu, cnt_cpu;
    std::tie(idx_cpu, dist_cpu, cnt_cpu) =
            nns_cpu.HybridSearch(query, radius, max_knn);

    core::nns::NearestNeighborSearch nns_sycl(dataset.To(sycl), core::Int32);
    nns_sycl.HybridIndex(radius);
    core::Tensor idx_sycl, dist_sycl, cnt_sycl;
    std::tie(idx_sycl, dist_sycl, cnt_sycl) =
            nns_sycl.HybridSearch(query.To(sycl), radius, max_knn);

    EXPECT_EQ(cnt_sycl.To(cpu)[0].Item<int32_t>(), cnt_cpu[0].Item<int32_t>());
    // Compare only the valid (non-padded) entries.
    const int cnt = cnt_cpu[0].Item<int32_t>();
    for (int i = 0; i < cnt; ++i) {
        EXPECT_EQ(idx_sycl.To(cpu)[0][i].Item<int32_t>(),
                  idx_cpu[0][i].Item<int32_t>())
                << "Hybrid index mismatch at i=" << i;
        EXPECT_NEAR(dist_sycl.To(cpu)[0][i].Item<float>(),
                    dist_cpu[0][i].Item<float>(), 1e-5f)
                << "Hybrid distance mismatch at i=" << i;
    }
}

// tile_bytes override for FixedRadiusIndex (exercises P2 across tile bounds).
TEST_F(SYCLNNSTest, FixedRadiusSearchNonDefaultTileBytes) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu).To(sycl);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, sycl);
    const double radius = 0.15;

    core::nns::FixedRadiusIndex idx_default(dataset, radius, core::Int32);
    core::Tensor g_idx, g_dist, g_splits;
    std::tie(g_idx, g_dist, g_splits) = idx_default.SearchRadius(query, radius);

    core::nns::FixedRadiusIndex idx_tiny(dataset, radius, core::Int32,
                                         /*tile_bytes=*/256LL);
    core::Tensor t_idx, t_dist, t_splits;
    std::tie(t_idx, t_dist, t_splits) = idx_tiny.SearchRadius(query, radius);

    // row_splits dtype matches index_dtype (Int32 here); compare as int32.
    EXPECT_EQ(t_splits.To(cpu)[1].Item<int32_t>(),
              g_splits.To(cpu)[1].Item<int32_t>());
    EXPECT_TRUE(t_idx.To(cpu).AllClose(g_idx.To(cpu)));
    EXPECT_TRUE(t_dist.To(cpu).AllClose(g_dist.To(cpu), 1e-5f, 1e-5f));
}

#endif  // BUILD_SYCL_MODULE

}  // namespace tests
}  // namespace open3d
