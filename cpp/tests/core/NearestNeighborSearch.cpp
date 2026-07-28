// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/nns/NearestNeighborSearch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

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
struct SYCLNNSTest : public ::testing::Test {
    void SetUp() override {
        if (!core::sy::IsAvailable()) GTEST_SKIP() << "No SYCL device.";
    }
    const core::Device cpu{"CPU:0"};
    const core::Device sycl{"SYCL:0"};
};
#endif

struct NNSParityTest : public testing::TestWithParam<core::Device> {
    void SetUp() override {
        if (GetParam().IsCPU()) {
            GTEST_SKIP() << "CPU is the oracle for this parity check.";
        }
        sycl = GetParam();
    }
    const core::Device cpu{"CPU:0"};
    core::Device sycl{"CPU:0"};
};

INSTANTIATE_TEST_SUITE_P(
        NearestNeighborSearchParity,
        NNSParityTest,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

// Shared 12-point dataset used by several tests below.
namespace {

core::Tensor MakeSYCLTestDataset(const core::Device& device) {
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

}  // namespace

// Parity test (regression): every accelerator backend must agree with CPU.
TEST_P(NNSParityTest, KnnSearchMatchesCPU) {
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

// Regression for high-dimensional, large-N KNN (mirrors FPFH feature matching:
// 33-dim, thousands of points, k=1). This exercises the multi-tile path and
// the −2qp+|p|² partial-distance selection with large feature norms, which the
// small 3D datasets above do not cover.
TEST_P(NNSParityTest, KnnSearchHighDimParityCPU) {
    const int64_t num_points = 5000;
    const int64_t num_queries = 1000;
    const int64_t dim = 33;
    // Deterministic pseudo-random features with LARGE norms but SMALL
    // inter-point distances (mirrors FPFH: similar geometry -> similar
    // histograms). The large offset makes |p|^2 ~ 33*100^2 ~ 3.3e5 while
    // neighbor distances are O(1), which stresses float32 cancellation in the
    // -2qp + |p|^2 + |q|^2 distance formulation.
    const float kOffset = 100.f;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<float> pts(num_points * dim), qrs(num_queries * dim);
    for (auto& v : pts) v = kOffset + dist(gen);
    for (auto& v : qrs) v = kOffset + dist(gen);
    core::Tensor dataset(pts, {num_points, dim}, core::Float32, cpu);
    core::Tensor query(qrs, {num_queries, dim}, core::Float32, cpu);

    core::nns::NearestNeighborSearch nns_cpu(dataset, core::Int32);
    nns_cpu.KnnIndex();
    core::Tensor idx_cpu, d_cpu;
    std::tie(idx_cpu, d_cpu) = nns_cpu.KnnSearch(query, 1);

    core::nns::NearestNeighborSearch nns_sycl(dataset.To(sycl), core::Int32);
    nns_sycl.KnnIndex();
    core::Tensor idx_sycl, d_sycl;
    std::tie(idx_sycl, d_sycl) = nns_sycl.KnnSearch(query.To(sycl), 1);

    // Fraction of queries whose k=1 nearest-neighbor index matches the CPU
    // reference. float32 vs CPU should agree on essentially all of them.
    core::Tensor match = idx_sycl.To(cpu).Eq(idx_cpu);
    int64_t num_match = match.To(core::Int64).Sum({0, 1}).Item<int64_t>();
    double valid_ratio = double(num_match) / double(num_queries);
    EXPECT_GT(valid_ratio, 0.99) << "valid_ratio=" << valid_ratio;
}

// C1: Query on a coincident point (distance = 0) must not produce a negative
// distance due to floating-point cancellation in −2qp + |p|².
TEST_P(NNSParityTest, KnnSearchCoincidentPoint_C1) {
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
// (smaller global index wins). SYCL KNN guarantees this; CUDA order may differ.
TEST_P(NNSParityTest, KnnSearchEquidistantTieBreak_C4) {
    if (!GetParam().IsSYCL()) {
        GTEST_SKIP() << "C4 tie-break order is specified for SYCL KNN.";
    }
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
TEST_P(NNSParityTest, KnnSearchMoreThanNumPoints_C5) {
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
TEST_P(NNSParityTest, KnnSearchKBucketParityCPU) {
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

// Double-precision K-bucket path coverage (regression for the subgroup-size
// dispatch in DispatchKnnDirectK): verify parity with CPU for k values that
// hit each dispatch bucket (1, 2, 4, 8, 16, 32), using double precision so
// the direct-distance path's double-only sub-group-width selection (SG=8 on
// devices that support it, SG=16 fallback otherwise) is exercised end to end
// on whatever the current device actually supports.
TEST_P(NNSParityTest, KnnSearchKBucketParityCPUDouble) {
    core::Tensor dataset_f = MakeSYCLTestDataset(cpu);
    core::Tensor dataset = dataset_f.To(core::Float64);
    core::Tensor query = core::Tensor::Init<double>(
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
        EXPECT_TRUE(dist_sycl.To(cpu).AllClose(dist_cpu, 1e-9, 1e-9))
                << "k=" << k << ": distance mismatch";
    }
}

#ifdef BUILD_SYCL_MODULE
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
#endif

// Radius search parity: SYCL vs CPU.
TEST_P(NNSParityTest, FixedRadiusSearchMatchesCPU) {
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

// core::nns::FixedRadiusIndex on CPU (impl::_FixedRadiusSearchCPU) ignores
// its `sort` argument -- unlike CUDA/SYCL, it never sorts by distance, so
// its output order is hash-table build order. Sort each query's segment by
// (distance, index) in-place so CPU-vs-SYCL parity checks below can compare
// with AllClose regardless of tie-break/build-order differences. `splits` is
// an exclusive prefix sum (row_splits) over num_queries segments.
namespace {

void SortSegmentsByDistanceCPU(core::Tensor& idx,
                               core::Tensor& dist,
                               const core::Tensor& splits) {
    const int64_t num_queries = splits.GetShape(0) - 1;
    for (int64_t q = 0; q < num_queries; ++q) {
        const int64_t begin = splits[q].Item<int32_t>();
        const int64_t end = splits[q + 1].Item<int32_t>();
        std::vector<std::pair<float, int32_t>> pairs;
        for (int64_t i = begin; i < end; ++i) {
            pairs.emplace_back(dist[i].Item<float>(), idx[i].Item<int32_t>());
        }
        std::sort(pairs.begin(), pairs.end());
        for (int64_t i = begin; i < end; ++i) {
            idx[i] = core::Tensor::Init<int32_t>(pairs[i - begin].second);
            dist[i] = core::Tensor::Init<float>(pairs[i - begin].first);
        }
    }
}

}  // namespace

// Linf metric parity (SparseConv uses Linf + float32).
TEST_P(NNSParityTest, FixedRadiusSearchLinfMatchesCPU) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    core::Tensor query =
            core::Tensor::Init<float>({{0.064705, 0.043921, 0.087843}}, cpu);
    const double radius = 0.1;
    core::Tensor queries_row_splits =
            core::Tensor::Init<int64_t>({0, query.GetShape(0)});

    core::nns::FixedRadiusIndex frs_cpu(dataset, radius, core::Int32);
    core::Tensor idx_cpu, dist_cpu, splits_cpu;
    std::tie(idx_cpu, dist_cpu, splits_cpu) = frs_cpu.SearchRadius(
            query, queries_row_splits, radius, true, core::nns::Linf, false);

    core::nns::FixedRadiusIndex frs_sycl(dataset.To(sycl), radius, core::Int32);
    core::Tensor idx_sycl, dist_sycl, splits_sycl;
    std::tie(idx_sycl, dist_sycl, splits_sycl) =
            frs_sycl.SearchRadius(query.To(sycl), queries_row_splits, radius,
                                  true, core::nns::Linf, false);
    idx_sycl = idx_sycl.To(cpu);
    dist_sycl = dist_sycl.To(cpu);
    splits_sycl = splits_sycl.To(cpu);

    // Sort both sides identically; CPU's `sort=true` above is a no-op (see
    // SortSegmentsByDistanceCPU), so compare on (distance, index) order.
    SortSegmentsByDistanceCPU(idx_cpu, dist_cpu, splits_cpu);
    SortSegmentsByDistanceCPU(idx_sycl, dist_sycl, splits_sycl);

    EXPECT_EQ(splits_sycl[1].Item<int32_t>(), splits_cpu[1].Item<int32_t>());
    EXPECT_TRUE(idx_sycl.AllClose(idx_cpu));
    EXPECT_TRUE(dist_sycl.AllClose(dist_cpu, 1e-5f, 1e-5f));
}

// ignore_query_point: coincident dataset point must be excluded when enabled.
TEST_P(NNSParityTest, FixedRadiusSearchIgnoreQueryPoint) {
    core::Tensor dataset = MakeSYCLTestDataset(cpu);
    core::Tensor query =
            core::Tensor::Init<float>({{0.0, 0.1, 0.1}}, cpu);  // point 4
    const double radius = 0.2;
    core::Tensor queries_row_splits = core::Tensor::Init<int64_t>({0, 1});

    core::nns::FixedRadiusIndex frs_cpu(dataset, radius, core::Int32);
    core::Tensor idx_ignore_cpu, dist_ignore_cpu, splits_ignore_cpu;
    std::tie(idx_ignore_cpu, dist_ignore_cpu, splits_ignore_cpu) =
            frs_cpu.SearchRadius(query, queries_row_splits, radius, true,
                                 core::nns::L2, true);
    core::Tensor idx_keep_cpu, dist_keep_cpu, splits_keep_cpu;
    std::tie(idx_keep_cpu, dist_keep_cpu, splits_keep_cpu) =
            frs_cpu.SearchRadius(query, queries_row_splits, radius, true,
                                 core::nns::L2, false);

    core::nns::FixedRadiusIndex frs_sycl(dataset.To(sycl), radius, core::Int32);
    core::Tensor idx_ignore_sycl, dist_ignore_sycl, splits_ignore_sycl;
    std::tie(idx_ignore_sycl, dist_ignore_sycl, splits_ignore_sycl) =
            frs_sycl.SearchRadius(query.To(sycl), queries_row_splits, radius,
                                  true, core::nns::L2, true);
    idx_ignore_sycl = idx_ignore_sycl.To(cpu);
    dist_ignore_sycl = dist_ignore_sycl.To(cpu);
    splits_ignore_sycl = splits_ignore_sycl.To(cpu);

    core::Tensor idx_keep_sycl, dist_keep_sycl, splits_keep_sycl;
    std::tie(idx_keep_sycl, dist_keep_sycl, splits_keep_sycl) =
            frs_sycl.SearchRadius(query.To(sycl), queries_row_splits, radius,
                                  true, core::nns::L2, false);
    idx_keep_sycl = idx_keep_sycl.To(cpu);
    dist_keep_sycl = dist_keep_sycl.To(cpu);
    splits_keep_sycl = splits_keep_sycl.To(cpu);

    // Sort both sides identically; CPU's `sort=true` above is a no-op (see
    // SortSegmentsByDistanceCPU), so compare on (distance, index) order.
    SortSegmentsByDistanceCPU(idx_ignore_cpu, dist_ignore_cpu,
                              splits_ignore_cpu);
    SortSegmentsByDistanceCPU(idx_ignore_sycl, dist_ignore_sycl,
                              splits_ignore_sycl);
    SortSegmentsByDistanceCPU(idx_keep_cpu, dist_keep_cpu, splits_keep_cpu);
    SortSegmentsByDistanceCPU(idx_keep_sycl, dist_keep_sycl, splits_keep_sycl);

    EXPECT_EQ(splits_ignore_sycl[1].Item<int32_t>(),
              splits_ignore_cpu[1].Item<int32_t>());
    EXPECT_TRUE(idx_ignore_sycl.AllClose(idx_ignore_cpu));
    EXPECT_TRUE(idx_keep_sycl.AllClose(idx_keep_cpu));
    // With ignore off, the query point itself is a neighbor at distance 0.
    EXPECT_EQ(splits_keep_cpu[1].Item<int32_t>(),
              splits_ignore_cpu[1].Item<int32_t>() + 1);
}

// Radius search C1: coincident query must have distance exactly 0.
TEST_P(NNSParityTest, FixedRadiusSearchCoincidentPoint_C1) {
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
TEST_P(NNSParityTest, HybridSearchMatchesCPU) {
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

// Regression for hybrid search on large-offset 3D coordinates with a small
// radius (mirrors registration: room-scale scans, max_correspondence_distance).
// Exercises float32 cancellation in the radius count/threshold path.
TEST_P(NNSParityTest, HybridSearchLargeOffsetParityCPU) {
    const int64_t n = 4000;
    const float kOffset = 1000.f;  // non-origin-centered scan coordinates
    const double radius = 0.05;
    const int max_knn = 1;
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> spread(0.f, 3.f);       // 3 m extent
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);  // < radius
    std::vector<float> pts(n * 3), qrs(n * 3);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            float v = kOffset + spread(gen);
            pts[i * 3 + d] = v;
            qrs[i * 3 + d] = v + jitter(gen);  // a close neighbor exists
        }
    }
    core::Tensor dataset(pts, {n, 3}, core::Float32, cpu);
    core::Tensor query(qrs, {n, 3}, core::Float32, cpu);

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

    int64_t total_cpu = cnt_cpu.To(core::Int64).Sum({0}).Item<int64_t>();
    int64_t total_sycl =
            cnt_sycl.To(cpu).To(core::Int64).Sum({0}).Item<int64_t>();
    EXPECT_GT(total_cpu, n / 2);  // most queries should have a neighbor
    EXPECT_EQ(total_sycl, total_cpu) << "SYCL hybrid count mismatch";
}

// tile_bytes override for FixedRadiusIndex (exercises P2 across tile bounds).
#ifdef BUILD_SYCL_MODULE
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
