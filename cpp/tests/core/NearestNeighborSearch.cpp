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

#include "open3d/core/nns/NearestNeighborSearch.h"

#include <cmath>
#include <limits>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class NNSPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(NearestNeighborSearch,
                         NNSPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

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
    if (device.IsCUDA()) {
        EXPECT_THROW(nns32.FixedRadiusIndex(-1.0), std::runtime_error);
        EXPECT_THROW(nns32.FixedRadiusIndex(0.0), std::runtime_error);
    } else {
        nns32.FixedRadiusIndex();
        EXPECT_THROW(nns32.FixedRadiusSearch(query_points, -1.0),
                     std::runtime_error);
        EXPECT_THROW(nns32.FixedRadiusSearch(query_points, 0.0),
                     std::runtime_error);
    }

    // If raidus == 0.1.
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
    if (device.IsCUDA()) {
        EXPECT_THROW(nns64.FixedRadiusIndex(-1.0), std::runtime_error);
        EXPECT_THROW(nns64.FixedRadiusIndex(0.0), std::runtime_error);
    } else {
        nns64.FixedRadiusIndex();
        EXPECT_THROW(nns64.FixedRadiusSearch(query_points, -1.0),
                     std::runtime_error);
        EXPECT_THROW(nns64.FixedRadiusSearch(query_points, 0.0),
                     std::runtime_error);
    }

    // If raidus == 0.1.
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

}  // namespace tests
}  // namespace open3d
