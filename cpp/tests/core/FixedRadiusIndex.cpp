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

#include "open3d/core/nns/FixedRadiusIndex.h"

#include <cmath>
#include <limits>

#include "core/CoreTest.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

using namespace open3d;
using namespace std;

namespace open3d {
namespace tests {

// Function to find permutation to sort the given array.
template <class T>
std::vector<size_t> FindPermutation(T* vec, const int64_t size) {
    std::vector<size_t> p(size);
    std::iota(p.begin(), p.end(), 0);

    // find permutation
    std::sort(p.begin(), p.end(),
              [&](size_t i, size_t j) { return vec[i] < vec[j]; });
    return p;
}

// Function to apply permutation to the given array.
// It is in-place sorting.
template <class T>
void ApplyPermutation(T* vec, const std::vector<size_t> p) {
    std::vector<bool> done(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

TEST(FixedRadiusIndex, SearchRadius) {
    // Define test data.
    core::Device device = core::Device("CUDA:0");
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
    core::Tensor gt_indices, gt_distances, gt_neighbors_row_splits;

    // Set up index.
    float radius = 0.1;
    core::nns::FixedRadiusIndex index(dataset_points, radius);

    // if raidus == 0.1
    core::Tensor indices, distances, neighbors_row_splits;
    core::SizeVector shape{2};
    gt_indices = core::Tensor::Init<int32_t>({1, 4}, device);
    gt_distances = core::Tensor::Init<float>({0.00626358, 0.00747938}, device);
    gt_neighbors_row_splits = core::Tensor::Init<int64_t>({0, 2}, device);

    std::tie(indices, distances, neighbors_row_splits) =
            index.SearchRadius(query_points, radius);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(neighbors_row_splits.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(neighbors_row_splits.AllClose(gt_neighbors_row_splits));
}

TEST(FixedRadiusIndex, SearchRadiusBatch) {
    // Define test data.
    core::Device device = core::Device("CUDA:0");
    core::Tensor dataset_points = core::Tensor::Init<float>(
            {{0.719, 0.128, 0.431}, {0.764, 0.970, 0.678},
             {0.692, 0.786, 0.211}, {0.692, 0.969, 0.942},
             {0.803, 0.416, 0.863}, {0.285, 0.235, 0.058},
             {0.576, 0.759, 0.718}, {0.419, 0.183, 0.601},
             {0.221, 0.781, 0.229}, {0.492, 0.882, 0.958},
             {0.787, 0.585, 0.662}, {0.630, 0.846, 0.006},
             {0.863, 0.892, 0.848}, {0.809, 0.418, 0.544},
             {0.283, 0.054, 0.391}, {0.043, 0.589, 0.478},
             {0.824, 0.629, 0.629}, {0.074, 0.315, 0.639},
             {0.170, 0.545, 0.767}, {0.140, 0.912, 0.459}},
            device);
    core::Tensor dataset_points_row_splits =
            core::Tensor::Init<int64_t>({0, 10, 20});
    core::Tensor query_points = core::Tensor::Init<float>(
            {{0.982, 0.974, 0.936}, {0.225, 0.345, 0.679},
             {0.747, 0.779, 0.056}, {0.261, 0.955, 0.034},
             {0.255, 0.003, 0.849}, {0.821, 0.475, 0.149},
             {0.112, 0.228, 0.129}, {0.751, 0.174, 0.068},
             {0.738, 0.345, 0.695}, {0.343, 0.273, 0.450},
             {0.069, 0.720, 0.619}, {0.352, 0.947, 0.759},
             {0.424, 0.756, 0.403}, {0.422, 0.179, 0.769},
             {0.027, 0.831, 0.765}, {0.294, 0.300, 0.245},
             {0.011, 0.409, 0.045}, {0.277, 0.310, 0.172},
             {0.264, 0.483, 0.190}, {0.610, 0.623, 0.839},
             {0.500, 0.063, 0.602}, {0.150, 0.145, 0.272},
             {0.695, 0.501, 0.067}, {0.556, 0.775, 0.474},
             {0.766, 0.954, 0.898}},
            device);
    core::Tensor query_points_row_splits =
            core::Tensor::Init<int64_t>({0, 15, 25});
    std::vector<int32_t> gt_indices = {
            3,  1,  9,  7,  2,  8,  2,  7,  2,  0,  5,  0,  5,  4,  0,  7,
            6,  7,  5,  0,  8,  9,  6,  3,  1,  8,  2,  6,  1,  7,  0,  4,
            14, 15, 17, 15, 14, 15, 15, 14, 10, 16, 12, 13, 18, 14, 13, 17,
            14, 17, 11, 13, 16, 10, 19, 13, 11, 12, 12, 16, 10};
    std::vector<float> gt_distances = {
            0.084161,  0.114104, 0.249048, 0.069964, 0.0270990, 0.0699010,
            0.245651,  0.120800, 0.117206, 0.210337, 0.0350190, 0.134909,
            0.220977,  0.03749,  0.117146, 0.136841, 0.198169,  0.036677,
            0.158472,  0.162762, 0.178925, 0.063426, 0.0872010, 0.149573,
            0.176834,  0.07211,  0.109588, 0.122338, 0.237021,  0.028249,
            0.205054,  0.210166, 0.081953, 0.200811, 0.203861,  0.220913,
            0.113533,  0.226233, 0.143021, 0.224803, 0.0641020, 0.089932,
            0.136451,  0.168651, 0.204868, 0.091691, 0.22487,   0.246349,
            0.0401310, 0.169365, 0.126971, 0.247414, 0.117165,  0.124805,
            0.192050,  0.196358, 0.229541, 0.247814, 0.015753,  0.18135,
            0.192298};
    std::vector<int64_t> gt_neighbors_row_splits = {
            0,  3,  4,  5,  7,  8,  10, 11, 13, 17, 20, 21, 25,
            29, 32, 32, 35, 36, 38, 40, 45, 48, 50, 52, 58, 61};

    // Set up index.
    float radius = 0.5;
    core::nns::FixedRadiusIndex index;
    index.SetTensorData(dataset_points, dataset_points_row_splits, radius);

    // Test sort == false.
    core::Tensor indices, distances, neighbors_row_splits;
    std::tie(indices, distances, neighbors_row_splits) = index.SearchRadius(
            query_points, query_points_row_splits, radius, /*sort*/ false);

    // Check neighbor_row_splits first, since indices and distances checks are
    // dependent on the row splits value.
    ASSERT_EQ(neighbors_row_splits.ToFlatVector<int64_t>(),
              gt_neighbors_row_splits);

    std::vector<int32_t> indices_vector = indices.ToFlatVector<int32_t>();
    std::vector<float> distances_vector = distances.ToFlatVector<float>();
    std::vector<int32_t> gt_indices_sorted(gt_indices.begin(),
                                           gt_indices.end());
    std::vector<float> gt_distances_sorted(gt_distances.begin(),
                                           gt_distances.end());

    for (size_t i = 0; i < gt_neighbors_row_splits.size() - 1; i++) {
        int64_t size_i =
                gt_neighbors_row_splits[i + 1] - gt_neighbors_row_splits[i];

        // Sort predicted indices and distances
        std::vector<size_t> p_i = FindPermutation<int32_t>(
                indices_vector.data() + gt_neighbors_row_splits[i], size_i);
        ApplyPermutation(indices_vector.data() + gt_neighbors_row_splits[i],
                         p_i);
        ApplyPermutation(distances_vector.data() + gt_neighbors_row_splits[i],
                         p_i);

        // Sort gt indices and distances
        std::vector<size_t> gt_p_i = FindPermutation<int32_t>(
                gt_indices_sorted.data() + gt_neighbors_row_splits[i], size_i);
        ApplyPermutation(gt_indices_sorted.data() + gt_neighbors_row_splits[i],
                         gt_p_i);
        ApplyPermutation(
                gt_distances_sorted.data() + gt_neighbors_row_splits[i],
                gt_p_i);
    }

    ExpectEQ(indices_vector, gt_indices_sorted);
    ExpectEQ(distances_vector, gt_distances_sorted);

    // Test sort = true
    std::tie(indices, distances, neighbors_row_splits) = index.SearchRadius(
            query_points, query_points_row_splits, radius, /*sort*/ true);
    ExpectEQ(indices.ToFlatVector<int32_t>(), gt_indices);
    ExpectEQ(distances.ToFlatVector<float>(), gt_distances);
    ExpectEQ(neighbors_row_splits.ToFlatVector<int64_t>(),
             gt_neighbors_row_splits);
}

TEST(FixedRadiusIndex, SearchHybrid) {
    // Define test data.
    core::Device device = core::Device("CUDA:0");
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

    // Set up index.
    float radius = 0.1;
    int max_knn = 3;
    core::nns::FixedRadiusIndex index(dataset_points, radius);

    // if raidus == 0.1
    core::Tensor indices, distances, counts;
    core::SizeVector shape{1, 3};
    core::SizeVector shape_counts{1};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, -1}}, device);
    gt_distances =
            core::Tensor::Init<float>({{0.00626358, 0.00747938, 0}}, device);
    gt_counts = core::Tensor::Init<int32_t>({2}, device);

    std::tie(indices, distances, counts) =
            index.SearchHybrid(query_points, radius, max_knn);
    indices = indices.To(core::Int32);
    counts = counts.To(core::Int32);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(counts.GetShape(), shape_counts);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(counts.AllClose(gt_counts));
}

TEST(FixedRadiusIndex, SearchHybridBatch) {
    // Define test data.
    core::Device device = core::Device("CUDA:0");
    core::Tensor dataset_points = core::Tensor::Init<float>(
            {{0.719, 0.128, 0.431}, {0.764, 0.970, 0.678},
             {0.692, 0.786, 0.211}, {0.692, 0.969, 0.942},
             {0.803, 0.416, 0.863}, {0.285, 0.235, 0.058},
             {0.576, 0.759, 0.718}, {0.419, 0.183, 0.601},
             {0.221, 0.781, 0.229}, {0.492, 0.882, 0.958},
             {0.787, 0.585, 0.662}, {0.630, 0.846, 0.006},
             {0.863, 0.892, 0.848}, {0.809, 0.418, 0.544},
             {0.283, 0.054, 0.391}, {0.043, 0.589, 0.478},
             {0.824, 0.629, 0.629}, {0.074, 0.315, 0.639},
             {0.170, 0.545, 0.767}, {0.140, 0.912, 0.459}},
            device);
    core::Tensor dataset_points_row_splits =
            core::Tensor::Init<int64_t>({0, 10, 20});
    core::Tensor query_points = core::Tensor::Init<float>(
            {{0.982, 0.974, 0.936}, {0.225, 0.345, 0.679},
             {0.747, 0.779, 0.056}, {0.261, 0.955, 0.034},
             {0.255, 0.003, 0.849}, {0.821, 0.475, 0.149},
             {0.112, 0.228, 0.129}, {0.751, 0.174, 0.068},
             {0.738, 0.345, 0.695}, {0.343, 0.273, 0.450},
             {0.069, 0.720, 0.619}, {0.352, 0.947, 0.759},
             {0.424, 0.756, 0.403}, {0.422, 0.179, 0.769},
             {0.027, 0.831, 0.765}, {0.294, 0.300, 0.245},
             {0.011, 0.409, 0.045}, {0.277, 0.310, 0.172},
             {0.264, 0.483, 0.190}, {0.610, 0.623, 0.839},
             {0.500, 0.063, 0.602}, {0.150, 0.145, 0.272},
             {0.695, 0.501, 0.067}, {0.556, 0.775, 0.474},
             {0.766, 0.954, 0.898}},
            device);
    core::Tensor query_points_row_splits =
            core::Tensor::Init<int64_t>({0, 15, 25});
    std::vector<int32_t> gt_indices = {
            3,  1,  9,  7,  -1, -1, 2,  -1, -1, 8,  2,  -1, 7,  -1, -1,
            2,  0,  -1, 5,  -1, -1, 0,  5,  -1, 4,  0,  7,  7,  5,  0,
            8,  -1, -1, 9,  6,  3,  8,  2,  6,  7,  0,  4,  -1, -1, -1,
            14, 15, 17, 15, -1, -1, 14, 15, -1, 15, 14, -1, 10, 16, 12,
            14, 13, 17, 14, 17, -1, 11, 13, -1, 16, 10, 19, 12, 16, 10};

    // Set up index.
    float radius = 0.5;
    int max_knn = 3;
    core::nns::FixedRadiusIndex index;
    index.SetTensorData(dataset_points, dataset_points_row_splits, radius);

    core::Tensor indices, distances, counts;
    core::SizeVector shape{25, 3};

    std::tie(indices, distances, counts) = index.SearchHybrid(
            query_points, query_points_row_splits, radius, max_knn);

    ExpectEQ(indices.ToFlatVector<int32_t>(), gt_indices);
    ExpectEQ(indices.GetShape(), shape);
}
}  // namespace tests
}  // namespace open3d
