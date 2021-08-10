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

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(FixedRadiusIndex, SearchRadius) {
    core::Device device = core::Device("CUDA:0");
    std::vector<int> ref_indices = {1, 4};
    std::vector<float> ref_distance = {0.00626358, 0.00747938};

    int size = 10;
    std::vector<float> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0,
                              0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2,
                              0.0, 0.0, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Float32, device);
    float radius = 0.1;
    core::nns::FixedRadiusIndex index(ref, radius);

    core::Tensor query(std::vector<float>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Float32, device);

    // if radius == 0.1
    core::Tensor indices, distances, neighbors_row_splits;
    std::tie(indices, distances, neighbors_row_splits) =
            index.SearchRadius(query, radius);
    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4}));
    ExpectEQ(distances.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938}));
}

TEST(FixedRadiusIndex, SearchRadiusBatch) {
    core::Device device = core::Device("CUDA:0");
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

    std::vector<float> points{
            0.719, 0.128, 0.431, 0.764, 0.970, 0.678, 0.692, 0.786, 0.211,
            0.692, 0.969, 0.942, 0.803, 0.416, 0.863, 0.285, 0.235, 0.058,
            0.576, 0.759, 0.718, 0.419, 0.183, 0.601, 0.221, 0.781, 0.229,
            0.492, 0.882, 0.958, 0.787, 0.585, 0.662, 0.630, 0.846, 0.006,
            0.863, 0.892, 0.848, 0.809, 0.418, 0.544, 0.283, 0.054, 0.391,
            0.043, 0.589, 0.478, 0.824, 0.629, 0.629, 0.074, 0.315, 0.639,
            0.170, 0.545, 0.767, 0.140, 0.912, 0.459};
    core::Tensor dataset_points(points, {20, 3}, core::Float32, device);
    core::Tensor points_row_splits(std::vector<int64_t>({0, 10, 20}), {3},
                                   core::Int64);

    std::vector<float> queries{
            0.982, 0.974, 0.936, 0.225, 0.345, 0.679, 0.747, 0.779, 0.056,
            0.261, 0.955, 0.034, 0.255, 0.003, 0.849, 0.821, 0.475, 0.149,
            0.112, 0.228, 0.129, 0.751, 0.174, 0.068, 0.738, 0.345, 0.695,
            0.343, 0.273, 0.450, 0.069, 0.720, 0.619, 0.352, 0.947, 0.759,
            0.424, 0.756, 0.403, 0.422, 0.179, 0.769, 0.027, 0.831, 0.765,
            0.294, 0.300, 0.245, 0.011, 0.409, 0.045, 0.277, 0.310, 0.172,
            0.264, 0.483, 0.190, 0.610, 0.623, 0.839, 0.500, 0.063, 0.602,
            0.150, 0.145, 0.272, 0.695, 0.501, 0.067, 0.556, 0.775, 0.474,
            0.766, 0.954, 0.898};
    core::Tensor query_points(queries, {25, 3}, core::Float32, device);
    core::Tensor queries_row_splits(std::vector<int64_t>({0, 15, 25}), {3},
                                    core::Int64);

    float radius = 0.5;
    core::nns::FixedRadiusIndex index;
    index.SetTensorData(dataset_points, points_row_splits, radius);

    core::Tensor indices, distances, neighbor_row_splits;

    // Test sort = false.
    std::tie(indices, distances, neighbor_row_splits) = index.SearchRadius(
            query_points, queries_row_splits, radius, /*sort*/ false);

    std::vector<int32_t> indices_vector = indices.ToFlatVector<int32_t>();
    std::vector<float> distances_vector = distances.ToFlatVector<float>();
    std::vector<int32_t> gt_indices_unsorted(gt_indices.begin(),
                                             gt_indices.end());
    std::vector<float> gt_distances_unsorted(gt_distances.begin(),
                                             gt_distances.end());
    for (size_t i = 0; i < gt_neighbors_row_splits.size() - 1; i++) {
        std::sort(indices_vector.begin() + gt_neighbors_row_splits[i],
                  indices_vector.begin() + gt_neighbors_row_splits[i + 1]);
        std::sort(distances_vector.begin() + gt_neighbors_row_splits[i],
                  distances_vector.begin() + gt_neighbors_row_splits[i + 1]);
        std::sort(gt_indices_unsorted.begin() + gt_neighbors_row_splits[i],
                  gt_indices_unsorted.begin() + gt_neighbors_row_splits[i + 1]);
        std::sort(gt_distances_unsorted.begin() + gt_neighbors_row_splits[i],
                  gt_distances_unsorted.end() + gt_neighbors_row_splits[i + 1]);
    }
    ExpectEQ(indices_vector, gt_indices_unsorted);
    ExpectEQ(distances_vector, gt_distances_unsorted);
    ExpectEQ(neighbor_row_splits.ToFlatVector<int64_t>(),
             gt_neighbors_row_splits);

    // Test sort = true
    std::tie(indices, distances, neighbor_row_splits) = index.SearchRadius(
            query_points, queries_row_splits, radius, /*sort*/ true);
    ExpectEQ(indices.ToFlatVector<int32_t>(), gt_indices);
    ExpectEQ(distances.ToFlatVector<float>(), gt_distances);
    ExpectEQ(neighbor_row_splits.ToFlatVector<int64_t>(),
             gt_neighbors_row_splits);
}

TEST(FixedRadiusIndex, SearchHybrid) {
    core::Device device = core::Device("CUDA:0");
    std::vector<int32_t> gt_indices = {1, 4, -1};
    std::vector<float> gt_distances = {0.00626358, 0.00747938, 0};

    float radius = 0.1;
    int max_knn = 3;

    std::vector<float> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0,
                              0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2,
                              0.0, 0.0, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor dataset_points(points, {10, 3}, core::Float32, device);
    core::nns::FixedRadiusIndex index(dataset_points, radius);

    core::Tensor query_points(
            std::vector<float>({0.064705, 0.043921, 0.087843}), {1, 3},
            core::Float32, device);

    // if radius == 0.1
    core::Tensor indices, distances, counts;
    std::tie(indices, distances, counts) =
            index.SearchHybrid(query_points, radius, max_knn);

    ExpectEQ(indices.ToFlatVector<int32_t>(), gt_indices);
    ExpectEQ(distances.ToFlatVector<float>(), gt_distances);
}

TEST(FixedRadiusIndex, SearchHybridBatch) {
    core::Device device = core::Device("CUDA:0");
    std::vector<int32_t> gt_indices = {
            3,  1,  9,  7,  -1, -1, 2,  -1, -1, 8,  2,  -1, 7,  -1, -1,
            2,  0,  -1, 5,  -1, -1, 0,  5,  -1, 4,  0,  7,  7,  5,  0,
            8,  -1, -1, 9,  6,  3,  8,  2,  6,  7,  0,  4,  -1, -1, -1,
            14, 15, 17, 15, -1, -1, 14, 15, -1, 15, 14, -1, 10, 16, 12,
            14, 13, 17, 14, 17, -1, 11, 13, -1, 16, 10, 19, 12, 16, 10};

    std::vector<float> points{
            0.719, 0.128, 0.431, 0.764, 0.970, 0.678, 0.692, 0.786, 0.211,
            0.692, 0.969, 0.942, 0.803, 0.416, 0.863, 0.285, 0.235, 0.058,
            0.576, 0.759, 0.718, 0.419, 0.183, 0.601, 0.221, 0.781, 0.229,
            0.492, 0.882, 0.958, 0.787, 0.585, 0.662, 0.630, 0.846, 0.006,
            0.863, 0.892, 0.848, 0.809, 0.418, 0.544, 0.283, 0.054, 0.391,
            0.043, 0.589, 0.478, 0.824, 0.629, 0.629, 0.074, 0.315, 0.639,
            0.170, 0.545, 0.767, 0.140, 0.912, 0.459};
    core::Tensor dataset_points(points, {20, 3}, core::Float32, device);
    core::Tensor points_row_splits(std::vector<int64_t>({0, 10, 20}), {3},
                                   core::Int64);

    std::vector<float> queries{
            0.982, 0.974, 0.936, 0.225, 0.345, 0.679, 0.747, 0.779, 0.056,
            0.261, 0.955, 0.034, 0.255, 0.003, 0.849, 0.821, 0.475, 0.149,
            0.112, 0.228, 0.129, 0.751, 0.174, 0.068, 0.738, 0.345, 0.695,
            0.343, 0.273, 0.450, 0.069, 0.720, 0.619, 0.352, 0.947, 0.759,
            0.424, 0.756, 0.403, 0.422, 0.179, 0.769, 0.027, 0.831, 0.765,
            0.294, 0.300, 0.245, 0.011, 0.409, 0.045, 0.277, 0.310, 0.172,
            0.264, 0.483, 0.190, 0.610, 0.623, 0.839, 0.500, 0.063, 0.602,
            0.150, 0.145, 0.272, 0.695, 0.501, 0.067, 0.556, 0.775, 0.474,
            0.766, 0.954, 0.898};
    core::Tensor query_points(queries, {25, 3}, core::Float32, device);
    core::Tensor queries_row_splits(std::vector<int64_t>({0, 15, 25}), {3},
                                    core::Int64);

    float radius = 0.5;
    int max_knn = 3;
    core::nns::FixedRadiusIndex index;
    index.SetTensorData(dataset_points, points_row_splits, radius);

    core::Tensor indices, distances, counts;

    std::tie(indices, distances, counts) = index.SearchHybrid(
            query_points, queries_row_splits, radius, max_knn);

    ExpectEQ(indices.ToFlatVector<int32_t>(), gt_indices);
    ExpectEQ(indices.GetShape(), core::SizeVector{25, 3});
}
}  // namespace tests
}  // namespace open3d