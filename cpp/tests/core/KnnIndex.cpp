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
#include "open3d/core/nns/KnnIndex.h"

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

TEST(KnnIndex, KnnSearch) {
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
    core::Tensor gt_indices, gt_distances;

    // Set up Knn index.
    core::nns::KnnIndex knn_index(dataset_points);

    // If k <= 0.
    EXPECT_THROW(knn_index.SearchKnn(query_points, -1), std::runtime_error);
    EXPECT_THROW(knn_index.SearchKnn(query_points, 0), std::runtime_error);

    // If k == 3.
    core::Tensor indices, distances;
    core::SizeVector shape{1, 3};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, 9}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.00626358, 0.00747938, 0.0108912}}, device);

    std::tie(indices, distances) = knn_index.SearchKnn(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // If k > size.
    shape = core::SizeVector{1, 10};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, 9, 0, 3, 2, 5, 7, 6, 8}},
                                             device);
    gt_distances = core::Tensor::Init<float>(
            {{0.00626358, 0.00747938, 0.0108912, 0.0138322, 0.015048, 0.018695,
              0.0199108, 0.0286952, 0.0362638, 0.0411266}},
            device);
    std::tie(indices, distances) = knn_index.SearchKnn(query_points, 12);
    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // Multiple points.
    query_points = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843}, {0.064705, 0.043921, 0.087843}},
            device);
    shape = core::SizeVector{2, 3};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, 9}, {1, 4, 9}}, device);
    gt_distances =
            core::Tensor::Init<float>({{0.00626358, 0.00747938, 0.0108912},
                                       {0.00626358, 0.00747938, 0.0108912}},
                                      device);
    std::tie(indices, distances) = knn_index.SearchKnn(query_points, 3);
    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
}

TEST(KnnIndex, KnnSearchHighdim) {
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
    core::Tensor query_points;
    core::Tensor gt_indices, gt_distances;

    // Dimension = 5.
    dataset_points = dataset_points.Reshape({-1, 5});
    query_points = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843, 0.0, 0.0}}, device);
    core::nns::KnnIndex knn_index(dataset_points);

    // If k <= 0.
    EXPECT_THROW(knn_index.SearchKnn(query_points, -1), std::runtime_error);
    EXPECT_THROW(knn_index.SearchKnn(query_points, 0), std::runtime_error);

    // If k == 3.
    core::Tensor indices, distances;
    core::SizeVector shape{1, 3};
    gt_indices = core::Tensor::Init<int32_t>({{0, 4, 2}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.01383218, 0.02869498, 0.03089118}}, device);

    std::tie(indices, distances) = knn_index.SearchKnn(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // Dimension = 6.
    dataset_points = dataset_points.Reshape({-1, 6});
    query_points = core::Tensor::Init<float>(
            {{0.064705, 0.043921, 0.087843, 0.0, 0.0, 0.0}}, device);
    knn_index.SetTensorData(dataset_points);

    // If k <= 0.
    EXPECT_THROW(knn_index.SearchKnn(query_points, -1), std::runtime_error);
    EXPECT_THROW(knn_index.SearchKnn(query_points, 0), std::runtime_error);

    // If k == 3.
    gt_indices = core::Tensor::Init<int32_t>({{0, 1, 4}}, device);
    gt_distances = core::Tensor::Init<float>(
            {{0.02383218, 0.02869498, 0.05112658}}, device);

    std::tie(indices, distances) = knn_index.SearchKnn(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
}

TEST(KnnIndex, KnnSearchBatch) {
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
    core::Tensor points_row_splits = core::Tensor::Init<int64_t>({0, 10, 20});
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
    core::Tensor queries_row_splits = core::Tensor::Init<int64_t>({0, 15, 25});

    // Set up Knn index.
    core::nns::KnnIndex index;
    index.SetTensorData(dataset_points, points_row_splits);

    // If k == 3.
    core::Tensor indices, distances;
    core::SizeVector shape{25 * 3};
    core::Tensor gt_indices =
            core::Tensor::Init<int32_t>(
                    {{3, 1, 9}, {7, 6, 0}, {2, 8, 1}, {8, 2, 5}, {7, 0, 4},
                     {2, 0, 5}, {5, 7, 8}, {0, 5, 7}, {4, 0, 7}, {7, 5, 0},
                     {8, 6, 9}, {9, 6, 3}, {8, 2, 6}, {7, 0, 4}, {9, 6, 8},
                     {4, 5, 7}, {5, 4, 7}, {4, 5, 7}, {5, 4, 7}, {0, 6, 2},
                     {4, 3, 7}, {4, 7, 5}, {1, 3, 6}, {6, 0, 9}, {2, 6, 0}},
                    device)
                    .Reshape({-1});
    core::Tensor gt_distances =
            core::Tensor::Init<float>(
                    {{0.084, 0.114, 0.249}, {0.070, 0.296, 0.353},
                     {0.027, 0.307, 0.424}, {0.070, 0.246, 0.520},
                     {0.121, 0.406, 0.471}, {0.117, 0.210, 0.353},
                     {0.035, 0.319, 0.328}, {0.135, 0.221, 0.394},
                     {0.037, 0.117, 0.137}, {0.037, 0.158, 0.163},
                     {0.179, 0.268, 0.320}, {0.063, 0.087, 0.150},
                     {0.072, 0.110, 0.122}, {0.028, 0.205, 0.210},
                     {0.256, 0.309, 0.327}, {0.082, 0.201, 0.204},
                     {0.221, 0.320, 0.366}, {0.114, 0.226, 0.259},
                     {0.143, 0.225, 0.266}, {0.064, 0.090, 0.136},
                     {0.092, 0.225, 0.246}, {0.040, 0.169, 0.251},
                     {0.127, 0.247, 0.349}, {0.117, 0.125, 0.192},
                     {0.016, 0.181, 0.192}},
                    device)
                    .Reshape({-1});

    std::tie(indices, distances) =
            index.SearchKnn(query_points, queries_row_splits, 3);
    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances, 1e-5, 1e-3));
}

}  // namespace tests
}  // namespace open3d
