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

#include "open3d/core/nns/NanoFlannIndex.h"

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

TEST(NanoFlannIndex, SearchKnn) {
    // Define test data.
    core::Device device = core::Device("CPU:0");
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

    // Set up index.
    core::nns::NanoFlannIndex index(dataset_points);

    // if k <= 0.
    EXPECT_THROW(index.SearchKnn(query_points, -1), std::runtime_error);
    EXPECT_THROW(index.SearchKnn(query_points, 0), std::runtime_error);

    // if k == 3
    core::Tensor indices, distances;
    core::SizeVector shape{1, 3};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, 9}}, device);
    gt_distances = core::Tensor::Init<double>(
            {{0.00626358, 0.00747938, 0.0108912}}, device);

    std::tie(indices, distances) = index.SearchKnn(query_points, 3);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));

    // if k > size.
    shape = core::SizeVector{1, 10};
    gt_indices = core::Tensor::Init<int32_t>({{1, 4, 9, 0, 3, 2, 5, 7, 6, 8}},
                                             device);
    gt_distances = core::Tensor::Init<double>(
            {{0.00626358, 0.00747938, 0.0108912, 0.0138322, 0.015048, 0.018695,
              0.0199108, 0.0286952, 0.0362638, 0.0411266}},
            device);
    std::tie(indices, distances) = index.SearchKnn(query_points, 12);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
}

TEST(NanoFlannIndex, SearchRadius) {
    // Define test data.
    core::Device device = core::Device("CPU:0");
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
    core::Tensor gt_indices, gt_distances, gt_neighbors_row_splits;

    // Set up index.
    core::nns::NanoFlannIndex index(dataset_points);

    // if radius <= 0
    EXPECT_THROW(index.SearchRadius(query_points, -1.0), std::runtime_error);
    EXPECT_THROW(index.SearchRadius(query_points, 0.0), std::runtime_error);

    // if radius == 0.1
    core::Tensor indices, distances, neighbors_row_splits;
    core::SizeVector shape{2};
    gt_indices = core::Tensor::Init<int32_t>({1, 4}, device);
    gt_distances = core::Tensor::Init<double>({0.00626358, 0.00747938}, device);
    gt_neighbors_row_splits = core::Tensor::Init<int64_t>({0, 2}, device);
    core::Tensor radii = core::Tensor::Init<double>({0.1});

    std::tie(indices, distances, neighbors_row_splits) =
            index.SearchRadius(query_points, radii, false);

    EXPECT_EQ(indices.GetShape(), shape);
    EXPECT_EQ(distances.GetShape(), shape);
    EXPECT_EQ(neighbors_row_splits.GetShape(), shape);
    EXPECT_TRUE(indices.AllClose(gt_indices));
    EXPECT_TRUE(distances.AllClose(gt_distances));
    EXPECT_TRUE(neighbors_row_splits.AllClose(gt_neighbors_row_splits));
}

}  // namespace tests
}  // namespace open3d
