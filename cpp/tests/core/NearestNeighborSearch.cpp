// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class NNSPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(NearestNeighborSearch,
                         NNSPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class NNSPermuteDevicesWithFaiss : public PermuteDevicesWithFaiss {};
INSTANTIATE_TEST_SUITE_P(
        NearestNeighborSearch,
        NNSPermuteDevicesWithFaiss,
        testing::ValuesIn(PermuteDevicesWithFaiss::TestCases()));

TEST_P(NNSPermuteDevicesWithFaiss, KnnSearch) {
    // Set up nns.
    int size = 10;
    core::Device device = GetParam();

    std::vector<float> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0,
                              0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2,
                              0.0, 0.0, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float32, device);
    core::nns::NearestNeighborSearch nns(ref);
    nns.KnnIndex();

    core::Tensor query(std::vector<float>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float32, device);
    std::pair<core::Tensor, core::Tensor> result;
    core::Tensor indices;
    core::Tensor distances;

    // If k <= 0.
    EXPECT_THROW(nns.KnnSearch(query, -1), std::runtime_error);
    EXPECT_THROW(nns.KnnSearch(query, 0), std::runtime_error);

    // If k == 3.
    result = nns.KnnSearch(query, 3);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938, 0.0108912}));

    // If k > size.result.
    result = nns.KnnSearch(query, 12);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 9, 0, 3, 2, 5, 7, 6, 8}));
    ExpectEQ(distances.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938, 0.0108912, 0.0138322,
                                 0.015048, 0.018695, 0.0199108, 0.0286952,
                                 0.0362638, 0.0411266}));

    // Multiple points.
    query = core::Tensor(std::vector<float>({0.064705, 0.043921, 0.087843,
                                             0.064705, 0.043921, 0.087843}),
                         {2, 3}, core::Dtype::Float32);
    result = nns.KnnSearch(query, 3);
    indices = result.first;
    distances = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 9, 1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938, 0.0108912, 0.00626358,
                                 0.00747938, 0.0108912}));
}

TEST_P(NNSPermuteDevices, FixedRadiusSearch) {
    // Set up nns.
    int size = 10;
    core::Device device = GetParam();
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64, device);
    core::nns::NearestNeighborSearch nns(ref);
    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64, device);

    // If radius <= 0.
    if (device.GetType() == core::Device::DeviceType::CUDA) {
        EXPECT_THROW(nns.FixedRadiusIndex(-1.0), std::runtime_error);
        EXPECT_THROW(nns.FixedRadiusIndex(0.0), std::runtime_error);
    } else {
        nns.FixedRadiusIndex();
        EXPECT_THROW(nns.FixedRadiusSearch(query, -1.0), std::runtime_error);
        EXPECT_THROW(nns.FixedRadiusSearch(query, 0.0), std::runtime_error);
    }

    // If radius == 0.1.
    nns.FixedRadiusIndex(0.1);
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            nns.FixedRadiusSearch(query, 0.1);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    std::vector<int64_t> indices_vec = indices.ToFlatVector<int64_t>();
    std::vector<double> distances_vec = distances.ToFlatVector<double>();
    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938}));
}

TEST(NearestNeighborSearch, MultiRadiusSearch) {
    // Set up nns.
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighborSearch nns(ref);
    nns.MultiRadiusIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843,
                                            0.064705, 0.043921, 0.087843}),
                       {2, 3}, core::Dtype::Float64);
    core::Tensor radius;

    // If radius <= 0.
    radius = core::Tensor(std::vector<double>({1.0, 0.0}), {2},
                          core::Dtype::Float64);
    EXPECT_THROW(nns.MultiRadiusSearch(query, radius), std::runtime_error);
    EXPECT_THROW(nns.MultiRadiusSearch(query, radius), std::runtime_error);

    // If radius == 0.1.
    radius = core::Tensor(std::vector<double>({0.1, 0.1}), {2},
                          core::Dtype::Float64);
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            nns.MultiRadiusSearch(query, radius);
    core::Tensor indices = std::get<0>(result);
    core::Tensor distances = std::get<1>(result);

    ExpectEQ(indices.ToFlatVector<int64_t>(),
             std::vector<int64_t>({1, 4, 1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>(
                     {0.00626358, 0.00747938, 0.00626358, 0.00747938}));
}

TEST_P(NNSPermuteDevicesWithFaiss, HybridSearch) {
    // Set up nns.
    int size = 10;
    core::Device device = GetParam();
    std::vector<float> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0,
                              0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2,
                              0.0, 0.0, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float32, device);
    core::nns::NearestNeighborSearch nns(ref);
    double radius = 0.1;
    int max_knn = 3;
    nns.HybridIndex(radius);

    core::Tensor query(std::vector<float>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float32, device);

    std::pair<core::Tensor, core::Tensor> result =
            nns.HybridSearch(query, radius, max_knn);

    core::Tensor indices = result.first;
    core::Tensor distainces = result.second;
    ExpectEQ(indices.ToFlatVector<int64_t>(), std::vector<int64_t>({1, 4, -1}));
    ExpectEQ(distainces.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938, -1}));
}

}  // namespace tests
}  // namespace open3d
