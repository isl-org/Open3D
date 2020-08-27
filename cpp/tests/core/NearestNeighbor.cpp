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

#include "open3d/core/nns/NearestNeighbor.h"

#include <cmath>
#include <limits>

#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(NearestNeighbor, KnnSearch) {
    // set up index
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighbor index(ref);
    index.KnnIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    // if k <= 0
    EXPECT_THROW(index.KnnSearch(query, -1), std::runtime_error);
    EXPECT_THROW(index.KnnSearch(query, 0), std::runtime_error);

    // if k == 3
    std::pair<core::Tensor, core::Tensor> result_3 = index.KnnSearch(query, 3);
    core::Tensor indices = result_3.first.To(core::Dtype::Int32);
    core::Tensor distances = result_3.second;
    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4, 9}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912}));

    // if k > size
    std::pair<core::Tensor, core::Tensor> result_12 =
            index.KnnSearch(query, 12);

    core::Tensor indices_12 = result_12.first.To(core::Dtype::Int32);
    core::Tensor distances_12 = result_12.second;
    ExpectEQ(indices_12.ToFlatVector<int>(),
             std::vector<int>({1, 4, 9, 0, 3, 2, 5, 7, 6, 8}));
    ExpectEQ(distances_12.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938, 0.0108912, 0.0138322,
                                  0.015048, 0.018695, 0.0199108, 0.0286952,
                                  0.0362638, 0.0411266}));

    // float
    core::Tensor ref_float = ref.To(core::Dtype::Float32);
    core::Tensor query_float = query.To(core::Dtype::Float32);
    core::nns::NearestNeighbor index_float(ref_float);
    index_float.KnnIndex();

    // throw if query and reference have different data type
    EXPECT_THROW(index_float.KnnSearch(query, 3), std::runtime_error);

    std::pair<core::Tensor, core::Tensor> result_float =
            index_float.KnnSearch(query_float, 3);
    core::Tensor indices_float = result_float.first.To(core::Dtype::Int32);
    core::Tensor distances_float = result_float.second;
    ExpectEQ(indices_float.ToFlatVector<int>(), std::vector<int>({1, 4, 9}));
    ExpectEQ(distances_float.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938, 0.0108912}));
}

TEST(NearestNeighbor, FixedRadiusSearch) {
    // set up index
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighbor index(ref);
    index.FixedRadiusIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    // if radius <= 0
    EXPECT_THROW(index.FixedRadiusSearch(query, -1.0), std::runtime_error);
    EXPECT_THROW(index.FixedRadiusSearch(query, 0.0), std::runtime_error);

    // if radius == 0.1
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.FixedRadiusSearch(query, 0.1);
    core::Tensor indices = std::get<0>(result).To(core::Dtype::Int32);
    core::Tensor distances = std::get<1>(result);

    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>({0.00626358, 0.00747938}));

    // float
    core::Tensor ref_float = ref.To(core::Dtype::Float32);
    core::Tensor query_float = query.To(core::Dtype::Float32);
    core::nns::NearestNeighbor index_float(ref_float);
    index_float.FixedRadiusIndex();

    std::tuple<core::Tensor, core::Tensor, core::Tensor> result_float =
            index_float.FixedRadiusSearch(query_float, (float)0.1);

    core::Tensor indices_float =
            std::get<0>(result_float).To(core::Dtype::Int32);
    core::Tensor distances_float = std::get<1>(result_float);
    ExpectEQ(indices_float.ToFlatVector<int>(), std::vector<int>({1, 4}));
    ExpectEQ(distances_float.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938}));
}

TEST(NearestNeighbor, RadiusSearch) {
    // set up index
    int size = 10;
    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighbor index(ref);
    index.RadiusIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843,
                                            0.064705, 0.043921, 0.087843}),
                       {2, 3}, core::Dtype::Float64);

    // if radius <= 0
    double radius_0[] = {1.0, 0.0};
    EXPECT_THROW(index.RadiusSearch(query, radius_0), std::runtime_error);
    EXPECT_THROW(index.RadiusSearch(query, radius_0), std::runtime_error);

    // if radius == 0.1
    double radius[] = {0.1, 0.1};
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.RadiusSearch(query, radius);
    core::Tensor indices = std::get<0>(result).To(core::Dtype::Int32);
    core::Tensor distances = std::get<1>(result);

    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4, 1, 4}));
    ExpectEQ(distances.ToFlatVector<double>(),
             std::vector<double>(
                     {0.00626358, 0.00747938, 0.00626358, 0.00747938}));

    // float
    core::Tensor ref_float = ref.To(core::Dtype::Float32);
    core::Tensor query_float = query.To(core::Dtype::Float32);
    core::nns::NearestNeighbor index_float(ref_float);
    index_float.RadiusIndex();

    float radius_0_float[] = {1.0, 0.0};
    float radius_float[] = {0.1, 0.1};

    EXPECT_THROW(index_float.RadiusSearch(query, radius), std::runtime_error);
    EXPECT_THROW(index_float.RadiusSearch(query_float, radius_0_float),
                 std::runtime_error);

    std::tuple<core::Tensor, core::Tensor, core::Tensor> result_float =
            index_float.RadiusSearch(query_float, radius_float);
    core::Tensor indices_float =
            std::get<0>(result_float).To(core::Dtype::Int32);
    core::Tensor distances_float = std::get<1>(result_float);
    ExpectEQ(indices_float.ToFlatVector<int>(), std::vector<int>({1, 4, 1, 4}));
    ExpectEQ(distances_float.ToFlatVector<float>(),
             std::vector<float>(
                     {0.00626358, 0.00747938, 0.00626358, 0.00747938}));
}

TEST(NearestNeighbor, HybridSearch) {
    // set up index
    int size = 10;

    std::vector<double> points{0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                               0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0,
                               0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, 0.1,
                               0.0, 0.2, 0.2, 0.1, 0.0, 0.0};
    core::Tensor ref(points, {size, 3}, core::Dtype::Float64);
    core::nns::NearestNeighbor index(ref);
    index.HybridIndex();

    core::Tensor query(std::vector<double>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float64);

    std::pair<core::Tensor, core::Tensor> result =
            index.HybridSearch(query, 0.1, 1);

    core::Tensor indices = result.first.To(core::Dtype::Int32);
    core::Tensor distainces = result.second;
    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1}));
    ExpectEQ(distainces.ToFlatVector<double>(),
             std::vector<double>({0.00626358}));

    // float
    core::Tensor ref_float = ref.To(core::Dtype::Float32);
    core::Tensor query_float = query.To(core::Dtype::Float32);
    core::nns::NearestNeighbor index_float(ref_float);
    index_float.HybridIndex();

    EXPECT_THROW(index_float.HybridSearch(query, (float)0.1, 1),
                 std::runtime_error);

    std::pair<core::Tensor, core::Tensor> result_float =
            index_float.HybridSearch(query_float, (float)0.1, 1);
    core::Tensor indices_float = result_float.first.To(core::Dtype::Int32);
    core::Tensor distances_float = result_float.second;
    ExpectEQ(indices_float.ToFlatVector<int>(), std::vector<int>({1}));
    ExpectEQ(distances_float.ToFlatVector<float>(),
             std::vector<float>({0.00626358}));
}

}  // namespace tests
}  // namespace open3d
