// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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
    core::Tensor ref(points, {size, 3}, core::Dtype::Float32, device);
    float radius = 0.1;
    core::nns::FixedRadiusIndex index(ref, radius);

    core::Tensor query(std::vector<float>({0.064705, 0.043921, 0.087843}),
                       {1, 3}, core::Dtype::Float32, device);

    // if radius <= 0
    // EXPECT_THROW(index.SearchRadius(query, -1.0), std::runtime_error);
    // EXPECT_THROW(index.SearchRadius(query, 0.0), std::runtime_error);

    // if radius == 0.1
    std::tuple<core::Tensor, core::Tensor, core::Tensor> result =
            index.SearchRadius(query, radius);
    core::Tensor indices = std::get<0>(result).To(core::Dtype::Int32);
    core::Tensor distances = std::get<1>(result);
    ExpectEQ(indices.ToFlatVector<int>(), std::vector<int>({1, 4}));
    ExpectEQ(distances.ToFlatVector<float>(),
             std::vector<float>({0.00626358, 0.00747938}));
}

}  // namespace tests
}  // namespace open3d
