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

#include "open3d/t/geometry/TensorMap.h"

#include <vector>

#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorMapPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorMap,
                         TensorMapPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorMapPermuteDevices, Constructor) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = GetParam();

    // Empty TensorMap.
    t::geometry::TensorMap tm0("points");
    EXPECT_EQ(tm0.GetPrimaryKey(), "points");
    EXPECT_EQ(tm0.size(), 0);

    // Primary key is required.
    EXPECT_ANY_THROW(t::geometry::TensorMap());

    // Initializer list.
    t::geometry::TensorMap tm1(
            "points", {{"points", core::Tensor::Zeros({10, 3}, dtype, device)},
                       {"colors", core::Tensor::Ones({10, 3}, dtype, device)}});

    // Copy constructor.
    t::geometry::TensorMap tm1_copied(tm1);
    EXPECT_EQ(tm1_copied["points"][0][0].Item<float>(), 0);
    tm1["points"][0][0] = 100;
    EXPECT_EQ(tm1_copied["points"][0][0].Item<float>(), 100);

    // Move constructor.
    t::geometry::TensorMap tm1_moved = std::move(tm1);
    EXPECT_EQ(tm1_moved["points"][0][1].Item<float>(), 0);
    tm1["points"][0][1] = 200;
    EXPECT_EQ(tm1_moved["points"][0][1].Item<float>(), 200);
}

// TEST_P(TensorMap, Assign) {
//     core::Dtype dtype = core::Dtype::Float32;
//     core::Device device = core::Device("CPU:0");

//     t::geometry::TensorMap tm("points");
//     tm["points"] = core::Tensor::Zeros({10, 3}, dtype, device);
//     tm["dummy"] = core::Tensor::Zeros({10, 3}, dtype, device);
//     EXPECT_TRUE(tm.Contains("points"));
//     EXPECT_TRUE(tm.Contains("dummy"));

//     std::unordered_map<std::string, core::Tensor> replacement{
//             {"points", core::Tensor::Ones({2, 3}, dtype, device)},
//             {"colors", core::Tensor::Ones({2, 3}, dtype, device)},
//     };
//     tm.Assign(replacement);
//     EXPECT_TRUE(tm.Contains("points"));
//     EXPECT_TRUE(tm.Contains("colors"));
//     EXPECT_FALSE(tm.Contains("dummy"));

//     // Underlying memory are the same.
//     EXPECT_TRUE(tm["points"].IsSame(replacement["points"]));
//     EXPECT_TRUE(tm["colors"].IsSame(replacement["colors"]));
// }

// TEST_P(TensorMap, IsSizeSynchronized) {
//     core::Dtype dtype = core::Dtype::Float32;
//     core::Device device = core::Device("CPU:0");

//     t::geometry::TensorMap tm(
//             "points",
//             {{"points", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({5, 3}, dtype, device))},
//              {"colors", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({4, 3}, dtype,
//                                 device))}});
//     EXPECT_FALSE(tm.IsSizeSynchronized());

//     tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
//     EXPECT_TRUE(tm.IsSizeSynchronized());
// }

// TEST_P(TensorMap, AssertSizeSynchronized) {
//     core::Dtype dtype = core::Dtype::Float32;
//     core::Device device = core::Device("CPU:0");

//     t::geometry::TensorMap tm(
//             "points",
//             {{"points", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({5, 3}, dtype, device))},
//              {"colors", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({4, 3}, dtype,
//                                 device))}});
//     EXPECT_ANY_THROW(tm.AssertSizeSynchronized());

//     tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
//     EXPECT_NO_THROW(tm.AssertSizeSynchronized());
// }

// TEST_P(TensorMap, Contains) {
//     core::Dtype dtype = core::Dtype::Float32;
//     core::Device device = core::Device("CPU:0");

//     t::geometry::TensorMap tm(
//             "points",
//             {{"points", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({5, 3}, dtype, device))},
//              {"colors", core::TensorList::FromTensor(
//                                 core::Tensor::Ones({4, 3}, dtype,
//                                 device))}});
//     EXPECT_TRUE(tm.Contains("points"));
//     EXPECT_TRUE(tm.Contains("colors"));
//     EXPECT_FALSE(tm.Contains("normals"));
// }

}  // namespace tests
}  // namespace open3d
