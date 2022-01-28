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

#include <cmath>
#include <limits>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorFormatterPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         TensorFormatterPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TensorFormatterPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorFormatterPermuteDevicePairs,
        testing::ValuesIn(TensorFormatterPermuteDevicePairs::TestCases()));

class TensorFormatterPermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<core::SizeVector, core::SizeVector>,
                         core::Device>> {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorFormatterPermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

/// Convert to const reference.
/// https://stackoverflow.com/a/15519125/1255535
template <typename T>
static constexpr const T &AsConst(T &t) noexcept {
    return t;
}

TEST_P(TensorFormatterPermuteDevices, FormatTensor) {
    core::Device device = GetParam();
    core::Tensor t;

    // 0D
    t = core::Tensor::Ones({}, core::Float32, device);
    EXPECT_EQ(t.ToString(), R"(1.0)");
    t = core::Tensor::Full({}, std::numeric_limits<float>::quiet_NaN(),
                           core::Float32, device);
    EXPECT_EQ(t.ToString(), R"(nan)");
    t = core::Tensor::Full({}, std::numeric_limits<double>::quiet_NaN(),
                           core::Float32, device);  // Casting
    EXPECT_EQ(t.ToString(), R"(nan)");

    // 1D float
    t = core::Tensor(std::vector<float>{0, 1, 2, 3, 4}, {5}, core::Float32,
                     device);
    EXPECT_EQ(t.ToString(), R"([0.0 1.0 2.0 3.0 4.0])");

    // 1D int
    std::vector<int32_t> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    t = core::Tensor(vals, {24}, core::Int32, device);
    EXPECT_EQ(
            t.ToString(),
            R"([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23])");

    // 2D
    t = core::Tensor(vals, {6, 4}, core::Int32, device);
    EXPECT_EQ(t.ToString(),
              R"([[0 1 2 3],
 [4 5 6 7],
 [8 9 10 11],
 [12 13 14 15],
 [16 17 18 19],
 [20 21 22 23]])");

    // 3D
    t = core::Tensor(vals, {2, 3, 4}, core::Int32, device);
    EXPECT_EQ(t.ToString(),
              R"([[[0 1 2 3],
  [4 5 6 7],
  [8 9 10 11]],
 [[12 13 14 15],
  [16 17 18 19],
  [20 21 22 23]]])");

    // 4D
    t = core::Tensor(vals, {2, 3, 2, 2}, core::Int32, device);
    EXPECT_EQ(t.ToString(),
              R"([[[[0 1],
   [2 3]],
  [[4 5],
   [6 7]],
  [[8 9],
   [10 11]]],
 [[[12 13],
   [14 15]],
  [[16 17],
   [18 19]],
  [[20 21],
   [22 23]]]])");

    // Boolean
    t = core::Tensor(std::vector<bool>{true, false, true, true, false, false},
                     {2, 3}, core::Bool, device);
    EXPECT_EQ(t.ToString(),
              R"([[True False True],
 [True False False]])");
}

}  // namespace tests
}  // namespace open3d
