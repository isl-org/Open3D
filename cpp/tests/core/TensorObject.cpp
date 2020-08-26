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

#include <cmath>
#include <limits>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/utility/Helper.h"
#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorObjectPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorObject,
                         TensorObjectPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TensorObjectPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TensorObject,
        TensorObjectPermuteDevicePairs,
        testing::ValuesIn(TensorObjectPermuteDevicePairs::TestCases()));

class TensorObjectPermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<core::SizeVector, core::SizeVector>,
                         core::Device>> {};
INSTANTIATE_TEST_SUITE_P(
        TensorObject,
        TensorObjectPermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

class TestObject {
public:
    TestObject() = default;
    TestObject(int val) : val_(val) {}

    bool operator==(const TestObject &other) const {
        return val_ == other.val_;
    }

private:
    int val_;
};

TEST_P(TensorObjectPermuteDevices, Constructor) {
    core::Device device = GetParam();
    int64_t byte_size = sizeof(TestObject);
    core::Dtype dtype = core::Dtype(core::Dtype::DtypeCode::Object, byte_size,
                                    "TestObjectDtype");

    for (const core::SizeVector &shape : std::vector<core::SizeVector>{
                 {}, {0}, {0, 0}, {0, 1}, {1, 0}, {2, 3}}) {
        core::Tensor t(shape, dtype, device);
        EXPECT_EQ(t.GetShape(), shape);
        EXPECT_EQ(t.GetDtype(), dtype);
        EXPECT_EQ(t.GetDtype().ToString(), "TestObjectDtype");
        EXPECT_EQ(t.GetDtype().ByteSize(), byte_size);
        EXPECT_EQ(t.GetDevice(), device);
    }

    EXPECT_ANY_THROW(core::Tensor({-1}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({0, -2}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({-1, -1}, dtype, device));
}

}  // namespace tests
}  // namespace open3d
