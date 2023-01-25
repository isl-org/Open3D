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

#include "open3d/core/TensorFunction.h"

#include "open3d/utility/Helper.h"
#include "open3d/utility/Timer.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorFunctionPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         TensorFunctionPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorFunctionPermuteDevices, Concatenate) {
    core::Device device = GetParam();

    core::Tensor a, b, c, output_tensor;

    // 0-D cannot be concatenated.
    a = core::Tensor::Init<float>(0, device);
    b = core::Tensor::Init<float>(1, device);
    c = core::Tensor::Init<float>(1, device);
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, 0));
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, -1));

    // Same Shape.
    // Concatenating 1-D tensors.
    a = core::Tensor::Init<float>({0, 1, 2}, device);
    b = core::Tensor::Init<float>({3, 4}, device);
    c = core::Tensor::Init<float>({5, 6, 7}, device);

    // 1-D can be concatenated along axis = 0, -1.
    // Default axis is 0.
    output_tensor = core::Concatenate({a, b, c});
    EXPECT_TRUE(output_tensor.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6, 7}, device)));

    output_tensor = core::Concatenate({a, b, c}, -1);
    EXPECT_TRUE(output_tensor.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6, 7}, device)));

    // 1-D can be concatenated along axis = 1, -2.
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, 1));
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, -2));

    // Concatenating 2-D tensors.
    a = core::Tensor::Init<float>({{0, 1}, {2, 3}}, device);
    b = core::Tensor::Init<float>({{4, 5}}, device);
    c = core::Tensor::Init<float>({{6, 7}}, device);

    // Above tensors can be concatenated along axis = 0, -2.
    output_tensor = core::Concatenate({a, b, c}, 0);
    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));
    output_tensor = core::Concatenate({a, b, c}, -2);
    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));

    // Above 2-D tensors cannot be appended to 2-D along axis = 1, -1.
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, 1));
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, -1));

    // Concatenating 2-D tensors of shape {3, 1}.
    a = core::Tensor::Init<float>({{0}, {1}, {2}}, device);
    b = core::Tensor::Init<float>({{3}, {4}, {5}}, device);
    c = core::Tensor::Init<float>({{6}, {7}, {8}}, device);

    // Above tensors can be concatenated along axis = 0, 1, -1, -2.
    output_tensor = core::Concatenate({a, b, c}, 0);
    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}, device)));
    output_tensor = core::Concatenate({a, b, c}, -2);
    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}, device)));

    output_tensor = core::Concatenate({a, b, c}, 1);

    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0, 3, 6}, {1, 4, 7}, {2, 5, 8}}, device)));
    output_tensor = core::Concatenate({a, b, c}, -1);
    EXPECT_TRUE(output_tensor.AllClose(core::Tensor::Init<float>(
            {{0, 3, 6}, {1, 4, 7}, {2, 5, 8}}, device)));

    // 2-D can not be concatenated along axis = 2, -3.
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, 2));
    EXPECT_ANY_THROW(core::Concatenate({a, b, c}, -3));

    // Using Concatenate for a single tensor. The tensor is split along its
    // first dimension, and concatenated along the axis.
    a = core::Tensor::Init<float>(
            {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}}, device);
    EXPECT_TRUE(core::Concatenate({a}, 1).AllClose(core::Tensor::Init<float>(
            {{0, 1, 4, 5, 8, 9}, {2, 3, 6, 7, 10, 11}}, device)));

    // Dtype and Device of both the tensors must be same.
    // Taking the above case of [1, 2] to [2, 2] with different dtype and
    // device.
    EXPECT_ANY_THROW(core::Concatenate({a, b.To(core::Float64), c}));
    if (device.IsCUDA()) {
        EXPECT_ANY_THROW(
                core::Concatenate({a, b.To(core::Device("CPU:0")), c}));
    }
}

TEST_P(TensorFunctionPermuteDevices, Append) {
    core::Device device = GetParam();

    core::Tensor self, other, output;

    // Appending 0-D to 0-D.
    self = core::Tensor::Init<float>(0, device);
    other = core::Tensor::Init<float>(1, device);

    // 0-D can be appended to 0-D along axis = null.
    output = core::Append(self, other);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>({0, 1}, device)));

    // 0-D can not be appended to 0-D along axis = 0, -1.
    EXPECT_ANY_THROW(core::Append(self, other, 0));
    EXPECT_ANY_THROW(core::Append(self, other, -1));

    // Same Shape.
    // Appending 1-D [3,] self to 1-D [4,].
    self = core::Tensor::Init<float>({0, 1, 2, 3}, device);
    other = core::Tensor::Init<float>({4, 5, 6}, device);

    // 1-D can be appended to 1-D along axis = null, 0, -1.
    output = core::Append(self, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    output = core::Append(self, other, 0);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    output = core::Append(self, other, -1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    // 1-D can not be appended to 1-D along axis = 1, -2.
    EXPECT_ANY_THROW(core::Append(self, other, 1));
    EXPECT_ANY_THROW(core::Append(self, other, -2));

    // Appending 2-D [2, 2] self to 2-D [2, 2].
    self = core::Tensor::Init<float>({{0, 1}, {2, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}, {6, 7}}, device);

    // 2-D self can be appended to 2-D self along axis = null, 0, 1, -1, -2.
    output = core::Append(self, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6, 7}, device)));

    output = core::Append(self, other, 0);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));

    output = core::Append(self, other, -2);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));

    output = core::Append(self, other, 1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1, 4, 5}, {2, 3, 6, 7}}, device)));

    output = core::Append(self, other, -1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1, 4, 5}, {2, 3, 6, 7}}, device)));

    // 2-D can not be appended to 2-D along axis = 2, -3.
    EXPECT_ANY_THROW(core::Append(self, other, 2));
    EXPECT_ANY_THROW(core::Append(self, other, -3));

    // Appending 2-D [1, 2] self to 2-D [2, 2].
    self = core::Tensor::Init<float>({{0, 1}, {2, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}}, device);

    // Only the dimension along the axis can be different, so self of shape
    // [1, 2] can be appended to [2, 2] along axis = null, 0, -2.
    output = core::Append(self, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5}, device)));

    output = core::Append(self, other, 0);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1}, {2, 3}, {4, 5}}, device)));

    output = core::Append(self, other, -2);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1}, {2, 3}, {4, 5}}, device)));

    // [1, 2] can not be appended to [2, 2] along axis = 1, -1.
    EXPECT_ANY_THROW(core::Append(self, other, 1));
    EXPECT_ANY_THROW(core::Append(self, other, -1));

    // Dtype and Device of both the tensors must be same.
    // Taking the above case of [1, 2] to [2, 2] with different dtype and
    // device.
    EXPECT_ANY_THROW(core::Append(self, other.To(core::Float64)));
    if (device.IsCUDA()) {
        EXPECT_ANY_THROW(core::Append(self, other.To(core::Device("CPU:0"))));
    }

    // output = core::Append(self, other);
    // is same as:
    // output = self.Append(other);
    EXPECT_TRUE(core::Append(self, other).AllClose(self.Append(other)));
}

TEST_P(TensorFunctionPermuteDevices, Maximum) {
    core::Device device = GetParam();

    core::Tensor input, other, output;

    // 0-D Tensor.
    input = core::Tensor::Init<float>(0, device);
    other = core::Tensor::Init<float>(1, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(output.AllClose(other));

    // 1-D Tensor.
    input = core::Tensor::Init<float>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<float>({4, 5, 2, 7}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(
            output.AllClose(core::Tensor::Init<float>({4, 5, 4, 7}, device)));

    // 2-D Tensor.
    input = core::Tensor::Init<float>({{2, 1}, {4, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}, {2, 7}}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{4, 5}, {4, 7}}, device)));

    // Special case: Different input shape.
    input = core::Tensor::Init<float>({2, 4, 3}, device);
    other = core::Tensor::Init<float>({{4, 5, 2}, {3, 5, 1}}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{4, 5, 3}, {3, 5, 3}}, device)));

    // 1-D Tensor with int32 dtype.
    input = core::Tensor::Init<int>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<int>({4, 5, 2, 7}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<int>({4, 5, 4, 7}, device)));

    // 1-D Tensor with uint8 dtype.
    input = core::Tensor::Init<uint8_t>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<uint8_t>({4, 5, 2, 7}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(
            output.AllClose(core::Tensor::Init<uint8_t>({4, 5, 4, 7}, device)));

    // 1-D Tensor with bool dtype.
    input = core::Tensor::Init<bool>({true, true, false, false}, device);
    other = core::Tensor::Init<bool>({false, false, true, true}, device);
    output = core::Maximum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<bool>({true, true, true, true}, device)));
}

TEST_P(TensorFunctionPermuteDevices, Minimum) {
    core::Device device = GetParam();

    core::Tensor input, other, output;

    // 0-D Tensor.
    input = core::Tensor::Init<float>(0, device);
    other = core::Tensor::Init<float>(1, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(output.AllClose(input));

    // 1-D Tensor.
    input = core::Tensor::Init<float>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<float>({4, 5, 2, 7}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(
            output.AllClose(core::Tensor::Init<float>({2, 1, 2, 3}, device)));

    // 2-D Tensor.
    input = core::Tensor::Init<float>({{2, 1}, {4, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}, {2, 7}}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{2, 1}, {2, 3}}, device)));

    // Special case: Different input shape.
    input = core::Tensor::Init<float>({2, 4, 3}, device);
    other = core::Tensor::Init<float>({{4, 5, 2}, {3, 5, 1}}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{2, 4, 2}, {2, 4, 1}}, device)));

    // 1-D Tensor with int32 dtype.
    input = core::Tensor::Init<int>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<int>({4, 5, 2, 7}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<int>({2, 1, 2, 3}, device)));

    // 1-D Tensor with uint8 dtype.
    input = core::Tensor::Init<uint8_t>({2, 1, 4, 3}, device);
    other = core::Tensor::Init<uint8_t>({4, 5, 2, 7}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(
            output.AllClose(core::Tensor::Init<uint8_t>({2, 1, 2, 3}, device)));

    // 1-D Tensor with bool dtype.
    input = core::Tensor::Init<bool>({true, true, false, false}, device);
    other = core::Tensor::Init<bool>({false, false, true, true}, device);
    output = core::Minimum(input, other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<bool>({false, false, false, false}, device)));
}

}  // namespace tests
}  // namespace open3d
