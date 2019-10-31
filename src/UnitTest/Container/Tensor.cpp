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

#include "Open3D/Container/Tensor.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

#include "Container/ContainerTest.h"
#include "TestUtility/UnitTest.h"

#include <vector>

using namespace std;
using namespace open3d;

class TensorPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         TensorPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TensorPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorPermuteDevicePairs,
        testing::ValuesIn(TensorPermuteDevicePairs::TestCases()));

class TensorPermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<SizeVector, SizeVector>, Device>> {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorPermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

TEST_P(TensorPermuteDevices, Constructor) {
    Device device = GetParam();

    SizeVector shape{2, 3};
    Dtype dtype = Dtype::Float32;
    Tensor t(shape, dtype, device);

    EXPECT_EQ(t.GetShape(), shape);
    EXPECT_EQ(t.GetBlob()->byte_size_, 4 * 2 * 3);
    EXPECT_EQ(t.GetBlob()->device_, device);
}

TEST_P(TensorPermuteDevices, WithInitValue) {
    Device device = GetParam();

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor t(vals, {2, 3}, Dtype::Float32, device);
}

TEST_P(TensorPermuteDevices, WithInitValueTypeMismatch) {
    Device device = GetParam();

    std::vector<int> vals{0, 1, 2, 3, 4, 5};
    EXPECT_THROW(Tensor(vals, {2, 3}, Dtype::Float32, device),
                 std::runtime_error);
}

TEST_P(TensorPermuteDevices, WithInitValueSizeMismatch) {
    Device device = GetParam();

    std::vector<float> vals{0, 1, 2, 3, 4};
    EXPECT_THROW(Tensor(vals, {2, 3}, Dtype::Float32, device),
                 std::runtime_error);
}

TEST_P(TensorPermuteDevicePairs, Copy) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    SizeVector shape{2, 3};
    Dtype dtype(Dtype::Float32);

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor src_t(vals, shape, dtype, src_device);

    Tensor dst_t = src_t.Copy(dst_device);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetBlob()->byte_size_,
              shape.NumElements() * DtypeUtil::ByteSize(dtype));
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
}

TEST_P(TensorPermuteDevices, DefaultStrides) {
    Device device = GetParam();

    Tensor t0({}, Dtype::Float32, device);
    EXPECT_EQ(t0.GetShape(), SizeVector{});
    EXPECT_EQ(t0.GetStrides(), SizeVector{});
}

TEST_P(TensorPermuteSizesDefaultStridesAndDevices, DefaultStrides) {
    SizeVector shape;
    SizeVector expected_strides;
    std::tie(shape, expected_strides) = std::get<0>(GetParam());

    Device device = std::get<1>(GetParam());
    Tensor t(shape, Dtype::Float32, device);
    EXPECT_EQ(t.GetStrides(), expected_strides);
}

TEST_P(TensorPermuteDevices, OperatorSquareBrackets) {
    Device device = GetParam();

    // Zero dim
    EXPECT_THROW(Tensor({}, Dtype::Float32)[0], std::runtime_error);

    // Index out-of-bounds
    EXPECT_THROW(Tensor({0, 1}, Dtype::Float32)[0], std::runtime_error);
    EXPECT_THROW(Tensor({1, 2}, Dtype::Float32)[10], std::runtime_error);

    // Regular cases
    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    Tensor t_0 = t[0];
    EXPECT_EQ(t_0.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_0.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_0.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_0.GetBlob(), t.GetBlob());

    Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(), static_cast<uint8_t *>(t.GetDataPtr()) +
                                        1 * 3 * 4 * sizeof(float));
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());

    Tensor t_1_2 = t[1][2];
    EXPECT_EQ(t_1_2.GetShape(), SizeVector({4}));
    EXPECT_EQ(t_1_2.GetStrides(), SizeVector({1}));
    EXPECT_EQ(t_1_2.GetDataPtr(), static_cast<uint8_t *>(t.GetDataPtr()) +
                                          (1 * 3 * 4 + 2 * 4) * sizeof(float));
    EXPECT_EQ(t_1_2.GetBlob(), t.GetBlob());

    Tensor t_1_2_3 = t[1][2][3];
    EXPECT_EQ(t_1_2_3.GetShape(), SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetStrides(), SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetDataPtr(),
              static_cast<uint8_t *>(t.GetDataPtr()) +
                      (1 * 3 * 4 + 2 * 4 + 3) * sizeof(float));
    EXPECT_EQ(t_1_2_3.GetBlob(), t.GetBlob());
}

TEST_P(TensorPermuteDevices, Item) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);

    Tensor t_1 = t[1];
    EXPECT_THROW(t_1.Item<float>(), std::runtime_error);

    Tensor t_1_2 = t[1][2];
    EXPECT_THROW(t_1_2.Item<float>(), std::runtime_error);

    Tensor t_1_2_3 = t[1][2][3];
    EXPECT_THROW(t_1_2_3.Item<int32_t>(), std::runtime_error);
    EXPECT_EQ(t_1_2_3.Item<float>(), 23.f);
}

TEST_P(TensorPermuteDevices, ToString) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t1(vals, {24}, Dtype::Float32, device);
    EXPECT_EQ(
            t1.ToString(/*with_suffix=*/false),
            R"([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23])");

    Tensor t2(vals, {6, 4}, Dtype::Float32, device);
    EXPECT_EQ(t2.ToString(/*with_suffix=*/false),
              R"([[0 1 2 3],
 [4 5 6 7],
 [8 9 10 11],
 [12 13 14 15],
 [16 17 18 19],
 [20 21 22 23]])");

    Tensor t3(vals, {2, 3, 4}, Dtype::Float32, device);
    EXPECT_EQ(t3.ToString(/*with_suffix=*/false),
              R"([[[0 1 2 3],
  [4 5 6 7],
  [8 9 10 11]],
 [[12 13 14 15],
  [16 17 18 19],
  [20 21 22 23]]])");

    Tensor t4(vals, {2, 3, 2, 2}, Dtype::Float32, device);
    EXPECT_EQ(t4.ToString(/*with_suffix=*/false),
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

    // utility::LogDebug("\n{}", t1.ToString());
    // utility::LogDebug("\n{}", t3.ToString());
    // utility::LogDebug("\n{}", t2.ToString());
    // utility::LogDebug("\n{}", t4.ToString());
}

TEST_P(TensorPermuteDevices, CopyContiguous) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);
    EXPECT_TRUE(t.IsContiguous());

    Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);
    EXPECT_TRUE(t_0.IsContiguous());

    Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(), static_cast<uint8_t *>(t.GetDataPtr()) +
                                        1 * 3 * 4 * sizeof(float));
    EXPECT_NE(t_1.GetDataPtr(), t_1.GetBlob()->v_);
    EXPECT_TRUE(t_1.IsContiguous());

    Tensor t_1_copy = t_1.Copy(device);
    EXPECT_EQ(t_1_copy.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1_copy.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1_copy.GetDataPtr(),
              t_1_copy.GetBlob()->v_);  // Points to beginning of Blob
}

TEST_P(TensorPermuteDevices, Slice) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);
    const void *blob_head = t.GetBlob()->v_;
    EXPECT_EQ(t.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(t.GetStrides(), SizeVector({12, 4, 1}));
    EXPECT_EQ(t.GetDataPtr(), blob_head);

    // t_1 = t[0:2:1], effectively not sliced
    Tensor t_1 = t.Slice(0, 0, 2, 1);
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({12, 4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(), blob_head);

    // t_2 = t[0:2:1][:, 0:3:2, :]
    Tensor t_2 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2);
    EXPECT_EQ(t_2.GetShape(), SizeVector({2, 2, 4}));
    EXPECT_EQ(t_2.GetStrides(), SizeVector({12, 8, 1}));
    EXPECT_EQ(t_2.GetDataPtr(), blob_head);

    // t_3 = [0:2:1, 0:3:2, 0:4:2]
    Tensor t_3 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_EQ(t_3.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_3.GetStrides(), SizeVector({12, 8, 2}));
    EXPECT_EQ(t_3.GetDataPtr(), blob_head);

    // t_4 = t[1, 0:3:2, 0:4:2], a mix of [] and slice
    Tensor t_4 = t[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    EXPECT_EQ(t_4.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_4.GetStrides(), SizeVector({8, 2}));
    EXPECT_EQ(t_4.GetDataPtr(),
              static_cast<const uint8_t *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
}

TEST_P(TensorPermuteDevices, CopyNonContiguous) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[0:2:1, 0:3:2, 0:4:2]
    Tensor t_1 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_FALSE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({12, 8, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // Copy ensures contiguous
    Tensor t_1_copy = t_1.Copy(device);
    EXPECT_TRUE(t_1_copy.IsContiguous());
    EXPECT_EQ(t_1_copy.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_1_copy.GetStrides(), SizeVector({4, 2, 1}));
    EXPECT_EQ(t_1_copy.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // Clone replicates the exact syntax
    Tensor t_1_clone = t_1.Clone(device);
    EXPECT_FALSE(t_1_clone.IsContiguous());
    EXPECT_EQ(t_1_clone.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_1_clone.GetStrides(), SizeVector({12, 8, 2}));
    EXPECT_EQ(t_1_clone.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));
}
