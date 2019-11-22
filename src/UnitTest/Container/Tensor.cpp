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
#include "Open3D/Container/Indexing.h"
#include "Open3D/Container/Kernel/Kernel.h"
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/SizeVector.h"

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

TEST_P(TensorPermuteDevices, Fill) {
    Device device = GetParam();
    Tensor t(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    t.Slice(1, 0, 3, 2).Fill(1);  // t[:, 0:3:2].fill(1)
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 0, 1, 1, 0, 1}));
}

TEST_P(TensorPermuteDevices, FillSlice) {
    Device device = GetParam();
    Tensor t(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    t.Fill(1);
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorPermuteDevices, FillFancy) {
    Device device = GetParam();
    Tensor t(std::vector<float>(2 * 3 * 4, 0), {2, 3, 4}, Dtype::Float32,
             device);
    Tensor v(std::vector<float>({1}), SizeVector({}), Dtype::Float32, device);

    // t[:, [1, 2], [1, 2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device)};

    t.IndexSet(indices, v);  // We cannot use T.Fill() here
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}));
}

TEST_P(TensorPermuteDevicePairs, Copy) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    Dtype dtype(Dtype::Float32);
    SizeVector shape{2, 3};

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor src_t(vals, shape, dtype, src_device);

    Tensor dst_t = src_t.Copy(dst_device);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetBlob()->byte_size_,
              shape.NumElements() * DtypeUtil::ByteSize(dtype));
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
}

TEST_P(TensorPermuteDevicePairs, CopyBroadcast) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    Dtype dtype(Dtype::Float32);

    // Broadcast {2, 1, 3} to {2, 2, 2, 3}
    SizeVector src_shape{2, 1, 3};
    SizeVector dst_shape{2, 2, 2, 3};

    std::vector<float> src_vals{0, 1, 2, 3, 4, 5};
    std::vector<float> dst_vals{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
    Tensor src_t(src_vals, src_shape, dtype, src_device);
    Tensor dst_t(dst_shape, dtype, dst_device);
    dst_t.CopyFrom(src_t);  // Equivalently, dst_t = src_t

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<float>(), dst_vals);
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

TEST_P(TensorPermuteDevices, ItemAssign) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // Assigning to rvalue
    float new_val_0 = 100.f;
    t[1][2][3] = new_val_0;
    EXPECT_EQ(t[1][2][3].Item<float>(), 100);

    // Assigning to rvalue, with implicit casting (uint8_t -> float)
    uint8_t new_val_1 = 101;
    t[1][2][3] = new_val_1;
    EXPECT_EQ(t[1][2][3].Item<float>(), 101);
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

TEST_P(TensorPermuteDevices, SliceAssign) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor dst(vals, {2, 3, 4}, Dtype::Float32, device);

    // Assigning a contiguous Tensor to lvalue
    // src_0 == [[120, 140], [200, 220]]
    Tensor src_0(std::vector<float>({120, 140, 200, 220}), {2, 2},
                 Dtype::Float32, device);
    Tensor dst_slice = dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    dst_slice = src_0;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 120, 13, 140, 15,
                                  16, 17, 18, 19, 200, 21, 220, 23}));

    // Assigning a contiguous Tensor to rvalue
    // src_1 == [[121, 141], [201, 221]]
    Tensor src_1(std::vector<float>({121, 141, 201, 221}), {2, 2},
                 Dtype::Float32, device);
    dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2) = src_1;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 121, 13, 141, 15,
                                  16, 17, 18, 19, 201, 21, 221, 23}));

    // Assigning a non-contiguous Tensor to lvalue
    // src_2 == [[122, 142], [202, 222]]
    Tensor src_2_tmp(std::vector<float>({122, 142, -1, -1, 202, 222}), {3, 2},
                     Dtype::Float32, device);    // Shape (3, 2)
    Tensor src_2 = src_2_tmp.Slice(0, 0, 3, 2);  // Shape (2, 2)
    dst_slice = src_2;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 122, 13, 142, 15,
                                  16, 17, 18, 19, 202, 21, 222, 23}));

    // Assigning a non-contiguous Tensor to rvalue
    // src_3 == [[123, 143], [203, 223]]
    Tensor src_3_tmp(std::vector<float>({123, 143, -1, -1, 203, 223}), {3, 2},
                     Dtype::Float32, device);    // Shape (3, 2)
    Tensor src_3 = src_3_tmp.Slice(0, 0, 3, 2);  // Shape (2, 2)
    dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2) = src_3;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 123, 13, 143, 15,
                                  16, 17, 18, 19, 203, 21, 223, 23}));
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

TEST_P(TensorPermuteDevices, IndexGet) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1, 2], [1, 2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device)};

    Tensor t_1 = t.IndexGet(indices);
    EXPECT_TRUE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevices, IndexGetNegative) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1, -1], [1, -2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, -1}), {2}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, -2}), {2}, Dtype::Int64, device)};

    Tensor t_1 = t.IndexGet(indices);
    EXPECT_TRUE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevices, IndexGetBroadcast) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1, 2], [1, 2]] to shape {2, 2}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device)};

    // Broadcast to shape {3, 2, 2}
    SizeVector dst_shape{3, 2, 2};
    Tensor dst_t(dst_shape, Dtype::Float32, device);
    dst_t = src_t.IndexGet(indices);  // Intermediate tensor copied internally

    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({3, 2, 2}));
    EXPECT_EQ(
            dst_t.ToFlatVector<float>(),
            std::vector<float>({5, 10, 17, 22, 5, 10, 17, 22, 5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevices, IndexGetActualBroadcast) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1, 2], [1, 2]] to shape {2, 2}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device)};

    // Broadcast to shape {3, 2, 2}
    SizeVector dst_shape{3, 2, 2};
    Tensor dst_t(dst_shape, Dtype::Float32, device);

    // Expose the interal step to remove copy
    std::vector<Tensor> full_index_tensors;
    SizeVector indexed_out_shape;
    std::tie(full_index_tensors, indexed_out_shape) =
            PreprocessIndexTensors(src_t, indices);
    kernel::IndexedGet(src_t, dst_t, full_index_tensors, indexed_out_shape);

    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({3, 2, 2}));
    EXPECT_EQ(
            dst_t.ToFlatVector<float>(),
            std::vector<float>({5, 10, 17, 22, 5, 10, 17, 22, 5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevices, DISABLED_IndexGetSeparateBySlice) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[[0, 1], :, [0, 1]]
    std::vector<Tensor> indices = {
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, device),
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, device)};

    Tensor t_fancy = t.IndexGet(indices);
    EXPECT_EQ(t_fancy.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(t_fancy.ToFlatVector<float>(),
              std::vector<float>({0, 4, 8, 13, 17, 21}));
}

TEST_P(TensorPermuteDevices, IndexSet) {
    Device device = GetParam();

    std::vector<float> vals({4, 6, 5, 16, 18, 17});
    Tensor rhs(vals, {2, 3}, Dtype::Float32, device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    Tensor t(zeros, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1], [0, 2, 1]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1}), {1}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({0, 2, 1}), {3}, Dtype::Int64, device)};

    t.IndexSet(indices, rhs);
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 4,  5,  6,  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 16, 17, 18, 0, 0, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, IndexSetBroadcast) {
    Device device = GetParam();

    std::vector<float> src_vals({10, 20});
    Tensor src_t(src_vals, {2, 1}, Dtype::Float32, device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    Tensor dst_t(zeros, {2, 3, 4}, Dtype::Float32, device);

    // t[:, [1], [0, 2, 1]] -> slice {2, 3, 4} to {2, 3}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, device),
            Tensor(std::vector<int64_t>({1}), {1}, Dtype::Int64, device),
            Tensor(std::vector<int64_t>({0, 2, 1}), {3}, Dtype::Int64, device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0}));
}
