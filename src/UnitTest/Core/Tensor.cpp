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

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/Kernel/Kernel.h"
#include "Open3D/Core/MemoryManager.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Helper.h"

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

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
    EXPECT_EQ(t.GetBlob()->GetDevice(), device);
}

TEST_P(TensorPermuteDevices, ConstructorBool) {
    Device device = GetParam();

    SizeVector shape{2, 3};
    Dtype dtype = Dtype::Bool;
    Tensor t(shape, dtype, device);

    EXPECT_EQ(t.GetShape(), shape);
    EXPECT_EQ(t.GetBlob()->GetDevice(), device);
    EXPECT_EQ(t.GetDtype(), dtype);
}

TEST_P(TensorPermuteDevices, WithInitValue) {
    Device device = GetParam();

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor t(vals, {2, 3}, Dtype::Float32, device);
    EXPECT_EQ(t.ToFlatVector<float>(), vals);
}

TEST_P(TensorPermuteDevices, WithInitValueBool) {
    Device device = GetParam();

    std::vector<bool> vals{true, false, true, true, false, false};
    Tensor t(vals, {2, 3}, Dtype::Bool, device);
    EXPECT_EQ(t.ToFlatVector<bool>(), vals);
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
    t.Fill(1);
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorPermuteDevices, FillBool) {
    Device device = GetParam();
    Tensor t(std::vector<bool>(2 * 3, false), {2, 3}, Dtype::Bool, device);
    t.Fill(true);
    EXPECT_EQ(t.ToFlatVector<bool>(), std::vector<bool>(2 * 3, true));
}

TEST_P(TensorPermuteDevices, FillSlice) {
    Device device = GetParam();
    Tensor t(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    t.Slice(1, 0, 3, 2).Fill(1);  // t[:, 0:3:2].fill(1)
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 0, 1, 1, 0, 1}));
}

TEST_P(TensorPermuteDevicePairs, IndexSetFillFancy) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    Tensor dst_t(std::vector<float>(2 * 3 * 4, 0), {2, 3, 4}, Dtype::Float32,
                 dst_device);
    Tensor src_t(std::vector<float>({1}), SizeVector({}), Dtype::Float32,
                 src_device);

    // t[:, [1, 2], [1, 2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, dst_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, src_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64,
                   dst_device)};

    dst_t.IndexSet(indices, src_t);  // We cannot use T.Fill() here
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
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
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetDtype(), src_t.GetDtype());
    EXPECT_EQ(dst_t.ToFlatVector<float>(), vals);
}

TEST_P(TensorPermuteDevicePairs, CopyBool) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    Dtype dtype(Dtype::Bool);
    SizeVector shape{2, 3};

    std::vector<bool> vals{true, false, true, false, true, true};
    Tensor src_t(vals, shape, dtype, src_device);

    Tensor dst_t = src_t.Copy(dst_device);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetDtype(), src_t.GetDtype());
    EXPECT_EQ(dst_t.ToFlatVector<bool>(), vals);
}

TEST_P(TensorPermuteDevices, To) {
    Device device = GetParam();

    Dtype dtype(Dtype::Float32);
    SizeVector shape{2, 3};

    std::vector<float> src_vals{0.1, 1.2, 2.3, 3.4, 4.5, 5.6};
    std::vector<int> dst_vals{0, 1, 2, 3, 4, 5};
    Tensor src_t(src_vals, shape, dtype, device);

    Tensor dst_t = src_t.To(Dtype::Int32);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), device);
    EXPECT_EQ(dst_t.GetDtype(), Dtype::Int32);
    EXPECT_EQ(dst_t.ToFlatVector<int>(), dst_vals);
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
    dst_t.CopyFrom(src_t);  // Equivalently, dst_t.AsRvalue() = src_t;

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Expand) {
    Device device = GetParam();
    Dtype dtype(Dtype::Float32);

    // Expand {2, 1, 3} to {2, 2, 2, 3} without memory copy
    SizeVector src_shape{2, 1, 3};
    SizeVector dst_shape{2, 2, 2, 3};

    std::vector<float> src_vals{0, 1, 2, 3, 4, 5};
    std::vector<float> dst_vals{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
    Tensor src_t(src_vals, src_shape, dtype, device);
    Tensor dst_t = src_t.Expand(dst_shape);

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<float>(), dst_vals);
    EXPECT_EQ(dst_t.GetBlob(), src_t.GetBlob());
    EXPECT_EQ(dst_t.GetDataPtr(), src_t.GetDataPtr());
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
    EXPECT_THROW(Tensor({}, Dtype::Float32)[-1], std::runtime_error);
    EXPECT_THROW(Tensor({}, Dtype::Float32)[2], std::runtime_error);

    // Index out-of-bounds
    EXPECT_THROW(Tensor({0, 1}, Dtype::Float32)[0], std::runtime_error);
    EXPECT_THROW(Tensor({0, 1}, Dtype::Float32)[-1], std::runtime_error);
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

    t_0 = t[-2];  // t[-2] == t[0]
    EXPECT_EQ(t_0.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_0.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_0.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_0.GetBlob(), t.GetBlob());

    Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());

    t_1 = t[-1];  // t[-1] == t[1]
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());

    Tensor t_1_2 = t[1][2];
    EXPECT_EQ(t_1_2.GetShape(), SizeVector({4}));
    EXPECT_EQ(t_1_2.GetStrides(), SizeVector({1}));
    EXPECT_EQ(t_1_2.GetDataPtr(), static_cast<char *>(t.GetDataPtr()) +
                                          (1 * 3 * 4 + 2 * 4) * sizeof(float));
    EXPECT_EQ(t_1_2.GetBlob(), t.GetBlob());

    Tensor t_1_2_3 = t[1][2][3];
    EXPECT_EQ(t_1_2_3.GetShape(), SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetStrides(), SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) +
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

    Tensor t5(std::vector<bool>{true, false, true, true, false, false}, {2, 3},
              Dtype::Bool, device);
    EXPECT_EQ(t5.ToString(/*with_suffix=*/false),
              R"([[True False True],
 [True False False]])");
}

TEST_P(TensorPermuteDevicePairs, CopyContiguous) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, src_device);
    EXPECT_TRUE(t.IsContiguous());

    Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);
    EXPECT_TRUE(t_0.IsContiguous());

    Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_NE(t_1.GetDataPtr(), t_1.GetBlob()->GetDataPtr());
    EXPECT_TRUE(t_1.IsContiguous());

    Tensor t_1_copy = t_1.Copy(dst_device);
    EXPECT_EQ(t_1_copy.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(t_1_copy.GetStrides(), SizeVector({4, 1}));
    EXPECT_EQ(t_1_copy.GetDataPtr(),
              t_1_copy.GetBlob()->GetDataPtr());  // Points to beginning of Blob
}

TEST_P(TensorPermuteDevices, Slice) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);
    const void *blob_head = t.GetBlob()->GetDataPtr();
    EXPECT_EQ(t.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(t.GetStrides(), SizeVector({12, 4, 1}));
    EXPECT_EQ(t.GetDataPtr(), blob_head);

    // t_1 = t[0:2:1], effectively not sliced
    Tensor t_1 = t.Slice(0, 0, 2, 1);
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({12, 4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(), blob_head);
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23}));

    // t_2 = t[0:2:1][:, 0:3:2, :]
    Tensor t_2 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2);
    EXPECT_EQ(t_2.GetShape(), SizeVector({2, 2, 4}));
    EXPECT_EQ(t_2.GetStrides(), SizeVector({12, 8, 1}));
    EXPECT_EQ(t_2.GetDataPtr(), blob_head);
    EXPECT_EQ(t_2.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20,
                                  21, 22, 23}));

    // t_3 = [0:2:1, 0:3:2, 0:4:2]
    Tensor t_3 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_EQ(t_3.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_3.GetStrides(), SizeVector({12, 8, 2}));
    EXPECT_EQ(t_3.GetDataPtr(), blob_head);
    EXPECT_EQ(t_3.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // t_4 = t[1, 0:3:2, 0:4:2], a mix of [] and slice
    Tensor t_4 = t[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    EXPECT_EQ(t_4.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_4.GetStrides(), SizeVector({8, 2}));
    EXPECT_EQ(t_4.GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
    EXPECT_EQ(t_4.ToFlatVector<float>(), std::vector<float>({12, 14, 20, 22}));

    // t_5 = t[1, 0:-1, 0:-2:2] == t[1, 0:2, 0:2:2]
    Tensor t_5 = t[1].Slice(0, 0, -1).Slice(1, 0, -2, 2);
    EXPECT_EQ(t_5.GetShape(), SizeVector({2, 1}));
    EXPECT_EQ(t_5.GetStrides(), SizeVector({4, 2}));
    EXPECT_EQ(t_5.GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      DtypeUtil::ByteSize(Dtype::Float32) * 3 * 4);
    EXPECT_EQ(t_5.ToFlatVector<float>(), std::vector<float>({12, 16}));
}

TEST_P(TensorPermuteDevices, GetItem) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t_1 = t[1, :3, 0:-1:2], effectively not sliced
    Tensor t_1 =
            t.GetItem({TensorKey::Index(1), TensorKey::Slice(None, 3, None),
                       TensorKey::Slice(0, -1, 2)});
    EXPECT_EQ(t_1.GetShape(), SizeVector({3, 2}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({4, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({12, 14, 16, 18, 20, 22}));
}

TEST_P(TensorPermuteDevices, GetItemAdvancedIndexing) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {24}, Dtype::Float32, device);

    // t_1 = t[[0, 1, 1, 2, 3, 5, 8, 13, 21]]
    Tensor index_tensor(std::vector<int64_t>{0, 1, 1, 2, 3, 5, 8, 13, 21}, {9},
                        Dtype::Int64, device);
    Tensor t_1 = t.GetItem(TensorKey::IndexTensor(index_tensor));
    EXPECT_EQ(t_1.GetShape(), SizeVector({9}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 1, 1, 2, 3, 5, 8, 13, 21}));
}

TEST_P(TensorPermuteDevices, GetItemAdvancedIndexingMixed) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t_1 = t[1, 0:2, [1, 2]]
    Tensor index_tensor(std::vector<int64_t>{1, 2}, {2}, Dtype::Int64, device);

    Tensor t_1 = t.GetItem({TensorKey::Index(1), TensorKey::Slice(0, 2, None),
                            TensorKey::IndexTensor(index_tensor)});
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({2, 1}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({13, 17, 14, 18}));
}

TEST_P(TensorPermuteDevices, SetItemAdvancedIndexing) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {24}, Dtype::Float32, device);

    // t[[1, 3]] = np.array([100, 300])
    Tensor index_tensor(std::vector<int64_t>{1, 3}, {2}, Dtype::Int64, device);
    Tensor fill_tensor(std::vector<float>{100, 300}, {2}, Dtype::Float32,
                       device);
    t.SetItem(TensorKey::IndexTensor(index_tensor), fill_tensor);
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0,  100, 2,  300, 4,  5,  6,  7,
                                  8,  9,   10, 11,  12, 13, 14, 15,
                                  16, 17,  18, 19,  20, 21, 22, 23}));
}

TEST_P(TensorPermuteDevices, SetItemAdvancedIndexingMixed) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    // t[1, 0:2, [1, 2]] = np.array([[100, 200], [300, 400]])
    Tensor index_tensor(std::vector<int64_t>{1, 2}, {2}, Dtype::Int64, device);
    Tensor fill_tensor(std::vector<float>{100, 200, 300, 400}, {2, 2},
                       Dtype::Float32, device);
    t.SetItem({TensorKey::Index(1), TensorKey::Slice(0, 2, None),
               TensorKey::IndexTensor(index_tensor)},
              fill_tensor);
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0,  1,   2,   3,  4,  5,   6,   7,
                                  8,  9,   10,  11, 12, 100, 300, 15,
                                  16, 200, 400, 19, 20, 21,  22,  23}));
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
    dst_slice.AsRvalue() = src_0;
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
    dst_slice.AsRvalue() = src_2;
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

TEST_P(TensorPermuteDevicePairs, CopyNonContiguous) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    // t[0:2:1, 0:3:2, 0:4:2]
    Tensor t_1 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_FALSE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2, 2}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({12, 8, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // Copy ensures contiguous
    {
        Tensor t_1_copy = t_1.Copy(src_device);
        EXPECT_TRUE(t_1_copy.IsContiguous());
        EXPECT_EQ(t_1_copy.GetShape(), SizeVector({2, 2, 2}));
        EXPECT_EQ(t_1_copy.GetStrides(), SizeVector({4, 2, 1}));
        EXPECT_EQ(t_1_copy.ToFlatVector<float>(),
                  std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));
    }
    {
        Tensor t_1_copy = t_1.Copy(dst_device);
        EXPECT_TRUE(t_1_copy.IsContiguous());
        EXPECT_EQ(t_1_copy.GetShape(), SizeVector({2, 2, 2}));
        EXPECT_EQ(t_1_copy.GetStrides(), SizeVector({4, 2, 1}));
        EXPECT_EQ(t_1_copy.ToFlatVector<float>(),
                  std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));
    }
}

TEST_P(TensorPermuteDevicePairs, IndexGet) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    // t[:, [1, 2], [1, 2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64,
                   idx_device)};

    Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetNegative) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    // t[:, [1, -1], [1, -2]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({1, -1}), {2}, Dtype::Int64,
                   idx_device),
            Tensor(std::vector<int64_t>({1, -2}), {2}, Dtype::Int64,
                   idx_device)};

    Tensor t_1 = t.IndexGet(indices);
    EXPECT_TRUE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), SizeVector({2, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevicePairs, IndexGet2DBroadcastedIndex) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    Tensor src_t(vals, {2, 3, 4, 2}, Dtype::Float32, src_device);

    // t[:, [[1], [0], [2]], [[0, 1], [2, 3], [0, 1]], :] to shape {2, 3, 2, 2}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({1, 0, 2}), {3, 1}, Dtype::Int64,
                   idx_device),
            Tensor(std::vector<int64_t>({0, 1, 2, 3, 0, 1}), {3, 2},
                   Dtype::Int64, idx_device),
            Tensor(SizeVector(), Dtype::Int64, idx_device),
    };

    Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({2, 3, 2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({8,  9,  10, 11, 4,  5,  6,  7,
                                  16, 17, 18, 19, 32, 33, 34, 35,
                                  28, 29, 30, 31, 40, 41, 42, 43}));
}

TEST_P(TensorPermuteDevicePairs, IndexGet2DBroadcastedIndexSplitBySlice) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    Tensor src_t(vals, {2, 3, 2, 4}, Dtype::Float32, src_device);

    // t[:, [[1], [0], [2]], :, [[0, 1], [2, 3], [0, 1]]] to shape {3, 2, 2, 2}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({1, 0, 2}), {3, 1}, Dtype::Int64,
                   idx_device),
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>({0, 1, 2, 3, 0, 1}), {3, 2},
                   Dtype::Int64, idx_device),

    };

    Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({3, 2, 2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({8,  12, 32, 36, 9,  13, 33, 37,
                                  2,  6,  26, 30, 3,  7,  27, 31,
                                  16, 20, 40, 44, 17, 21, 41, 45}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetAssignToBroadcast) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    // t[:, [1, 2], [1, 2]] to shape {2, 2}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, dst_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, dst_device),
            Tensor(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64,
                   dst_device)};

    // Broadcast to shape {3, 2, 2}
    SizeVector dst_shape{3, 2, 2};
    Tensor dst_t(dst_shape, Dtype::Float32, dst_device);
    dst_t.AsRvalue() =
            src_t.IndexGet(indices);  // Intermediate tensor copied internally

    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), SizeVector({3, 2, 2}));
    EXPECT_EQ(
            dst_t.ToFlatVector<float>(),
            std::vector<float>({5, 10, 17, 22, 5, 10, 17, 22, 5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetSeparateBySlice) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    // t[[0, 1], :, [0, 1]]
    std::vector<Tensor> indices = {
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, idx_device),
            Tensor(SizeVector(), Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, idx_device)};

    Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_EQ(dst_t.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 4, 8, 13, 17, 21}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetSliceEnd) {
    Device idx_device;
    Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor src_t(vals, {2, 3, 4}, Dtype::Float32, src_device);

    std::vector<Tensor> indices = {
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, idx_device),
            Tensor(std::vector<int64_t>{0, 1}, {2}, Dtype::Int64, idx_device),
            Tensor(SizeVector(), Dtype::Int64, idx_device)};

    Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_EQ(dst_t.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 16, 17, 18, 19}));
}

TEST_P(TensorPermuteDevicePairs, IndexSet) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals({4, 6, 5, 16, 18, 17});
    Tensor src_t(vals, {2, 3}, Dtype::Float32, src_device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    Tensor dst_t(zeros, {2, 3, 4}, Dtype::Float32, dst_device);

    // t[:, [1], [0, 2, 1]]
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, src_device),
            Tensor(std::vector<int64_t>({1}), {1}, Dtype::Int64, dst_device),
            Tensor(std::vector<int64_t>({0, 2, 1}), {3}, Dtype::Int64,
                   src_device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 4,  5,  6,  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 16, 17, 18, 0, 0, 0, 0, 0}));
}

TEST_P(TensorPermuteDevicePairs, IndexSetBroadcast) {
    Device dst_device;
    Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> src_vals({10, 20});
    Tensor src_t(src_vals, {2, 1}, Dtype::Float32, src_device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    Tensor dst_t(zeros, {2, 3, 4}, Dtype::Float32, dst_device);

    // t[:, [1], [0, 2, 1]] -> slice {2, 3, 4} to {2, 3}
    std::vector<Tensor> indices = {
            Tensor(SizeVector(), Dtype::Int64, src_device),
            Tensor(std::vector<int64_t>({1}), {1}, Dtype::Int64, dst_device),
            Tensor(std::vector<int64_t>({0, 2, 1}), {3}, Dtype::Int64,
                   src_device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, Permute) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    Tensor t_1 = t.Permute({2, 1, 0});
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_1.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_1.GetShape(), SizeVector({4, 3, 2}));
    EXPECT_EQ(t_1.GetStrides(), SizeVector({1, 4, 12}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 12, 4, 16, 8,  20, 1, 13, 5, 17, 9,  21,
                                  2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}));

    Tensor t_2 = t.Permute({0, 2, 1});
    EXPECT_EQ(t_2.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_2.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_2.GetShape(), SizeVector({2, 4, 3}));
    EXPECT_EQ(t_2.GetStrides(), SizeVector({12, 1, 4}));
    EXPECT_EQ(t_2.ToFlatVector<float>(),
              std::vector<float>({0,  4,  8,  1,  5,  9,  2,  6,
                                  10, 3,  7,  11, 12, 16, 20, 13,
                                  17, 21, 14, 18, 22, 15, 19, 23}));
}

TEST_P(TensorPermuteDevices, Transpose) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {2, 3, 4}, Dtype::Float32, device);

    Tensor t_t = t.Transpose(1, 2);
    EXPECT_EQ(t_t.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_t.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_t.GetShape(), SizeVector({2, 4, 3}));
    EXPECT_EQ(t_t.GetStrides(), SizeVector({12, 1, 4}));
    EXPECT_EQ(t_t.ToFlatVector<float>(),
              std::vector<float>({0,  4,  8,  1,  5,  9,  2,  6,
                                  10, 3,  7,  11, 12, 16, 20, 13,
                                  17, 21, 14, 18, 22, 15, 19, 23}));
    EXPECT_THROW(t.Transpose(3, 5), std::runtime_error);
}

TEST_P(TensorPermuteDevices, T) {
    Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor t(vals, {6, 4}, Dtype::Float32, device);

    Tensor t_t = t.T();
    EXPECT_EQ(t_t.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_t.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_t.GetShape(), SizeVector({4, 6}));
    EXPECT_EQ(t_t.GetStrides(), SizeVector({1, 4}));
    EXPECT_EQ(t_t.ToFlatVector<float>(),
              std::vector<float>({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                  2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));

    Tensor t_3d(vals, {2, 3, 4}, Dtype::Float32, device);
    EXPECT_THROW(t_3d.T(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, ShallowCopyConstructor) {
    Device device = GetParam();
    Tensor t({2, 3}, Dtype::Float32, device);

    // Copy constructor.
    Tensor t_copy(t);
    EXPECT_EQ(t.GetDataPtr(), t_copy.GetDataPtr());

    // Vector initialization.
    std::vector<Tensor> t_vec0{t};
    EXPECT_EQ(t.GetDataPtr(), t_vec0[0].GetDataPtr());

    std::vector<Tensor> t_vec1({t});
    EXPECT_EQ(t.GetDataPtr(), t_vec1[0].GetDataPtr());

    // Vector initialization list passed to function.
    auto FirstTensorDataPtr = [](const std::vector<Tensor> &tensors) -> void * {
        return const_cast<void *>(tensors[0].GetDataPtr());
    };
    EXPECT_EQ(t.GetDataPtr(), FirstTensorDataPtr({t}));
}

TEST_P(TensorPermuteDevices, AdvancedIndexing_IsIndexSplittedBySlice) {
    Device device = GetParam();

    Tensor idx(std::vector<int64_t>({1, 2}), {2}, Dtype::Int64, device);
    Tensor slice(Tensor(SizeVector(), Dtype::Int64, device));

    EXPECT_FALSE(AdvancedIndexPreprocessor::IsIndexSplittedBySlice({slice}));
    EXPECT_FALSE(
            AdvancedIndexPreprocessor::IsIndexSplittedBySlice({slice, idx}));
    EXPECT_FALSE(
            AdvancedIndexPreprocessor::IsIndexSplittedBySlice({idx, slice}));
    EXPECT_FALSE(AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {slice, idx, idx}));
    EXPECT_FALSE(AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {slice, idx, idx, slice}));

    EXPECT_TRUE(AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {idx, slice, idx}));
    EXPECT_TRUE(AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {idx, slice, slice, idx}));
}

TEST_P(TensorPermuteDevices, Add) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({10, 11, 12, 13, 14, 15}), {2, 3},
             Dtype::Float32, device);
    Tensor c = a + b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({10, 12, 14, 16, 18, 20}));
}

TEST_P(TensorPermuteDevices, Add_) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({10, 11, 12, 13, 14, 15}), {2, 3},
             Dtype::Float32, device);
    a += b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({10, 12, 14, 16, 18, 20}));
}

TEST_P(TensorPermuteDevices, Add_BroadcastException) {
    // A.shape = (   3, 4)
    // B.shape = (2, 3, 4)
    // A += B should throw exception.
    // B += A is fine.
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}), {3, 4},
             Dtype::Float32, device);
    Tensor b(std::vector<float>({0,  1,  2,  3,  4,  5,  6,  7,
                                 8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23}),
             {2, 3, 4}, Dtype::Float32, device);
    EXPECT_THROW(a += b, std::runtime_error);
    b += a;
    EXPECT_EQ(b.ToFlatVector<float>(),
              std::vector<float>({0,  2,  4,  6,  8,  10, 12, 14,
                                  16, 18, 20, 22, 12, 14, 16, 18,
                                  20, 22, 24, 26, 28, 30, 32, 34}));
}

TEST_P(TensorPermuteDevices, Sub) {
    Device device = GetParam();
    Tensor a(std::vector<float>({10, 12, 14, 16, 18, 20}), {2, 3},
             Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    Tensor c = a - b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({10, 11, 12, 13, 14, 15}));
}

TEST_P(TensorPermuteDevices, Sub_) {
    Device device = GetParam();
    Tensor a(std::vector<float>({10, 12, 14, 16, 18, 20}), {2, 3},
             Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    a -= b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({10, 11, 12, 13, 14, 15}));
}

TEST_P(TensorPermuteDevices, Mul) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({6, 7, 8, 9, 10, 11}), {2, 3}, Dtype::Float32,
             device);
    Tensor c = a * b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({0, 7, 16, 27, 40, 55}));
}

TEST_P(TensorPermuteDevices, Mul_) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({6, 7, 8, 9, 10, 11}), {2, 3}, Dtype::Float32,
             device);
    a *= b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({0, 7, 16, 27, 40, 55}));
}

TEST_P(TensorPermuteDevices, Div) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 7, 16, 27, 40, 55}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({6, 7, 8, 9, 10, 11}), {2, 3}, Dtype::Float32,
             device);
    Tensor c = a / b;
    EXPECT_EQ(c.ToFlatVector<float>(), std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, Div_) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 7, 16, 27, 40, 55}), {2, 3}, Dtype::Float32,
             device);
    Tensor b(std::vector<float>({6, 7, 8, 9, 10, 11}), {2, 3}, Dtype::Float32,
             device);
    a /= b;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, ReduceSumKeepDim) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22.f, 23.f, 20.f, 9.f,  6.f, 14.f, 18.f, 13.f,
                                15.f, 3.f,  17.f, 0.f,  7.f, 21.f, 11.f, 1.f,
                                4.f,  2.f,  10.f, 19.f, 5.f, 8.f,  16.f, 12.f}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.Sum({}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Sum({0}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({1, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({29.f, 44.f, 31.f, 10.f, 10.f, 16.f, 28.f,
                                  32.f, 20.f, 11.f, 33.f, 12.f}));

    dst = src.Sum({1}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 1, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {43.f, 40.f, 55.f, 22.f, 16.f, 31.f, 37.f, 32.f}));

    dst = src.Sum({2}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({74.f, 51.f, 35.f, 40.f, 35.f, 41.f}));

    dst = src.Sum({0, 1}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({1, 1, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({59.f, 71.f, 92.f, 54.f}));

    dst = src.Sum({0, 2}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({1, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    dst = src.Sum({1, 2}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    dst = src.Sum({0, 1, 2}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({1, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({276.f}));

    // Dim order does not matter: {2, 1} -> {1, 2}.
    dst = src.Sum({2, 1}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    // Dim should be wrapped automatically: {-1, 0} -> {2, 0} -> {0, 2}.
    dst = src.Sum({-1, 0}, true);
    EXPECT_EQ(dst.GetShape(), SizeVector({1, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    // Exception cases.
    EXPECT_THROW(src.Sum({5}, true), std::runtime_error);      // Out-of-range.
    EXPECT_THROW(src.Sum({0, -4}, true), std::runtime_error);  // Out-of-range.
    EXPECT_THROW(src.Sum({0, 0}, true), std::runtime_error);   // Repeated.
    EXPECT_THROW(src.Sum({2, -1}, true), std::runtime_error);  // Repeated.
}

TEST_P(TensorPermuteDevices, ReduceSumNotKeepDim) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22.f, 23.f, 20.f, 9.f,  6.f, 14.f, 18.f, 13.f,
                                15.f, 3.f,  17.f, 0.f,  7.f, 21.f, 11.f, 1.f,
                                4.f,  2.f,  10.f, 19.f, 5.f, 8.f,  16.f, 12.f}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({29.f, 44.f, 31.f, 10.f, 10.f, 16.f, 28.f,
                                  32.f, 20.f, 11.f, 33.f, 12.f}));

    dst = src.Sum({1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {43.f, 40.f, 55.f, 22.f, 16.f, 31.f, 37.f, 32.f}));

    dst = src.Sum({2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({74.f, 51.f, 35.f, 40.f, 35.f, 41.f}));

    dst = src.Sum({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({59.f, 71.f, 92.f, 54.f}));

    dst = src.Sum({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    dst = src.Sum({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    dst = src.Sum({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({276.f}));
}

TEST_P(TensorPermuteDevices, ReduceMultipleOutputsSumLargeArray) {
    Device device = GetParam();
    SizeVector shape{3, 7, 8234719};
    int64_t size = shape.NumElements();
    std::vector<int> vals(size, 1);
    Tensor src(vals, shape, Dtype::Int32, device);
    Tensor dst;

    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 7, 8234719}));
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>(3 * 7 * 8234719, 1));

    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({7, 8234719}));
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>(7 * 8234719, 3));
}

TEST_P(TensorPermuteDevices, ReduceSum64bit1D) {
    Device device = GetParam();
    // num_bytes = 8 * (2 ^ 28) + 1 = 2 ^ 31 + 1 ~= 2GB
    // max_offsets = num_bytes - 1 = 2 ^ 31
    // max_32_bit_indexing = 2 ^ 31 - 1
    // max_offsets > max_32_bit_indexing
    int64_t num_elements = (1ULL << 28) + 10;
    std::vector<int64_t> vals(num_elements, 1);
    Tensor src(vals, {num_elements}, Dtype::Int64, device);
    Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(1, num_elements));
}

// np.sum(np.ones((2, large_dim)), dim=0)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase0) {
    Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    Tensor src(vals, shape, Dtype::Int64, device);
    Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({large_dim}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(large_dim, 2));

    Tensor src_sliced = src.GetItem({TensorKey::Slice(None, None, None),
                                     TensorKey::Slice(30, large_dim, None)});
    EXPECT_EQ(src_sliced.GetShape(), SizeVector({2, large_dim - 30}));
    dst = src_sliced.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({large_dim - 30}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(large_dim - 30, 2));
}

// np.sum(np.ones((2, large_dim)), dim=1)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase1) {
    Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    Tensor src(vals, shape, Dtype::Int64, device);
    Tensor dst;

    dst = src.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(2, large_dim));

    Tensor src_sliced = src.GetItem({TensorKey::Slice(None, None, None),
                                     TensorKey::Slice(30, large_dim, None)});
    EXPECT_EQ(src_sliced.GetShape(), SizeVector({2, large_dim - 30}));
    dst = src_sliced.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(2, large_dim - 30));
}

// np.sum(np.ones((large_dim, 2)), dim=0)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase2) {
    Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{large_dim, 2};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    Tensor src(vals, shape, Dtype::Int64, device);
    Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(2, large_dim));

    Tensor src_sliced = src.GetItem({TensorKey::Slice(30, large_dim, None),
                                     TensorKey::Slice(None, None, None)});
    EXPECT_EQ(src_sliced.GetShape(), SizeVector({large_dim - 30, 2}));
    dst = src_sliced.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(2, large_dim - 30));
}

// np.sum(np.ones((large_dim, 2)), dim=1)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase3) {
    Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{large_dim, 2};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    Tensor src(vals, shape, Dtype::Int64, device);
    Tensor dst;

    dst = src.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({large_dim}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(large_dim, 2));

    Tensor src_sliced = src.GetItem({TensorKey::Slice(30, large_dim, None),
                                     TensorKey::Slice(None, None, None)});
    EXPECT_EQ(src_sliced.GetShape(), SizeVector({large_dim - 30, 2}));
    dst = src_sliced.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), SizeVector({large_dim - 30}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(large_dim - 30, 2));
}

TEST_P(TensorPermuteDevices, ReduceSumLargeArray) {
    Device device = GetParam();

    std::vector<int64_t> sizes = TensorSizes::TestCases();
    int64_t max_size = *std::max_element(sizes.begin(), sizes.end());
    std::vector<int> vals(max_size);
    std::transform(vals.begin(), vals.end(), vals.begin(),
                   [](int x) -> int { return utility::UniformRandInt(0, 3); });

    for (int64_t size : sizes) {
        int ref_result = std::accumulate(vals.begin(), vals.begin() + size, 0,
                                         std::plus<int>());
        Tensor src(std::vector<int>(vals.begin(), vals.begin() + size), {size},
                   Dtype::Int32, device);
        Tensor dst = src.Sum({0}, false);

        EXPECT_EQ(dst.GetShape(), SizeVector({}));
        EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>({ref_result}));
    }
}

TEST_P(TensorPermuteDevices, ReduceProd) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22.f, 23.f, 20.f, 9.f,  6.f, 14.f, 18.f, 13.f,
                                15.f, 3.f,  17.f, 0.f,  7.f, 21.f, 11.f, 1.f,
                                4.f,  2.f,  10.f, 19.f, 5.f, 8.f,  16.f, 12.f}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.Prod({}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Prod({0}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({154.f, 483.f, 220.f, 9.f, 24.f, 28.f, 180.f,
                                  247.f, 75.f, 24.f, 272.f, 0.f}));

    dst = src.Prod({1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({1980.f, 966.f, 6120.f, 0.f, 140.f, 336.f,
                                  1760.f, 228.f}));

    dst = src.Prod({2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {91080.f, 19656.f, 0.f, 1617.f, 1520.f, 7680.f}));

    dst = src.Prod({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({277200.f, 324576.f, 10771200.f, 0.f}));

    dst = src.Prod({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({147276360.f, 29877120.f, 0.f}));

    dst = src.Prod({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0.f, 18876211200.f}));

    dst = src.Prod({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f}));
}

TEST_P(TensorPermuteDevices, ReduceMin) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22.f, 23.f, 20.f, 9.f,  6.f, 14.f, 18.f, 13.f,
                                15.f, 3.f,  17.f, 0.f,  7.f, 21.f, 11.f, 1.f,
                                4.f,  2.f,  10.f, 19.f, 5.f, 8.f,  16.f, 12.f}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.Min({}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Min({0}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({7.f, 21.f, 11.f, 1.f, 4.f, 2.f, 10.f, 13.f,
                                  5.f, 3.f, 16.f, 0.f}));

    dst = src.Min({1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({6.f, 3.f, 17.f, 0.f, 4.f, 2.f, 10.f, 1.f}));

    dst = src.Min({2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({9.f, 6.f, 0.f, 1.f, 2.f, 5.f}));

    dst = src.Min({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({4.f, 2.f, 10.f, 0.f}));

    dst = src.Min({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1.f, 2.f, 0.f}));

    dst = src.Min({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f, 1.f}));

    dst = src.Min({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f}));
}

TEST_P(TensorPermuteDevices, ReduceMax) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22.f, 23.f, 20.f, 9.f,  6.f, 14.f, 18.f, 13.f,
                                15.f, 3.f,  17.f, 0.f,  7.f, 21.f, 11.f, 1.f,
                                4.f,  2.f,  10.f, 19.f, 5.f, 8.f,  16.f, 12.f}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.Max({}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Max({0}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f, 14.f, 18.f, 19.f,
                                  15.f, 8.f, 17.f, 12.f}));

    dst = src.Max({1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {22.f, 23.f, 20.f, 13.f, 7.f, 21.f, 16.f, 19.f}));

    dst = src.Max({2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({23.f, 18.f, 17.f, 21.f, 19.f, 16.f}));

    dst = src.Max({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 19.f}));

    dst = src.Max({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({23.f, 19.f, 17.f}));

    dst = src.Max({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({23.f, 21.f}));

    dst = src.Max({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({23.f}));
}

TEST_P(TensorPermuteDevices, ReduceArgMin) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22, 23, 20, 9, 6, 14, 18, 13, 15, 3, 17, 0,
                                7,  21, 11, 1, 4, 2,  10, 19, 5,  8, 16, 12}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.ArgMin({0, 1, 2});
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>({11}));

    dst = src.ArgMin({0});
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0}));

    dst = src.ArgMin({1});
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 2, 2, 2, 1, 1, 1, 0}));

    dst = src.ArgMin({2});
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({3, 0, 3, 3, 1, 0}));
}

TEST_P(TensorPermuteDevices, ReduceArgMax) {
    Device device = GetParam();
    Tensor src(
            std::vector<float>({22, 23, 20, 9, 6, 14, 18, 13, 15, 3, 17, 0,
                                7,  21, 11, 1, 4, 2,  10, 19, 5,  8, 16, 12}),
            {2, 3, 4}, Dtype::Float32, device);
    Tensor dst;

    dst = src.ArgMax({0, 1, 2});
    EXPECT_EQ(dst.GetShape(), SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>({1}));

    dst = src.ArgMax({0});
    EXPECT_EQ(dst.GetShape(), SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1}));

    dst = src.ArgMax({1});
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({0, 0, 0, 1, 0, 0, 2, 1}));

    dst = src.ArgMax({2});
    EXPECT_EQ(dst.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 2, 2, 1, 3, 2}));
}

TEST_P(TensorPermuteDevices, Sqrt) {
    Device device = GetParam();
    Tensor src(std::vector<float>({0, 1, 4, 9, 16, 25}), {2, 3}, Dtype::Float32,
               device);
    Tensor dst = src.Sqrt();
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));

    // Sqrt only works for float types, throws exception otherwise.
    src = Tensor({2, 3}, Dtype::Int32, device);
    EXPECT_THROW(src.Sqrt(), std::runtime_error);

    // Negative number's sqrt shall be NaN.
    src = Tensor(std::vector<float>({0, 1, 4, 9, -16, -25}), {2, 3},
                 Dtype::Float32, device);
    dst = src.Sqrt();
    std::vector<float> dst_vals = dst.ToFlatVector<float>();
    EXPECT_EQ(dst_vals[0], 0);
    EXPECT_EQ(dst_vals[1], 1);
    EXPECT_EQ(dst_vals[2], 2);
    EXPECT_EQ(dst_vals[3], 3);
    EXPECT_TRUE(std::isnan(dst_vals[4]));
    EXPECT_TRUE(std::isnan(dst_vals[5]));

    // Inplace version.
    src = Tensor(std::vector<float>({0, 1, 4, 9, 16, 25}), {2, 3},
                 Dtype::Float32, device);
    src.Sqrt_();
    EXPECT_EQ(src.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, Sin) {
    Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::sin(v); });

    Tensor src(src_vals, {2, 3}, Dtype::Float32, device);
    Tensor dst = src.Sin();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Sin_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);

    // Only works for float types, throws exception otherwise.
    src = Tensor({2, 3}, Dtype::Int32, device);
    EXPECT_THROW(src.Sin(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Cos) {
    Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::cos(v); });

    Tensor src(src_vals, {2, 3}, Dtype::Float32, device);
    Tensor dst = src.Cos();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Cos_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);

    // Only works for float types, throws exception otherwise.
    src = Tensor({2, 3}, Dtype::Int32, device);
    EXPECT_THROW(src.Cos(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Neg) {
    Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals{2, 1, 0, -1, -2, -3};

    Tensor src(src_vals, {2, 3}, Dtype::Float32, device);
    Tensor dst = src.Neg();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Neg_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);

    // Also works for int.
    src = Tensor(std::vector<int>{-1, 0, 2}, {1, 3}, Dtype::Int32, device);
    dst = src.Neg();
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>({1, 0, -2}));
}

TEST_P(TensorPermuteDevices, Exp) {
    Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::exp(v); });

    Tensor src(src_vals, {2, 3}, Dtype::Float32, device);
    Tensor dst = src.Exp();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Exp_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);

    // Only works for float types, throws exception otherwise.
    src = Tensor({2, 3}, Dtype::Int32, device);
    EXPECT_THROW(src.Exp(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Abs) {
    Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::abs(v); });

    Tensor src(src_vals, {2, 3}, Dtype::Float32, device);
    Tensor dst = src.Abs();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Abs_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalNot) {
    Device device = GetParam();

    std::vector<bool> src_vals{true, false, true, false};
    std::vector<bool> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](bool v) -> bool { return !static_cast<bool>(v); });

    Tensor src(src_vals, {2, 2}, Dtype::Bool, device);
    Tensor dst = src.LogicalNot();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_vals);

    // Inplace version.
    src.LogicalNot_();
    EXPECT_EQ(src.ToFlatVector<bool>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalNotFloat) {
    Device device = GetParam();

    std::vector<float> src_vals{0, -1, 1, 2};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals), [](float v) -> float {
                       return static_cast<float>(!static_cast<bool>(v));
                   });
    std::vector<bool> dst_bool_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_bool_vals), [](float v) -> bool {
                       return static_cast<bool>(!static_cast<bool>(v));
                   });

    Tensor src(src_vals, {2, 2}, Dtype::Float32, device);
    Tensor dst = src.LogicalNot();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_bool_vals);

    // Inplace version.
    src.LogicalNot_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalAnd) {
    Device device = GetParam();
    Tensor a(std::vector<bool>({true, false, true, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor b(std::vector<bool>({true, true, false, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor c = a.LogicalAnd(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));
    c = a && b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));

    // Inplace version.
    a.LogicalAnd_(b);
    EXPECT_EQ(a.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));
}

TEST_P(TensorPermuteDevices, LogicalAndFloat) {
    Device device = GetParam();
    Tensor a(std::vector<float>({-1, 0, 1, 0}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({1, 0, 0, 0}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.LogicalAnd(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));

    // Inplace version.
    a.LogicalAnd_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, LogicalOr) {
    Device device = GetParam();
    Tensor a(std::vector<bool>({true, false, true, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor b(std::vector<bool>({true, true, false, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor c = a.LogicalOr(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, true, false}));
    c = a || b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, true, false}));

    // Inplace version.
    a.LogicalOr_(b);
    EXPECT_EQ(a.ToFlatVector<bool>(),
              std::vector<bool>({true, true, true, false}));
}

TEST_P(TensorPermuteDevices, LogicalOrFloat) {
    Device device = GetParam();
    Tensor a(std::vector<float>({-1, 0, 1, 0}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({1, -1, 0, 0}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.LogicalOr(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, true, false}));

    // Inplace version.
    a.LogicalOr_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 1, 1, 0}));
}

TEST_P(TensorPermuteDevices, LogicalXor) {
    Device device = GetParam();
    Tensor a(std::vector<bool>({true, false, true, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor b(std::vector<bool>({true, true, false, false}), {2, 2}, Dtype::Bool,
             device);
    Tensor c = a.LogicalXor(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));

    // Inplace version.
    a.LogicalXor_(b);
    EXPECT_EQ(a.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));
}

TEST_P(TensorPermuteDevices, LogicalXorFloat) {
    Device device = GetParam();
    Tensor a(std::vector<float>({-1, 0, 1, 0}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({1, -1, 0, 0}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.LogicalXor(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));

    // Inplace version.
    a.LogicalXor_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 1, 0}));
}

TEST_P(TensorPermuteDevices, Gt) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Gt(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, false, false}));
    c = a > b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, false, false}));

    // Inplace version.
    a.Gt_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 0, 0}));
}

TEST_P(TensorPermuteDevices, Lt) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Lt(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, false, true, true}));
    c = a < b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, false, true, true}));

    // Inplace version.
    a.Lt_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 0, 1, 1}));
}

TEST_P(TensorPermuteDevices, Ge) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Ge(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, false, false}));
    c = a >= b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, false, false}));

    // Inplace version.
    a.Ge_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 1, 0, 0}));
}

TEST_P(TensorPermuteDevices, Le) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Le(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, true, true}));
    c = a <= b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, true, true}));

    // Inplace version.
    a.Le_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 0, 1, 1}));
}

TEST_P(TensorPermuteDevices, Eq) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Eq(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));
    c = a == b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));

    // Inplace version.
    a.Eq_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, Ne) {
    Device device = GetParam();
    Tensor a(std::vector<float>({0, 1, -1, 1}), {2, 2}, Dtype::Float32, device);
    Tensor b(std::vector<float>({0, 0, 0, 2}), {2, 2}, Dtype::Float32, device);
    Tensor c = a.Ne(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, true}));
    c = a != b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, true}));

    // Inplace version.
    a.Ne_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 1, 1}));
}

TEST_P(TensorPermuteDevices, CreationEmpty) {
    Device device = GetParam();

    Tensor a = Tensor::Empty({}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({}));
    EXPECT_EQ(a.NumElements(), 1);

    a = Tensor::Empty({0}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({0}));
    EXPECT_EQ(a.NumElements(), 0);

    a = Tensor::Empty({1}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({1}));
    EXPECT_EQ(a.NumElements(), 1);

    a = Tensor::Empty({0, 1}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({0, 1}));
    EXPECT_EQ(a.NumElements(), 0);

    a = Tensor::Empty({2, 3}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
}

TEST_P(TensorPermuteDevices, CreationFull) {
    Device device = GetParam();

    const float fill_value = 100;
    Tensor a = Tensor::Full({}, fill_value, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({}));
    EXPECT_EQ(a.NumElements(), 1);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = Tensor::Full({0}, fill_value, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({0}));
    EXPECT_EQ(a.NumElements(), 0);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = Tensor::Full({1}, fill_value, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({1}));
    EXPECT_EQ(a.NumElements(), 1);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = Tensor::Full({0, 1}, fill_value, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({0, 1}));
    EXPECT_EQ(a.NumElements(), 0);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = Tensor::Full({2, 3}, fill_value, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));
}

TEST_P(TensorPermuteDevices, CreationZeros) {
    Device device = GetParam();

    Tensor a = Tensor::Zeros({2, 3}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>(a.NumElements(), 0));
}

TEST_P(TensorPermuteDevices, CreationOnes) {
    Device device = GetParam();

    Tensor a = Tensor::Ones({2, 3}, Dtype::Float32, device);
    EXPECT_EQ(a.GetShape(), SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>(a.NumElements(), 1));
}

TEST_P(TensorPermuteDevices, ScalarOperatorOverload) {
    Device device = GetParam();
    Tensor a;
    Tensor b;

    // +
    a = Tensor::Ones({2}, Dtype::Float32, device);
    b = a.Add(1);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = a + 1;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = 1 + a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = a + true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));

    // +=
    a = Tensor::Ones({2}, Dtype::Float32, device);
    a.Add_(1);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({2, 2}));
    a += 1;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({3, 3}));
    a += true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({4, 4}));

    // -
    a = Tensor::Ones({2}, Dtype::Float32, device);
    b = a.Sub(1);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));
    b = a - 1;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));
    b = 10 - a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({9, 9}));
    b = a - true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));

    // -=
    a = Tensor::Ones({2}, Dtype::Float32, device);
    a.Sub_(1);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 0}));
    a -= 1;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({-1, -1}));
    a -= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({-2, -2}));

    // *
    a = Tensor::Full({2}, 2, Dtype::Float32, device);
    b = a.Mul(10);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = a * 10;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = 10 * a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = a * true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));

    // *=
    a = Tensor::Full({2}, 2, Dtype::Float32, device);
    a.Mul_(10);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({20, 20}));
    a *= 10;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({200, 200}));
    a *= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({200, 200}));

    // /
    a = Tensor::Full({2}, 20, Dtype::Float32, device);
    b = a.Div(2);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({10, 10}));
    b = a / 2;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({10, 10}));
    b = 10 / a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0.5, 0.5}));
    b = a / true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));

    // /=
    a = Tensor::Full({2}, 20, Dtype::Float32, device);
    a.Div_(2);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({10, 10}));
    a /= 2;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({5, 5}));
    a /= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({5, 5}));
}
