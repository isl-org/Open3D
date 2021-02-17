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
#include "open3d/core/TensorList.h"
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
    TestObject(int val, void *ptr = nullptr) : val_(val), ptr_(ptr) {}

    bool operator==(const TestObject &other) const {
        return val_ == other.val_ && ptr_ == other.ptr_;
    }

private:
    int val_;
    void *ptr_;
};

static_assert(std::is_pod<TestObject>(), "TestObject must be a POD.");
static const int64_t byte_size = sizeof(TestObject);
static const std::string class_name = "TestObject";

TEST_P(TensorObjectPermuteDevices, Constructor) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    for (const core::SizeVector &shape : std::vector<core::SizeVector>{
                 {}, {0}, {0, 0}, {0, 1}, {1, 0}, {2, 3}}) {
        core::Tensor t(shape, dtype, device);
        EXPECT_EQ(t.GetShape(), shape);
        EXPECT_EQ(t.GetDtype(), dtype);
        EXPECT_EQ(t.GetDtype().ToString(), class_name);
        EXPECT_EQ(t.GetDtype().ByteSize(), byte_size);
        EXPECT_EQ(t.GetDevice(), device);
    }

    EXPECT_ANY_THROW(core::Tensor({-1}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({0, -2}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({-1, -1}, dtype, device));
}

TEST_P(TensorObjectPermuteDevices, WithInitValueObject) {
    core::Device device = GetParam();
    core::Dtype dtype =
            core::Dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    std::vector<TestObject> vals{TestObject(0), TestObject(1), TestObject(2),
                                 TestObject(3), TestObject(4), TestObject(5)};
    core::Tensor t(vals, {2, 3}, dtype, device);
    EXPECT_EQ(t.ToFlatVector<TestObject>(), vals);
}

TEST_P(TensorObjectPermuteDevices, FillObject) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    core::Tensor t(std::vector<TestObject>(2 * 3, 0), {2, 3}, dtype, device);
    t.FillObject(TestObject(1));
    EXPECT_EQ(t.ToFlatVector<TestObject>(),
              std::vector<TestObject>({1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorObjectPermuteDevicePairs, IndexSetFillFancyObject) {
    core::Device dst_device;
    core::Device src_device;

    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    std::tie(dst_device, src_device) = GetParam();
    core::Tensor dst_t(std::vector<TestObject>(2 * 3 * 4, 0), {2, 3, 4}, dtype,
                       dst_device);
    core::Tensor src_t(std::vector<TestObject>{1}, core::SizeVector({}), dtype,
                       src_device);

    // t[:, [1, 2], [1, 2]]
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Dtype::Int64, dst_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Dtype::Int64,
                         src_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Dtype::Int64,
                         dst_device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<TestObject>(),
              std::vector<TestObject>({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}));
}

TEST_P(TensorObjectPermuteDevicePairs, CopyObject) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);
    core::SizeVector shape{2, 3};

    std::vector<TestObject> vals{0, 1, 2, 3, 4, 5};
    core::Tensor src_t(vals, shape, dtype, src_device);
    core::Tensor dst_t = src_t.To(dst_device);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetDtype(), src_t.GetDtype());
    EXPECT_EQ(dst_t.ToFlatVector<TestObject>(), vals);
}

TEST_P(TensorObjectPermuteDevicePairs, CopyBroadcastObject) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    // Broadcast {2, 1, 3} to {2, 2, 2, 3}
    core::SizeVector src_shape{2, 1, 3};
    core::SizeVector dst_shape{2, 2, 2, 3};

    std::vector<TestObject> src_vals{0, 1, 2, 3, 4, 5};
    std::vector<TestObject> dst_vals{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                     0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
    core::Tensor src_t(src_vals, src_shape, dtype, src_device);
    core::Tensor dst_t(dst_shape, dtype, dst_device);
    dst_t.CopyFrom(src_t);  // Equivalently, dst_t.AsRvalue() = src_t;

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<TestObject>(), dst_vals);
}

TEST_P(TensorObjectPermuteDevices, ItemObject) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    std::vector<TestObject> vals{3, 7, 4};
    core::Tensor t(vals, {3}, dtype, device);

    core::Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<int>(), std::runtime_error);
    EXPECT_THROW(t_0.Item<uint8_t>(), std::runtime_error);

    EXPECT_EQ(t[0].Item<TestObject>(), TestObject(3));
    EXPECT_EQ(t[1].Item<TestObject>(), TestObject(7));
    EXPECT_EQ(t[2].Item<TestObject>(), TestObject(4));
}

TEST_P(TensorObjectPermuteDevices, ItemAssignObject) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    std::vector<TestObject> vals{0,  1,  2,  3,  4,  5,  6,  7,
                                 8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23};
    core::Tensor t(vals, {2, 3, 4}, dtype, device);

    // Assigning to rvalue
    TestObject new_val_0(100);
    t[1][2][3].AssignObject(new_val_0);
    EXPECT_EQ(t[1][2][3].Item<TestObject>(), TestObject(100));
}

TEST_P(TensorObjectPermuteDevices, IsSameObject) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    // "Shallow" copy.
    core::Tensor t0 = core::Tensor::Empty({6, 8}, dtype, device);
    core::Tensor t1 = t0;  // "Shallow" copy
    EXPECT_TRUE(t0.IsSame(t1));
    EXPECT_TRUE(t1.IsSame(t0));

    // Copy constructor copies view.
    core::Tensor t0_copy_construct(t0);
    EXPECT_TRUE(t0.IsSame(t0_copy_construct));
    EXPECT_TRUE(t0_copy_construct.IsSame(t0));

    // New tensor of the same value.
    core::Tensor t2 = core::Tensor::Empty({6, 8}, dtype, device);
    EXPECT_FALSE(t0.IsSame(t2));
    EXPECT_FALSE(t2.IsSame(t0));

    // Tensor::Contiguous() does not copy if already contiguous.
    core::Tensor t0_contiguous = t0.Contiguous();
    EXPECT_TRUE(t0.IsSame(t0_contiguous));
    EXPECT_TRUE(t0_contiguous.IsSame(t0));

    // Slices are views.
    core::Tensor t0_slice = t0.GetItem({core::TensorKey::Slice(0, 5, 2)})[0];
    // t0 [0:5:2][0]
    core::Tensor t1_slice = t1[0];
    EXPECT_TRUE(t0_slice.IsSame(t1_slice));
    EXPECT_TRUE(t1_slice.IsSame(t0_slice));
    // Explicit copy to the same device.
    core::Tensor t0_copy = t0.To(device, /*copy=*/true);
    EXPECT_FALSE(t0.IsSame(t0_copy));
    EXPECT_FALSE(t0_copy.IsSame(t0));
    // std::vector<Tensor> initializer list and push_back() are views.
    std::vector<core::Tensor> vec{t0};
    vec.push_back(t1);
    EXPECT_TRUE(t0.IsSame(vec[0]));
    EXPECT_TRUE(t1.IsSame(vec[1]));
    EXPECT_TRUE(vec[0].IsSame(vec[1]));
}

TEST_P(TensorObjectPermuteDevices, ConstructFromObjectTensorVector) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    core::Tensor t0 = core::Tensor::Empty({2, 3}, dtype, device);
    core::Tensor t1 = core::Tensor::Empty({2, 3}, dtype, device);
    core::Tensor t2 = core::Tensor::Empty({2, 3}, dtype, device);

    core::TensorList tl(std::vector<core::Tensor>({t0, t1, t2}));

    // Check tensor list.
    core::SizeVector full_shape({3, 2, 3});
    EXPECT_EQ(tl.AsTensor().GetShape(), full_shape);
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);

    // Values should be copied.
    EXPECT_ANY_THROW(tl[0].AllClose(t0));
    EXPECT_ANY_THROW(tl[1].AllClose(t1));
    EXPECT_ANY_THROW(tl[2].AllClose(t2));
    EXPECT_FALSE(tl[0].IsSame(t0));
    EXPECT_FALSE(tl[1].IsSame(t1));
    EXPECT_FALSE(tl[2].IsSame(t2));
}

TEST_P(TensorObjectPermuteDevices, TensorListFromObjectTensor) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Dtype::DtypeCode::Object, byte_size, class_name);

    core::Tensor t = core::Tensor::Empty({3, 4, 5}, dtype, device);

    // Copied tensor.
    core::TensorList tl = core::TensorList::FromTensor(t);
    EXPECT_EQ(tl.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl.GetSize(), 3);
    EXPECT_EQ(tl.GetReservedSize(), 8);
    EXPECT_ANY_THROW(tl.AsTensor().AllClose(t));
    EXPECT_FALSE(tl.AsTensor().IsSame(t));

    // Inplace tensor.
    core::TensorList tl_inplace = core::TensorList::FromTensor(t, true);
    EXPECT_EQ(tl_inplace.GetElementShape(), core::SizeVector({4, 5}));
    EXPECT_EQ(tl_inplace.GetSize(), 3);
    EXPECT_EQ(tl_inplace.GetReservedSize(), 3);
    EXPECT_ANY_THROW(tl_inplace.AsTensor().AllClose(t));
    EXPECT_TRUE(tl_inplace.AsTensor().IsSame(t));
}

}  // namespace tests
}  // namespace open3d
