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

#include "open3d/core/Tensor.h"

#include <cmath>
#include <limits>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/kernel/Kernel.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "tests/Tests.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

class TensorPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(Tensor,
                         TensorPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TensorPermuteDevicesWithSYCL : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorPermuteDevicesWithSYCL,
        testing::ValuesIn(PermuteDevicesWithSYCL::TestCases()));

class TensorPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorPermuteDevicePairs,
        testing::ValuesIn(TensorPermuteDevicePairs::TestCases()));

class TensorPermuteSizesDefaultStridesAndDevices
    : public testing::TestWithParam<
              std::tuple<std::pair<core::SizeVector, core::SizeVector>,
                         core::Device>> {};
INSTANTIATE_TEST_SUITE_P(
        Tensor,
        TensorPermuteSizesDefaultStridesAndDevices,
        testing::Combine(
                testing::ValuesIn(PermuteSizesDefaultStrides::TestCases()),
                testing::ValuesIn(PermuteDevices::TestCases())));

/// Convert to const reference.
/// https://stackoverflow.com/a/15519125/1255535
template <typename T>
static constexpr const T &AsConst(T &t) noexcept {
    return t;
}

TEST_P(TensorPermuteDevices, Constructor) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    for (const core::SizeVector &shape : std::vector<core::SizeVector>{
                 {}, {0}, {0, 0}, {0, 1}, {1, 0}, {2, 3}}) {
        core::Tensor t(shape, dtype, device);
        EXPECT_EQ(t.GetShape(), shape);
        EXPECT_EQ(t.GetDtype(), dtype);
        EXPECT_EQ(t.GetDevice(), device);
    }

    EXPECT_ANY_THROW(core::Tensor({-1}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({0, -2}, dtype, device));
    EXPECT_ANY_THROW(core::Tensor({-1, -1}, dtype, device));
}

TEST_P(TensorPermuteDevices, ConstructorBool) {
    core::Device device = GetParam();

    core::SizeVector shape{2, 3};
    core::Dtype dtype = core::Bool;
    core::Tensor t(shape, dtype, device);

    EXPECT_EQ(t.GetShape(), shape);
    EXPECT_EQ(t.GetBlob()->GetDevice(), device);
    EXPECT_EQ(t.GetDtype(), dtype);
}

TEST_P(TensorPermuteDevices, WithInitValue) {
    core::Device device = GetParam();

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    core::Tensor t(vals, {2, 3}, core::Float32, device);
    EXPECT_EQ(t.ToFlatVector<float>(), vals);

    // Wrapper
    {
        core::Tensor wrapper(t.GetDataPtr(), t.GetDtype(), t.GetShape(), {},
                             t.GetDevice());
        EXPECT_EQ(t.GetStrides(), wrapper.GetStrides());
        EXPECT_EQ(wrapper.ToFlatVector<float>(), vals);
        // Updating original data updates wrapper.
        t[1][1] = 0;
        vals[4] = 0;
        EXPECT_EQ(wrapper.ToFlatVector<float>(), vals);
    }
    // Original data is present after wrapper is destructed.
    EXPECT_EQ(t.ToFlatVector<float>(), vals);
}

TEST_P(TensorPermuteDevices, WithInitList) {
    core::Device device = GetParam();

    core::Tensor t;

    // 0-D tensor with given value.
    t = core::Tensor::Init<float>(1, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({}));
    EXPECT_EQ(t.GetDtype(), core::Float32);
    EXPECT_EQ(t.GetBlob()->GetDevice(), device);
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1}));

    // 1-D tensor initialization with list.
    t = core::Tensor::Init<float>({1, 2, 3}, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(t.GetDtype(), core::Float32);
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 2, 3}));

    // 2-D tensor initialization with list.
    t = core::Tensor::Init<int>({{1, 2, 3}, {4, 5, 6}}, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(t.GetDtype(), core::Int32);
    EXPECT_EQ(t.ToFlatVector<int>(), std::vector<int>({1, 2, 3, 4, 5, 6}));

    // 3-D tensor initialization with list.
    t = core::Tensor::Init<double>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}},
                                   device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 2, 2}));
    EXPECT_EQ(t.GetDtype(), core::Float64);
    EXPECT_EQ(t.ToFlatVector<double>(),
              std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8}));

    // Test boolean datatype.
    t = core::Tensor::Init<bool>({{true, false}, {false, true}}, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(t.GetDtype(), core::Bool);
    EXPECT_EQ(t.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, true}));

    // Test uint8 datatype.
    t = core::Tensor::Init<uint8_t>(
            {{{0, 0}, {0, 0}}, {{255, 255}, {255, 255}}}, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 2, 2}));
    EXPECT_EQ(t.GetDtype(), core::UInt8);
    EXPECT_EQ(t.ToFlatVector<uint8_t>(),
              std::vector<uint8_t>({0, 0, 0, 0, 255, 255, 255, 255}));

    // Check tensor element size mismatch.
    EXPECT_THROW(core::Tensor::Init<int>({{1, 2, 3}, {4, 5}}, device),
                 std::runtime_error);

    // Test shapes with 0-element.
    t = core::Tensor::Init<double>({}, device);
    EXPECT_EQ(t.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(t.GetDtype(), core::Float64);
    EXPECT_EQ(t.ToFlatVector<double>(), std::vector<double>({}));

    t = core::Tensor::Init<bool>({{}, {}});
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 0}));
    EXPECT_EQ(t.GetDtype(), core::Bool);
    EXPECT_EQ(t.ToFlatVector<bool>(), std::vector<bool>({}));

    t = core::Tensor::Init<bool>({{}});
    EXPECT_EQ(t.GetShape(), core::SizeVector({1, 0}));
    EXPECT_EQ(t.GetDtype(), core::Bool);
    EXPECT_EQ(t.ToFlatVector<bool>(), std::vector<bool>({}));

    t = core::Tensor::Init<uint8_t>({{{}}});
    EXPECT_EQ(t.GetShape(), core::SizeVector({1, 1, 0}));
    EXPECT_EQ(t.GetDtype(), core::UInt8);
    EXPECT_EQ(t.ToFlatVector<uint8_t>(), std::vector<uint8_t>({}));

    EXPECT_THROW(core::Tensor::Init<uint64_t>({{{}}, {{}, {}}}, device),
                 std::exception);

    EXPECT_THROW(core::Tensor::Init<uint64_t>({{}, {{}}}, device),
                 std::exception);

    EXPECT_THROW(core::Tensor::Init<uint64_t>({{}, {1}}, device),
                 std::exception);
}

TEST_P(TensorPermuteDevices, WithInitValueBool) {
    core::Device device = GetParam();

    std::vector<bool> vals{true, false, true, true, false, false};
    core::Tensor t(vals, {2, 3}, core::Bool, device);
    EXPECT_EQ(t.ToFlatVector<bool>(), vals);
}

TEST_P(TensorPermuteDevices, WithInitValueTypeMismatch) {
    core::Device device = GetParam();

    std::vector<int> vals{0, 1, 2, 3, 4, 5};
    EXPECT_THROW(core::Tensor(vals, {2, 3}, core::Float32, device),
                 std::runtime_error);
}

TEST_P(TensorPermuteDevices, WithInitValueSizeMismatch) {
    core::Device device = GetParam();

    std::vector<float> vals{0, 1, 2, 3, 4};
    EXPECT_THROW(core::Tensor(vals, {2, 3}, core::Float32, device),
                 std::runtime_error);
}

TEST_P(TensorPermuteDevices, Arange) {
    core::Device device = GetParam();
    core::Tensor arange;

    // Double value to float type.
    arange = core::Tensor::Arange(0.0, 5.0, 1.0, core::Float32, device);
    EXPECT_EQ(arange.GetDtype(), core::Float32);
    EXPECT_EQ(arange.GetShape(), core::SizeVector({5}));
    EXPECT_EQ(arange.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4}));

    // Double value to int type.
    arange = core::Tensor::Arange(0.0, 5.0, 1.0, core::Int32, device);
    EXPECT_EQ(arange.GetDtype(), core::Int32);
    EXPECT_EQ(arange.GetShape(), core::SizeVector({5}));
    EXPECT_EQ(arange.ToFlatVector<int>(), std::vector<int>({0, 1, 2, 3, 4}));

    // Int value to float type.
    arange = core::Tensor::Arange(0, 5, 1, core::Float32, device);
    EXPECT_EQ(arange.GetDtype(), core::Float32);
    EXPECT_EQ(arange.GetShape(), core::SizeVector({5}));
    EXPECT_EQ(arange.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4}));

    // Float value with non-integer step.
    arange = core::Tensor::Arange(0.1, 6.0, 2.0, core::Float32, device);
    EXPECT_EQ(arange.ToFlatVector<float>(),
              std::vector<float>({0.1, 2.1, 4.1}));

    // Float value with negative step.
    arange = core::Tensor::Arange(0.0, -4.1, -2.0, core::Float32, device);
    EXPECT_EQ(arange.ToFlatVector<float>(),
              std::vector<float>({0, -2.0, -4.0}));

    // Test empty set -- empty Tensor.
    arange = core::Tensor::Arange(0, 2, -2, core::Float32, device);
    EXPECT_EQ(arange.NumElements(), 0);

    // Test zero step -- error.
    EXPECT_THROW(core::Tensor::Arange(0, 2, 0, core::Float32, device),
                 std::runtime_error);
}

TEST_P(TensorPermuteDevices, Fill) {
    core::Device device = GetParam();
    core::Tensor t(std::vector<float>(2 * 3, 0), {2, 3}, core::Float32, device);
    t.Fill(1);
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorPermuteDevices, FillBool) {
    core::Device device = GetParam();
    core::Tensor t(std::vector<bool>(2 * 3, false), {2, 3}, core::Bool, device);
    t.Fill(true);
    EXPECT_EQ(t.ToFlatVector<bool>(), std::vector<bool>(2 * 3, true));
}

TEST_P(TensorPermuteDevices, FillSlice) {
    core::Device device = GetParam();
    core::Tensor t(std::vector<float>(2 * 3, 0), {2, 3}, core::Float32, device);
    t.Slice(1, 0, 3, 2).Fill(1);  // t[:, 0:3:2].fill(1)
    EXPECT_EQ(t.ToFlatVector<float>(), std::vector<float>({1, 0, 1, 1, 0, 1}));
}

TEST_P(TensorPermuteDevicePairs, IndexSetFillFancy) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    core::Tensor dst_t(std::vector<float>(2 * 3 * 4, 0), {2, 3, 4},
                       core::Float32, dst_device);
    core::Tensor src_t(std::vector<float>({1}), core::SizeVector({}),
                       core::Float32, src_device);

    // t[:, [1, 2], [1, 2]]
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, dst_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         src_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         dst_device)};

    dst_t.IndexSet(indices, src_t);  // We cannot use T.Fill() here
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}));
}

TEST_P(TensorPermuteDevicePairs, Copy) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Dtype dtype(core::Float32);
    core::SizeVector shape{2, 3};

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    core::Tensor src_t(vals, shape, dtype, src_device);

    core::Tensor dst_t = src_t.To(dst_device, /*copy=*/true);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetDtype(), src_t.GetDtype());
    EXPECT_EQ(dst_t.ToFlatVector<float>(), vals);
}

TEST_P(TensorPermuteDevicePairs, CopyBool) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Dtype dtype(core::Bool);
    core::SizeVector shape{2, 3};

    std::vector<bool> vals{true, false, true, false, true, true};
    core::Tensor src_t(vals, shape, dtype, src_device);

    core::Tensor dst_t = src_t.To(dst_device, /*copy=*/true);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetDtype(), src_t.GetDtype());
    EXPECT_EQ(dst_t.ToFlatVector<bool>(), vals);
}

TEST_P(TensorPermuteDevices, To) {
    core::Device device = GetParam();
    core::SizeVector shape{2, 3};

    std::vector<int> dst_vals{0, 1, 2, 3, 4, 5};
    core::Tensor src_t = core::Tensor::Init<float>(
            {{0.1, 1.2, 2.3}, {3.4, 4.5, 5.6}}, device);

    core::Tensor dst_t = src_t.To(core::Int32);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetDevice(), device);
    EXPECT_EQ(dst_t.GetDtype(), core::Int32);
    EXPECT_EQ(dst_t.ToFlatVector<int>(), dst_vals);
}

TEST_P(TensorPermuteDevicePairs, ToDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Tensor src_t = core::Tensor::Init<float>({0, 1, 2, 3}, src_device);
    core::Tensor dst_t = src_t.To(dst_device);
    EXPECT_TRUE(dst_t.To(src_device).AllClose(src_t));

    EXPECT_ANY_THROW(src_t.To(core::Device("CPU:1")));

    EXPECT_ANY_THROW(src_t.To(core::Device("CUDA:-1")));
    EXPECT_ANY_THROW(src_t.To(core::Device("CUDA:100000")));
}

TEST_P(TensorPermuteDevicePairs, CopyBroadcast) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    core::Dtype dtype(core::Float32);

    // Broadcast {2, 1, 3} to {2, 2, 2, 3}
    core::SizeVector src_shape{2, 1, 3};
    core::SizeVector dst_shape{2, 2, 2, 3};

    std::vector<float> src_vals{0, 1, 2, 3, 4, 5};
    std::vector<float> dst_vals{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
    core::Tensor src_t(src_vals, src_shape, dtype, src_device);
    core::Tensor dst_t(dst_shape, dtype, dst_device);
    dst_t.CopyFrom(src_t);  // Equivalently, dst_t.AsRvalue() = src_t;

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Expand) {
    core::Device device = GetParam();
    core::Dtype dtype(core::Float32);

    // Expand {2, 1, 3} to {2, 2, 2, 3} without memory copy
    core::SizeVector src_shape{2, 1, 3};
    core::SizeVector dst_shape{2, 2, 2, 3};

    std::vector<float> src_vals{0, 1, 2, 3, 4, 5};
    std::vector<float> dst_vals{0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5,
                                0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5};
    core::Tensor src_t(src_vals, src_shape, dtype, device);
    core::Tensor dst_t = src_t.Expand(dst_shape);

    EXPECT_EQ(dst_t.GetShape(), dst_shape);
    EXPECT_EQ(dst_t.ToFlatVector<float>(), dst_vals);
    EXPECT_EQ(dst_t.GetBlob(), src_t.GetBlob());
    EXPECT_EQ(dst_t.GetDataPtr(), src_t.GetDataPtr());
}

TEST_P(TensorPermuteDevices, Flatten) {
    core::Device device = GetParam();

    // Flatten 0-D Tensor.
    core::Tensor src_t = core::Tensor::Init<float>(3, device);
    core::Tensor dst_t = core::Tensor::Init<float>({3}, device);

    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten()));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1)));

    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0, 0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0, -1)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1, 0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1, -1)));

    EXPECT_ANY_THROW(src_t.Flatten(-2));
    EXPECT_ANY_THROW(src_t.Flatten(1));
    EXPECT_ANY_THROW(src_t.Flatten(0, -2));
    EXPECT_ANY_THROW(src_t.Flatten(0, 1));

    // Flatten 1-D Tensor.
    src_t = core::Tensor::Init<float>({1, 2, 3}, device);
    dst_t = core::Tensor::Init<float>({1, 2, 3}, device);

    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten()));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1)));

    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0, 0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(0, -1)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1, 0)));
    EXPECT_TRUE(dst_t.AllEqual(src_t.Flatten(-1, -1)));

    EXPECT_ANY_THROW(src_t.Flatten(-2));
    EXPECT_ANY_THROW(src_t.Flatten(1));
    EXPECT_ANY_THROW(src_t.Flatten(0, -2));
    EXPECT_ANY_THROW(src_t.Flatten(0, 1));

    // Flatten 2-D Tensor.
    src_t = core::Tensor::Init<float>({{1, 2, 3}, {4, 5, 6}}, device);
    core::Tensor dst_t_flat =
            core::Tensor::Init<float>({1, 2, 3, 4, 5, 6}, device);
    core::Tensor dst_t_unchanged =
            core::Tensor::Init<float>({{1, 2, 3}, {4, 5, 6}}, device);

    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten()));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-2)));

    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0, 1)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-2, 1)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0, -1)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-2, -1)));

    EXPECT_TRUE(dst_t_unchanged.AllEqual(src_t.Flatten(1)));
    EXPECT_TRUE(dst_t_unchanged.AllEqual(src_t.Flatten(-1)));

    for (int64_t dim : {-2, -1, 0, 1}) {
        EXPECT_TRUE(dst_t_unchanged.AllEqual(src_t.Flatten(dim, dim)));
    }

    // Out of bounds dimensions.
    EXPECT_ANY_THROW(src_t.Flatten(0, 2));
    EXPECT_ANY_THROW(src_t.Flatten(0, -3));
    EXPECT_ANY_THROW(src_t.Flatten(-3, 0));
    EXPECT_ANY_THROW(src_t.Flatten(2, 0));

    // end_dim is greater than start_dim.
    EXPECT_ANY_THROW(src_t.Flatten(1, 0));
    EXPECT_ANY_THROW(src_t.Flatten(-1, 0));
    EXPECT_ANY_THROW(src_t.Flatten(1, -2));
    EXPECT_ANY_THROW(src_t.Flatten(-1, -2));

    // Flatten 3-D Tensor.
    src_t = core::Tensor::Init<float>(
            {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, device);
    dst_t_flat = core::Tensor::Init<float>(
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
    dst_t_unchanged = core::Tensor::Init<float>(
            {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, device);
    core::Tensor dst_t_first_two_flat = core::Tensor::Init<float>(
            {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}, device);
    core::Tensor dst_t_last_two_flat = core::Tensor::Init<float>(
            {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}}, device);

    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten()));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-3)));

    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0, 2)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-3, 2)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(0, -1)));
    EXPECT_TRUE(dst_t_flat.AllEqual(src_t.Flatten(-3, -1)));

    EXPECT_TRUE(dst_t_first_two_flat.AllEqual(src_t.Flatten(0, 1)));
    EXPECT_TRUE(dst_t_first_two_flat.AllEqual(src_t.Flatten(0, -2)));
    EXPECT_TRUE(dst_t_first_two_flat.AllEqual(src_t.Flatten(-3, 1)));
    EXPECT_TRUE(dst_t_first_two_flat.AllEqual(src_t.Flatten(-3, -2)));

    EXPECT_TRUE(dst_t_last_two_flat.AllEqual(src_t.Flatten(1, 2)));
    EXPECT_TRUE(dst_t_last_two_flat.AllEqual(src_t.Flatten(1, -1)));
    EXPECT_TRUE(dst_t_last_two_flat.AllEqual(src_t.Flatten(-2, 2)));
    EXPECT_TRUE(dst_t_last_two_flat.AllEqual(src_t.Flatten(-2, -1)));

    for (int64_t dim : {-3, -2, -1, 0, 1, 2}) {
        EXPECT_TRUE(dst_t_unchanged.AllEqual(src_t.Flatten(dim, dim)));
    }

    // Out of bounds dimensions.
    EXPECT_ANY_THROW(src_t.Flatten(0, 3));
    EXPECT_ANY_THROW(src_t.Flatten(0, -4));
    EXPECT_ANY_THROW(src_t.Flatten(-4, 0));
    EXPECT_ANY_THROW(src_t.Flatten(3, 0));

    // end_dim is greater than start_dim.
    EXPECT_ANY_THROW(src_t.Flatten(1, 0));
    EXPECT_ANY_THROW(src_t.Flatten(2, 0));
    EXPECT_ANY_THROW(src_t.Flatten(2, 1));
}

TEST_P(TensorPermuteDevices, DefaultStrides) {
    core::Device device = GetParam();

    core::Tensor t0({}, core::Float32, device);
    EXPECT_EQ(t0.GetShape(), core::SizeVector{});
    EXPECT_EQ(t0.GetStrides(), core::SizeVector{});
}

TEST_P(TensorPermuteSizesDefaultStridesAndDevices, DefaultStrides) {
    core::SizeVector shape;
    core::SizeVector expected_strides;
    std::tie(shape, expected_strides) = std::get<0>(GetParam());

    core::Device device = std::get<1>(GetParam());
    core::Tensor t(shape, core::Float32, device);
    EXPECT_EQ(t.GetStrides(), expected_strides);
}

TEST_P(TensorPermuteDevices, OperatorSquareBrackets) {
    core::Device device = GetParam();

    // Zero dim
    EXPECT_THROW(core::Tensor({}, core::Float32)[0], std::runtime_error);
    EXPECT_THROW(core::Tensor({}, core::Float32)[-1], std::runtime_error);
    EXPECT_THROW(core::Tensor({}, core::Float32)[2], std::runtime_error);

    // Index out-of-bounds
    EXPECT_THROW(core::Tensor({0, 1}, core::Float32)[0], std::runtime_error);
    EXPECT_THROW(core::Tensor({0, 1}, core::Float32)[-1], std::runtime_error);
    EXPECT_THROW(core::Tensor({1, 2}, core::Float32)[10], std::runtime_error);

    // Regular cases
    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    core::Tensor t_0 = t[0];
    EXPECT_EQ(t_0.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_0.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_0.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_0.GetBlob(), t.GetBlob());

    t_0 = t[-2];  // t[-2] == t[0]
    EXPECT_EQ(t_0.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_0.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_0.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_0.GetBlob(), t.GetBlob());

    core::Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());

    t_1 = t[-1];  // t[-1] == t[1]
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());

    core::Tensor t_1_2 = t[1][2];
    EXPECT_EQ(t_1_2.GetShape(), core::SizeVector({4}));
    EXPECT_EQ(t_1_2.GetStrides(), core::SizeVector({1}));
    EXPECT_EQ(t_1_2.GetDataPtr(), static_cast<char *>(t.GetDataPtr()) +
                                          (1 * 3 * 4 + 2 * 4) * sizeof(float));
    EXPECT_EQ(t_1_2.GetBlob(), t.GetBlob());

    core::Tensor t_1_2_3 = t[1][2][3];
    EXPECT_EQ(t_1_2_3.GetShape(), core::SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetStrides(), core::SizeVector({}));
    EXPECT_EQ(t_1_2_3.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) +
                      (1 * 3 * 4 + 2 * 4 + 3) * sizeof(float));
    EXPECT_EQ(t_1_2_3.GetBlob(), t.GetBlob());
}

TEST_P(TensorPermuteDevices, Item) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    core::Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);

    core::Tensor t_1 = t[1];
    EXPECT_THROW(t_1.Item<float>(), std::runtime_error);

    core::Tensor t_1_2 = t[1][2];
    EXPECT_THROW(t_1_2.Item<float>(), std::runtime_error);

    core::Tensor t_1_2_3 = t[1][2][3];
    EXPECT_THROW(t_1_2_3.Item<int32_t>(), std::runtime_error);
    EXPECT_EQ(t_1_2_3.Item<float>(), 23.f);
}

TEST_P(TensorPermuteDevices, ItemBool) {
    core::Device device = GetParam();

    std::vector<bool> vals{true, true, false};
    core::Tensor t(vals, {3}, core::Bool, device);

    core::Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);
    EXPECT_THROW(t_0.Item<uint8_t>(), std::runtime_error);

    EXPECT_EQ(t[0].Item<bool>(), true);
    EXPECT_EQ(t[1].Item<bool>(), true);
    EXPECT_EQ(t[2].Item<bool>(), false);
}

TEST_P(TensorPermuteDevices, ItemAssign) {
    core::Device device = GetParam();
    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

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
    core::Device device = GetParam();
    core::Tensor t;

    // 0D
    t = core::Tensor::Ones({}, core::Float32, device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false), R"(1.0)");
    t = core::Tensor::Full({}, std::numeric_limits<float>::quiet_NaN(),
                           core::Float32, device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false), R"(nan)");
    t = core::Tensor::Full({}, std::numeric_limits<double>::quiet_NaN(),
                           core::Float32, device);  // Casting
    EXPECT_EQ(t.ToString(/*with_suffix=*/false), R"(nan)");

    // 1D float
    t = core::Tensor(std::vector<float>{0, 1, 2, 3, 4}, {5}, core::Float32,
                     device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false), R"([0.0 1.0 2.0 3.0 4.0])");

    // 1D int
    std::vector<int32_t> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    t = core::Tensor(vals, {24}, core::Int32, device);
    EXPECT_EQ(
            t.ToString(/*with_suffix=*/false),
            R"([0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23])");

    // 2D
    t = core::Tensor(vals, {6, 4}, core::Int32, device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false),
              R"([[0 1 2 3],
 [4 5 6 7],
 [8 9 10 11],
 [12 13 14 15],
 [16 17 18 19],
 [20 21 22 23]])");

    // 3D
    t = core::Tensor(vals, {2, 3, 4}, core::Int32, device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false),
              R"([[[0 1 2 3],
  [4 5 6 7],
  [8 9 10 11]],
 [[12 13 14 15],
  [16 17 18 19],
  [20 21 22 23]]])");

    // 4D
    t = core::Tensor(vals, {2, 3, 2, 2}, core::Int32, device);
    EXPECT_EQ(t.ToString(/*with_suffix=*/false),
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
    EXPECT_EQ(t.ToString(/*with_suffix=*/false),
              R"([[True False True],
 [True False False]])");
}

TEST_P(TensorPermuteDevicePairs, CopyContiguous) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);
    EXPECT_TRUE(t.IsContiguous());

    core::Tensor t_0 = t[0];
    EXPECT_THROW(t_0.Item<float>(), std::runtime_error);
    EXPECT_TRUE(t_0.IsContiguous());

    core::Tensor t_1 = t[1];
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(),
              static_cast<char *>(t.GetDataPtr()) + 1 * 3 * 4 * sizeof(float));
    EXPECT_NE(t_1.GetDataPtr(), t_1.GetBlob()->GetDataPtr());
    EXPECT_TRUE(t_1.IsContiguous());

    core::Tensor t_1_copy = t_1.To(dst_device, /*copy=*/true);
    EXPECT_EQ(t_1_copy.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(t_1_copy.GetStrides(), core::SizeVector({4, 1}));
    EXPECT_EQ(t_1_copy.GetDataPtr(),
              t_1_copy.GetBlob()->GetDataPtr());  // Points to beginning of Blob
}

TEST_P(TensorPermuteDevices, Slice) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    const void *blob_head = t.GetBlob()->GetDataPtr();
    EXPECT_EQ(t.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(t.GetStrides(), core::SizeVector({12, 4, 1}));
    EXPECT_EQ(t.GetDataPtr(), blob_head);

    // t_1 = t[0:2:1], effectively not sliced
    core::Tensor t_1 = t.Slice(0, 0, 2, 1);
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({12, 4, 1}));
    EXPECT_EQ(t_1.GetDataPtr(), blob_head);
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,  5,  6,  7,
                                  8,  9,  10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23}));

    // t_2 = t[0:2:1][:, 0:3:2, :]
    core::Tensor t_2 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2);
    EXPECT_EQ(t_2.GetShape(), core::SizeVector({2, 2, 4}));
    EXPECT_EQ(t_2.GetStrides(), core::SizeVector({12, 8, 1}));
    EXPECT_EQ(t_2.GetDataPtr(), blob_head);
    EXPECT_EQ(t_2.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20,
                                  21, 22, 23}));

    // t_3 = [0:2:1, 0:3:2, 0:4:2]
    core::Tensor t_3 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_EQ(t_3.GetShape(), core::SizeVector({2, 2, 2}));
    EXPECT_EQ(t_3.GetStrides(), core::SizeVector({12, 8, 2}));
    EXPECT_EQ(t_3.GetDataPtr(), blob_head);
    EXPECT_EQ(t_3.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // t_4 = t[1, 0:3:2, 0:4:2], a mix of [] and slice
    core::Tensor t_4 = t[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    EXPECT_EQ(t_4.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(t_4.GetStrides(), core::SizeVector({8, 2}));
    EXPECT_EQ(t_4.GetDataPtr(), static_cast<const char *>(blob_head) +
                                        core::Float32.ByteSize() * 3 * 4);
    EXPECT_EQ(t_4.ToFlatVector<float>(), std::vector<float>({12, 14, 20, 22}));

    // t_5 = t[1, 0:-1, 0:-2:2] == t[1, 0:2, 0:2:2]
    core::Tensor t_5 = t[1].Slice(0, 0, -1).Slice(1, 0, -2, 2);
    EXPECT_EQ(t_5.GetShape(), core::SizeVector({2, 1}));
    EXPECT_EQ(t_5.GetStrides(), core::SizeVector({4, 2}));
    EXPECT_EQ(t_5.GetDataPtr(), static_cast<const char *>(blob_head) +
                                        core::Float32.ByteSize() * 3 * 4);
    EXPECT_EQ(t_5.ToFlatVector<float>(), std::vector<float>({12, 16}));
}

TEST_P(TensorPermuteDevices, GetItem) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    // t_1 = t[1, :3, 0:-1:2], effectively not sliced
    core::Tensor t_1 =
            t.GetItem({core::TensorKey::Index(1),
                       core::TensorKey::Slice(core::None, 3, core::None),
                       core::TensorKey::Slice(0, -1, 2)});
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({3, 2}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({4, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({12, 14, 16, 18, 20, 22}));
}

TEST_P(TensorPermuteDevices, GetItemAdvancedIndexing) {
    core::Device device = GetParam();
    core::Tensor t = core::Tensor::Init<float>(
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            device);

    // t_1 = t[[0, 1, 1, 2, 3, 5, 8, 13, 21]]
    core::Tensor index_tensor =
            core::Tensor::Init<int64_t>({0, 1, 1, 2, 3, 5, 8, 13, 21}, device);
    core::Tensor t_1 = t.GetItem(core::TensorKey::IndexTensor(index_tensor));
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({9}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 1, 1, 2, 3, 5, 8, 13, 21}));
}

TEST_P(TensorPermuteDevices, GetItemAdvancedIndexingMixed) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    // t_1 = t[1, 0:2, [1, 2]]
    core::Tensor index_tensor(std::vector<int64_t>{1, 2}, {2}, core::Int64,
                              device);

    core::Tensor t_1 = t.GetItem({core::TensorKey::Index(1),
                                  core::TensorKey::Slice(0, 2, core::None),
                                  core::TensorKey::IndexTensor(index_tensor)});
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({2, 1}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({13, 17, 14, 18}));
}

TEST_P(TensorPermuteDevices, SetItemAdvancedIndexing) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
            device);

    // t[[1, 3]] = np.array([100, 300])
    core::Tensor index_tensor(std::vector<int64_t>{1, 3}, {2}, core::Int64,
                              device);
    core::Tensor fill_tensor(std::vector<float>{100, 300}, {2}, core::Float32,
                             device);
    t.SetItem(core::TensorKey::IndexTensor(index_tensor), fill_tensor);
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0,  100, 2,  300, 4,  5,  6,  7,
                                  8,  9,   10, 11,  12, 13, 14, 15,
                                  16, 17,  18, 19,  20, 21, 22, 23}));
}

TEST_P(TensorPermuteDevices, SetItemAdvancedIndexingMixed) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    // t[1, 0:2, [1, 2]] = np.array([[100, 200], [300, 400]])
    core::Tensor index_tensor(std::vector<int64_t>{1, 2}, {2}, core::Int64,
                              device);
    core::Tensor fill_tensor(std::vector<float>{100, 200, 300, 400}, {2, 2},
                             core::Float32, device);
    t.SetItem({core::TensorKey::Index(1),
               core::TensorKey::Slice(0, 2, core::None),
               core::TensorKey::IndexTensor(index_tensor)},
              fill_tensor);
    EXPECT_EQ(t.ToFlatVector<float>(),
              std::vector<float>({0,  1,   2,   3,  4,  5,   6,   7,
                                  8,  9,   10,  11, 12, 100, 300, 15,
                                  16, 200, 400, 19, 20, 21,  22,  23}));
}

TEST_P(TensorPermuteDevices, SliceAssign) {
    core::Device device = GetParam();

    core::Tensor dst = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    // Assigning a contiguous core::Tensor to lvalue
    // src_0 == [[120, 140], [200, 220]]
    core::Tensor src_0(std::vector<float>({120, 140, 200, 220}), {2, 2},
                       core::Float32, device);
    core::Tensor dst_slice = dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    dst_slice.AsRvalue() = src_0;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 120, 13, 140, 15,
                                  16, 17, 18, 19, 200, 21, 220, 23}));

    // Assigning a contiguous core::Tensor to rvalue
    // src_1 == [[121, 141], [201, 221]]
    core::Tensor src_1(std::vector<float>({121, 141, 201, 221}), {2, 2},
                       core::Float32, device);
    dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2) = src_1;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 121, 13, 141, 15,
                                  16, 17, 18, 19, 201, 21, 221, 23}));

    // Assigning a non-contiguous core::Tensor to lvalue
    // src_2 == [[122, 142], [202, 222]]
    core::Tensor src_2_tmp(std::vector<float>({122, 142, -1, -1, 202, 222}),
                           {3, 2}, core::Float32,
                           device);                    // Shape (3, 2)
    core::Tensor src_2 = src_2_tmp.Slice(0, 0, 3, 2);  // Shape (2, 2)
    dst_slice.AsRvalue() = src_2;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 122, 13, 142, 15,
                                  16, 17, 18, 19, 202, 21, 222, 23}));

    // Assigning a non-contiguous core::Tensor to rvalue
    // src_3 == [[123, 143], [203, 223]]
    core::Tensor src_3_tmp(std::vector<float>({123, 143, -1, -1, 203, 223}),
                           {3, 2}, core::Float32,
                           device);                    // Shape (3, 2)
    core::Tensor src_3 = src_3_tmp.Slice(0, 0, 3, 2);  // Shape (2, 2)
    dst[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2) = src_3;
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0,  1,  2,  3,  4,   5,  6,   7,
                                  8,  9,  10, 11, 123, 13, 143, 15,
                                  16, 17, 18, 19, 203, 21, 223, 23}));
}

TEST_P(TensorPermuteDevices, Append) {
    core::Device device = GetParam();

    core::Tensor self, other, output;

    // Appending 0-D to 0-D.
    self = core::Tensor::Init<float>(0, device);
    other = core::Tensor::Init<float>(1, device);

    // 0-D can be appended to 0-D along axis = null.
    output = self.Append(other);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>({0, 1}, device)));

    // 0-D can not be appended to 0-D along axis = 0, -1.
    EXPECT_ANY_THROW(self.Append(other, 0));
    EXPECT_ANY_THROW(self.Append(other, -1));

    // Same Shape.
    // Appending 1-D [3,] tensor to 1-D [4,].
    self = core::Tensor::Init<float>({0, 1, 2, 3}, device);
    other = core::Tensor::Init<float>({4, 5, 6}, device);

    // 1-D can be appended to 1-D along axis = null, 0, -1.
    output = self.Append(other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    output = self.Append(other, 0);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    output = self.Append(other, -1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6}, device)));

    // 1-D can not be appended to 1-D along axis = 1, -2.
    EXPECT_ANY_THROW(self.Append(other, 1));
    EXPECT_ANY_THROW(self.Append(other, -2));

    // Appending 2-D [2, 2] tensor to 2-D [2, 2].
    self = core::Tensor::Init<float>({{0, 1}, {2, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}, {6, 7}}, device);

    // 2-D tensor can be appended to 2-D tensor along axis = null, 0, 1, -1, -2.
    output = self.Append(other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5, 6, 7}, device)));

    output = self.Append(other, 0);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));

    output = self.Append(other, -2);
    EXPECT_TRUE(output.AllClose(core::Tensor::Init<float>(
            {{0, 1}, {2, 3}, {4, 5}, {6, 7}}, device)));

    output = self.Append(other, 1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1, 4, 5}, {2, 3, 6, 7}}, device)));

    output = self.Append(other, -1);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1, 4, 5}, {2, 3, 6, 7}}, device)));

    // 2-D can not be appended to 2-D along axis = 2, -3.
    EXPECT_ANY_THROW(self.Append(other, 2));
    EXPECT_ANY_THROW(self.Append(other, -3));

    // Appending 2-D [1, 2] tensor to 2-D [2, 2].
    self = core::Tensor::Init<float>({{0, 1}, {2, 3}}, device);
    other = core::Tensor::Init<float>({{4, 5}}, device);

    // Only the dimension along the axis can be different, so tensor of shape
    // [1, 2] can be appended to [2, 2] along axis = null, 0, -2.
    output = self.Append(other);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({0, 1, 2, 3, 4, 5}, device)));

    output = self.Append(other, 0);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1}, {2, 3}, {4, 5}}, device)));

    output = self.Append(other, -2);
    EXPECT_TRUE(output.AllClose(
            core::Tensor::Init<float>({{0, 1}, {2, 3}, {4, 5}}, device)));

    // [1, 2] can not be appended to [2, 2] along axis = 1, -1.
    EXPECT_ANY_THROW(self.Append(other, 1));
    EXPECT_ANY_THROW(self.Append(other, -1));

    // Dtype and Device of both the tensors must be same.
    // Taking the above case of [1, 2] to [2, 2] with different dtype and
    // device.
    EXPECT_ANY_THROW(self.Append(other.To(core::Float64)));
    if (device.IsCUDA()) {
        EXPECT_ANY_THROW(self.Append(other.To(core::Device("CPU:0"))));
    }
}

TEST_P(TensorPermuteDevicePairs, CopyNonContiguous) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);

    // t[0:2:1, 0:3:2, 0:4:2]
    core::Tensor t_1 = t.Slice(0, 0, 2, 1).Slice(1, 0, 3, 2).Slice(2, 0, 4, 2);
    EXPECT_FALSE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({2, 2, 2}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({12, 8, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));

    // Copy ensures contiguous
    {
        core::Tensor t_1_copy = t_1.To(src_device, /*copy=*/true);
        EXPECT_TRUE(t_1_copy.IsContiguous());
        EXPECT_EQ(t_1_copy.GetShape(), core::SizeVector({2, 2, 2}));
        EXPECT_EQ(t_1_copy.GetStrides(), core::SizeVector({4, 2, 1}));
        EXPECT_EQ(t_1_copy.ToFlatVector<float>(),
                  std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));
    }
    {
        core::Tensor t_1_copy = t_1.To(dst_device, /*copy=*/true);
        EXPECT_TRUE(t_1_copy.IsContiguous());
        EXPECT_EQ(t_1_copy.GetShape(), core::SizeVector({2, 2, 2}));
        EXPECT_EQ(t_1_copy.GetStrides(), core::SizeVector({4, 2, 1}));
        EXPECT_EQ(t_1_copy.ToFlatVector<float>(),
                  std::vector<float>({0, 2, 8, 10, 12, 14, 20, 22}));
    }
}

TEST_P(TensorPermuteDevicePairs, IndexGet) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    core::Tensor src_t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);

    // t[:, [1, 2], [1, 2]]
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         idx_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         idx_device)};

    core::Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));

    // Check 0-D tensor.
    src_t = core::Tensor::Init<int>(1, src_device);
    indices = {core::Tensor::Init<bool>(true, idx_device)};
    dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.AllClose(src_t));
    EXPECT_EQ(src_t.GetDtype(), dst_t.GetDtype());

    src_t = core::Tensor::Init<int>(1, src_device);
    indices = {core::Tensor::Init<bool>(false, idx_device)};
    dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.AllClose(core::Tensor({0}, core::Int32, src_device)));
    EXPECT_EQ(src_t.GetDtype(), dst_t.GetDtype());
}

TEST_P(TensorPermuteDevicePairs, IndexGetNegative) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);

    // t[:, [1, -1], [1, -2]]
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>({1, -1}), {2}, core::Int64,
                         idx_device),
            core::Tensor(std::vector<int64_t>({1, -2}), {2}, core::Int64,
                         idx_device)};

    core::Tensor t_1 = t.IndexGet(indices);
    EXPECT_TRUE(t_1.IsContiguous());
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(t_1.ToFlatVector<float>(), std::vector<float>({5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevicePairs, IndexGet2DBroadcastedIndex) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    core::Tensor src_t(vals, {2, 3, 4, 2}, core::Float32, src_device);

    // t[:, [[1], [0], [2]], [[0, 1], [2, 3], [0, 1]], :] to shape {2, 3, 2, 2}
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>({1, 0, 2}), {3, 1}, core::Int64,
                         idx_device),
            core::Tensor(std::vector<int64_t>({0, 1, 2, 3, 0, 1}), {3, 2},
                         core::Int64, idx_device),
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
    };

    core::Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({2, 3, 2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({8,  9,  10, 11, 4,  5,  6,  7,
                                  16, 17, 18, 19, 32, 33, 34, 35,
                                  28, 29, 30, 31, 40, 41, 42, 43}));
}

TEST_P(TensorPermuteDevicePairs, IndexGet2DBroadcastedIndexSplitBySlice) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    core::Tensor src_t(vals, {2, 3, 2, 4}, core::Float32, src_device);

    // t[:, [[1], [0], [2]], :, [[0, 1], [2, 3], [0, 1]]] to shape {3, 2, 2, 2}
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>({1, 0, 2}), {3, 1}, core::Int64,
                         idx_device),
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>({0, 1, 2, 3, 0, 1}), {3, 2},
                         core::Int64, idx_device),

    };

    core::Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({3, 2, 2, 2}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({8,  12, 32, 36, 9,  13, 33, 37,
                                  2,  6,  26, 30, 3,  7,  27, 31,
                                  16, 20, 40, 44, 17, 21, 41, 45}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetAssignToBroadcast) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    core::Tensor src_t(vals, {2, 3, 4}, core::Float32, src_device);

    // t[:, [1, 2], [1, 2]] to shape {2, 2}
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, dst_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         dst_device),
            core::Tensor(std::vector<int64_t>({1, 2}), {2}, core::Int64,
                         dst_device)};

    // Broadcast to shape {3, 2, 2}
    core::SizeVector dst_shape{3, 2, 2};
    core::Tensor dst_t(dst_shape, core::Float32, dst_device);
    dst_t.AsRvalue() =
            src_t.IndexGet(indices);  // Intermediate tensor copied internally

    EXPECT_TRUE(dst_t.IsContiguous());
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({3, 2, 2}));
    EXPECT_EQ(
            dst_t.ToFlatVector<float>(),
            std::vector<float>({5, 10, 17, 22, 5, 10, 17, 22, 5, 10, 17, 22}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetSeparateBySlice) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    core::Tensor src_t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);

    // t[[0, 1], :, [0, 1]]
    std::vector<core::Tensor> indices = {
            core::Tensor(std::vector<int64_t>{0, 1}, {2}, core::Int64,
                         idx_device),
            core::Tensor(core::SizeVector(), core::Int64, idx_device),
            core::Tensor(std::vector<int64_t>{0, 1}, {2}, core::Int64,
                         idx_device)};

    core::Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 4, 8, 13, 17, 21}));
}

TEST_P(TensorPermuteDevicePairs, IndexGetSliceEnd) {
    core::Device idx_device;
    core::Device src_device;
    std::tie(idx_device, src_device) = GetParam();

    core::Tensor src_t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            src_device);

    std::vector<core::Tensor> indices = {
            core::Tensor(std::vector<int64_t>{0, 1}, {2}, core::Int64,
                         idx_device),
            core::Tensor(std::vector<int64_t>{0, 1}, {2}, core::Int64,
                         idx_device),
            core::Tensor(core::SizeVector(), core::Int64, idx_device)};

    core::Tensor dst_t = src_t.IndexGet(indices);
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 16, 17, 18, 19}));
}

TEST_P(TensorPermuteDevicePairs, IndexSet) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> vals({4, 6, 5, 16, 18, 17});
    core::Tensor src_t(vals, {2, 3}, core::Float32, src_device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    core::Tensor dst_t(zeros, {2, 3, 4}, core::Float32, dst_device);

    // t[:, [1], [0, 2, 1]]
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, src_device),
            core::Tensor(std::vector<int64_t>({1}), {1}, core::Int64,
                         dst_device),
            core::Tensor(std::vector<int64_t>({0, 2, 1}), {3}, core::Int64,
                         src_device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 4,  5,  6,  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 16, 17, 18, 0, 0, 0, 0, 0}));

    // Check 0-D tensor.
    // dst_t[np.array(True)] = 10 // Works, assigned
    // dst_t[np.array(True)] = np.array(10) // Works, assigned
    dst_t = core::Tensor::Init<float>(5, dst_device);
    dst_t.IndexSet({core::Tensor::Init<bool>(true, src_device)},
                   core::Tensor::Init<float>(10, src_device));
    EXPECT_TRUE(dst_t.AllClose(core::Tensor::Init<float>(10, dst_device)));

    // dst_t[np.array(True)] = np.array([10]) // Works, assigned
    dst_t = core::Tensor::Init<float>(5, dst_device);
    dst_t.IndexSet({core::Tensor::Init<bool>(true, src_device)},
                   core::Tensor::Init<float>({10}, src_device));
    EXPECT_TRUE(dst_t.AllClose(core::Tensor::Init<float>(10, dst_device)));

    // dst_t[np.array(True)] = np.array([[10]]) // Cannot assign 2D
    dst_t = core::Tensor::Init<float>(5, dst_device);
    EXPECT_ANY_THROW(
            dst_t.IndexSet({core::Tensor::Init<bool>(true, src_device)},
                           core::Tensor::Init<float>({{10}}, src_device)));

    // dst_t[np.array(True)] = np.array([10, 11]) // Cannot assign 1+ values
    EXPECT_ANY_THROW(
            dst_t.IndexSet({core::Tensor::Init<bool>(true, src_device)},
                           core::Tensor::Init<float>({10, 11}, src_device)));

    // dst_t[np.array(False)] = 10 // Works, unchanged
    dst_t = core::Tensor::Init<float>(5, dst_device);
    dst_t.IndexSet({core::Tensor::Init<bool>(false, src_device)},
                   core::Tensor::Init<float>(10, src_device));
    EXPECT_TRUE(dst_t.AllClose(core::Tensor::Init<float>(5, dst_device)));

    // dst_t[np.array(False)] = np.array(10) // Works, unchanged
    dst_t = core::Tensor::Init<float>(5, dst_device);
    dst_t.IndexSet({core::Tensor::Init<bool>(false, src_device)},
                   core::Tensor::Init<float>({10}, src_device));
    EXPECT_TRUE(dst_t.AllClose(core::Tensor::Init<float>(5, dst_device)));

    // dst_t[np.array(False)] = np.array([10]) // Works, unchanged
    dst_t = core::Tensor::Init<float>(5, dst_device);
    EXPECT_ANY_THROW(
            dst_t.IndexSet({core::Tensor::Init<bool>(false, src_device)},
                           core::Tensor::Init<float>({{5}}, src_device)));

    // dst_t[np.array(False)] = np.array([[10]]) // Cannot assign 2D
    dst_t = core::Tensor::Init<float>(5, dst_device);
    EXPECT_ANY_THROW(
            dst_t.IndexSet({core::Tensor::Init<bool>(false, src_device)},
                           core::Tensor::Init<float>({{10}}, src_device)));

    // dst_t[np.array(False)] = np.array([10, 11]) // Cannot assign 1+ values
    EXPECT_ANY_THROW(
            dst_t.IndexSet({core::Tensor::Init<bool>(false, src_device)},
                           core::Tensor::Init<float>({10, 11}, src_device)));
}

TEST_P(TensorPermuteDevicePairs, IndexSetBroadcast) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    std::vector<float> src_vals({10, 20});
    core::Tensor src_t(src_vals, {2, 1}, core::Float32, src_device);

    std::vector<float> zeros(2 * 3 * 4, 0);
    core::Tensor dst_t(zeros, {2, 3, 4}, core::Float32, dst_device);

    // t[:, [1], [0, 2, 1]] -> slice {2, 3, 4} to {2, 3}
    std::vector<core::Tensor> indices = {
            core::Tensor(core::SizeVector(), core::Int64, src_device),
            core::Tensor(std::vector<int64_t>({1}), {1}, core::Int64,
                         dst_device),
            core::Tensor(std::vector<int64_t>({0, 2, 1}), {3}, core::Int64,
                         src_device)};

    dst_t.IndexSet(indices, src_t);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 20, 20, 20, 0, 0, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, Permute) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    core::Tensor t_1 = t.Permute({2, 1, 0});
    EXPECT_EQ(t_1.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_1.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_1.GetShape(), core::SizeVector({4, 3, 2}));
    EXPECT_EQ(t_1.GetStrides(), core::SizeVector({1, 4, 12}));
    EXPECT_EQ(t_1.ToFlatVector<float>(),
              std::vector<float>({0, 12, 4, 16, 8,  20, 1, 13, 5, 17, 9,  21,
                                  2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}));

    core::Tensor t_2 = t.Permute({0, 2, 1});
    EXPECT_EQ(t_2.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_2.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_2.GetShape(), core::SizeVector({2, 4, 3}));
    EXPECT_EQ(t_2.GetStrides(), core::SizeVector({12, 1, 4}));
    EXPECT_EQ(t_2.ToFlatVector<float>(),
              std::vector<float>({0,  4,  8,  1,  5,  9,  2,  6,
                                  10, 3,  7,  11, 12, 16, 20, 13,
                                  17, 21, 14, 18, 22, 15, 19, 23}));
}

TEST_P(TensorPermuteDevices, Transpose) {
    core::Device device = GetParam();

    core::Tensor t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);

    core::Tensor t_t = t.Transpose(1, 2);
    EXPECT_EQ(t_t.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_t.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_t.GetShape(), core::SizeVector({2, 4, 3}));
    EXPECT_EQ(t_t.GetStrides(), core::SizeVector({12, 1, 4}));
    EXPECT_EQ(t_t.ToFlatVector<float>(),
              std::vector<float>({0,  4,  8,  1,  5,  9,  2,  6,
                                  10, 3,  7,  11, 12, 16, 20, 13,
                                  17, 21, 14, 18, 22, 15, 19, 23}));
    EXPECT_THROW(t.Transpose(3, 5), std::runtime_error);
}

TEST_P(TensorPermuteDevices, T) {
    core::Device device = GetParam();

    std::vector<float> vals{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    core::Tensor t(vals, {6, 4}, core::Float32, device);

    core::Tensor t_t = t.T();
    EXPECT_EQ(t_t.GetBlob(), t.GetBlob());
    EXPECT_EQ(t_t.GetDataPtr(), t.GetDataPtr());
    EXPECT_EQ(t_t.GetShape(), core::SizeVector({4, 6}));
    EXPECT_EQ(t_t.GetStrides(), core::SizeVector({1, 4}));
    EXPECT_EQ(t_t.ToFlatVector<float>(),
              std::vector<float>({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                  2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));

    core::Tensor t_3d(vals, {2, 3, 4}, core::Float32, device);
    EXPECT_THROW(t_3d.T(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Det) {
    core::Device device = GetParam();
    // Det supports both Float32 and Float64.
    core::Dtype dtype = core::Float32;

    // Float32 test.
    core::Tensor A_3x3f = core::Tensor::Init<float>(
            {{-5, 0, -1}, {1, 2, -1}, {-3, 4, 1}}, device);

    double A_3x3f_det = A_3x3f.Det();
    EXPECT_DOUBLE_EQ(A_3x3f_det, -40.0);

    // Float64 test.
    core::Tensor A_3x3d = core::Tensor::Init<double>(
            {{-5, 0, -1}, {1, 2, -1}, {-3, 4, 1}}, device);
    double A_3x3d_det = A_3x3d.Det();
    EXPECT_DOUBLE_EQ(A_3x3d_det, -40.0);

    // Det expects a 2D square matrix [shape test].
    EXPECT_ANY_THROW(core::Tensor::Ones({0}, dtype, device).Det());
    EXPECT_ANY_THROW(core::Tensor::Ones({2, 2, 2}, dtype, device).Det());
    EXPECT_ANY_THROW(core::Tensor::Ones({3, 4}, dtype, device).Det());
}

TEST_P(TensorPermuteDevices, ShallowCopyConstructor) {
    core::Device device = GetParam();
    core::Tensor t({2, 3}, core::Float32, device);

    // Copy constructor.
    core::Tensor t_copy(t);
    EXPECT_EQ(t.GetDataPtr(), t_copy.GetDataPtr());

    // Vector initialization.
    std::vector<core::Tensor> t_vec0{t};
    EXPECT_EQ(t.GetDataPtr(), t_vec0[0].GetDataPtr());

    std::vector<core::Tensor> t_vec1({t});
    EXPECT_EQ(t.GetDataPtr(), t_vec1[0].GetDataPtr());

    // Vector initialization list passed to function.
    auto FirstTensorDataPtr =
            [](const std::vector<core::Tensor> &tensors) -> void * {
        return const_cast<void *>(tensors[0].GetDataPtr());
    };
    EXPECT_EQ(t.GetDataPtr(), FirstTensorDataPtr({t}));
}

TEST_P(TensorPermuteDevices, AdvancedIndexing_IsIndexSplittedBySlice) {
    core::Device device = GetParam();

    core::Tensor idx = core::Tensor::Init<int64_t>({1, 2}, device);
    core::Tensor slice(core::Tensor(core::SizeVector(), core::Int64, device));

    EXPECT_FALSE(
            core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice({slice}));
    EXPECT_FALSE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {slice, idx}));
    EXPECT_FALSE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {idx, slice}));
    EXPECT_FALSE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {slice, idx, idx}));
    EXPECT_FALSE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {slice, idx, idx, slice}));

    EXPECT_TRUE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {idx, slice, idx}));
    EXPECT_TRUE(core::AdvancedIndexPreprocessor::IsIndexSplittedBySlice(
            {idx, slice, slice, idx}));
}

TEST_P(TensorPermuteDevicesWithSYCL, Add) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{10, 11, 12}, {13, 14, 15}}, device);
    core::Tensor c = a + b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({10, 12, 14, 16, 18, 20}));
}

TEST_P(TensorPermuteDevices, Add_) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{10, 11, 12}, {13, 14, 15}}, device);
    a += b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({10, 12, 14, 16, 18, 20}));
}

TEST_P(TensorPermuteDevices, Add_BroadcastException) {
    // A.shape = (   3, 4)
    // B.shape = (2, 3, 4)
    // A += B should throw exception.
    // B += A is fine.
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>(
            {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}}, device);
    core::Tensor b = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);
    EXPECT_THROW(a += b, std::runtime_error);
    b += a;
    EXPECT_EQ(b.ToFlatVector<float>(),
              std::vector<float>({0,  2,  4,  6,  8,  10, 12, 14,
                                  16, 18, 20, 22, 12, 14, 16, 18,
                                  20, 22, 24, 26, 28, 30, 32, 34}));
}

TEST_P(TensorPermuteDevices, Sub) {
    core::Device device = GetParam();
    core::Tensor a =
            core::Tensor::Init<float>({{10, 12, 14}, {16, 18, 20}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    core::Tensor c = a - b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({10, 11, 12, 13, 14, 15}));
}

TEST_P(TensorPermuteDevices, Sub_) {
    core::Device device = GetParam();
    core::Tensor a =
            core::Tensor::Init<float>({{10, 12, 14}, {16, 18, 20}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    a -= b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({10, 11, 12, 13, 14, 15}));
}

TEST_P(TensorPermuteDevices, Mul) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{6, 7, 8}, {9, 10, 11}}, device);
    core::Tensor c = a * b;
    EXPECT_EQ(c.ToFlatVector<float>(),
              std::vector<float>({0, 7, 16, 27, 40, 55}));
}

TEST_P(TensorPermuteDevices, Mul_) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1, 2}, {3, 4, 5}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{6, 7, 8}, {9, 10, 11}}, device);
    a *= b;
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>({0, 7, 16, 27, 40, 55}));
}

TEST_P(TensorPermuteDevices, Div) {
    core::Device device = GetParam();
    core::Tensor a =
            core::Tensor::Init<float>({{0, 7, 16}, {27, 40, 55}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{6, 7, 8}, {9, 10, 11}}, device);
    core::Tensor c = a / b;
    EXPECT_EQ(c.ToFlatVector<float>(), std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, Div_) {
    core::Device device = GetParam();
    core::Tensor a =
            core::Tensor::Init<float>({{0, 7, 16}, {27, 40, 55}}, device);
    core::Tensor b =
            core::Tensor::Init<float>({{6, 7, 8}, {9, 10, 11}}, device);
    a /= b;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, ReduceSumKeepDim) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({{{22.f, 23.f, 20.f, 9.f},
                                                   {6.f, 14.f, 18.f, 13.f},
                                                   {15.f, 3.f, 17.f, 0.f}},
                                                  {{7.f, 21.f, 11.f, 1.f},
                                                   {4.f, 2.f, 10.f, 19.f},
                                                   {5.f, 8.f, 16.f, 12.f}}},
                                                 device);
    core::Tensor dst;

    dst = src.Sum({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Sum({0}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({29.f, 44.f, 31.f, 10.f, 10.f, 16.f, 28.f,
                                  32.f, 20.f, 11.f, 33.f, 12.f}));

    dst = src.Sum({1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 1, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {43.f, 40.f, 55.f, 22.f, 16.f, 31.f, 37.f, 32.f}));

    dst = src.Sum({2}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({74.f, 51.f, 35.f, 40.f, 35.f, 41.f}));

    dst = src.Sum({0, 1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 1, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({59.f, 71.f, 92.f, 54.f}));

    dst = src.Sum({0, 2}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    dst = src.Sum({1, 2}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    dst = src.Sum({0, 1, 2}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({276.f}));

    // Dim order does not matter: {2, 1} -> {1, 2}.
    dst = src.Sum({2, 1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    // Dim should be wrapped automatically: {-1, 0} -> {2, 0} -> {0, 2}.
    dst = src.Sum({-1, 0}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 3, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    // Exception cases.
    EXPECT_THROW(src.Sum({5}, true), std::runtime_error);      // Out-of-range.
    EXPECT_THROW(src.Sum({0, -4}, true), std::runtime_error);  // Out-of-range.
    EXPECT_THROW(src.Sum({0, 0}, true), std::runtime_error);   // Repeated.
    EXPECT_THROW(src.Sum({2, -1}, true), std::runtime_error);  // Repeated.
}

TEST_P(TensorPermuteDevices, ReduceSumNotKeepDim) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({{{22.f, 23.f, 20.f, 9.f},
                                                   {6.f, 14.f, 18.f, 13.f},
                                                   {15.f, 3.f, 17.f, 0.f}},
                                                  {{7.f, 21.f, 11.f, 1.f},
                                                   {4.f, 2.f, 10.f, 19.f},
                                                   {5.f, 8.f, 16.f, 12.f}}},
                                                 device);
    core::Tensor dst;

    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({29.f, 44.f, 31.f, 10.f, 10.f, 16.f, 28.f,
                                  32.f, 20.f, 11.f, 33.f, 12.f}));

    dst = src.Sum({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {43.f, 40.f, 55.f, 22.f, 16.f, 31.f, 37.f, 32.f}));

    dst = src.Sum({2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({74.f, 51.f, 35.f, 40.f, 35.f, 41.f}));

    dst = src.Sum({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({59.f, 71.f, 92.f, 54.f}));

    dst = src.Sum({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({114.f, 86.f, 76.f}));

    dst = src.Sum({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({160.f, 116.f}));

    dst = src.Sum({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({276.f}));
}

TEST_P(TensorPermuteDevices, ReduceSumSpecialShapes) {
    core::Device device = GetParam();
    core::Tensor src;
    core::Tensor dst;

    // np.sum(np.ones(()), axis=(), keepdims=*)
    src = core::Tensor::Ones({}, core::Float32, device);
    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1}));
    dst = src.Sum({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1}));

    // np.sum(np.ones(()), axis=(out_of_bound), keepdims=*)
    EXPECT_THROW(dst.Sum({0}, false), std::runtime_error);
    EXPECT_THROW(dst.Sum({-1}, false), std::runtime_error);
    EXPECT_THROW(dst.Sum({-1}, true), std::runtime_error);
    EXPECT_THROW(dst.Sum({1}, false), std::runtime_error);
    EXPECT_THROW(dst.Sum({1}, true), std::runtime_error);

    // Empty reduction axis ().
    // This reduces no axis, which is different from reduce all axis.
    // np.sum(np.ones((0)), axis=(), keepdims=*)
    src = core::Tensor::Ones({0}, core::Float32, device);
    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));
    dst = src.Sum({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));

    // np.sum(np.ones((0)), axis=(0,), keepdims=*), fill with identity
    src = core::Tensor::Ones({0}, core::Float32, device);
    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));  // 1D becomes 0D.
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0}));
    dst = src.Sum({0}, true);
    EXPECT_EQ(dst.GetShape(),
              core::SizeVector({1}));  // Remains 1D, but with size 1.
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0}));

    // np.sum(np.ones((0, 2)), axis=(), keepdims=*)
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0, 2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));
    dst = src.Sum({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0, 2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));

    // np.sum(np.ones((0, 2)), axis=(0,), keepdims=*), fill with identity
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0, 0}));
    dst = src.Sum({0}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0, 0}));

    // np.sum(np.ones((0, 2)), axis=(1,), keepdims=*)
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Sum({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));
    dst = src.Sum({1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));

    // np.sum(np.ones((0, 2)), axis=(0, 1), keepdims=*), fill with identity
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Sum({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0}));
    dst = src.Sum({0, 1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0}));
}

TEST_P(TensorPermuteDevices, ReduceMultipleOutputsSumLargeArray) {
    core::Device device = GetParam();
    core::SizeVector shape{3, 7, 8234719};
    int64_t size = shape.NumElements();
    std::vector<int> vals(size, 1);
    core::Tensor src(vals, shape, core::Int32, device);
    core::Tensor dst;

    dst = src.Sum({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 7, 8234719}));
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>(3 * 7 * 8234719, 1));

    dst = src.Sum({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({7, 8234719}));
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>(7 * 8234719, 3));
}

TEST_P(TensorPermuteDevices, ReduceSum64bit1D) {
    core::Device device = GetParam();
    // num_bytes = 8 * (2 ^ 28) + 1 = 2 ^ 31 + 1 ~= 2GB
    // max_offsets = num_bytes - 1 = 2 ^ 31
    // max_32_bit_indexing = 2 ^ 31 - 1
    // max_offsets > max_32_bit_indexing
    int64_t num_elements = (1ULL << 28) + 10;
    std::vector<int64_t> vals(num_elements, 1);
    core::Tensor src(vals, {num_elements}, core::Int64, device);
    core::Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(1, num_elements));
}

// np.sum(np.ones((2, large_dim)), dim=0)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase0) {
    core::Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    core::SizeVector shape{2, large_dim};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    core::Tensor src(vals, shape, core::Int64, device);
    core::Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({large_dim}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(large_dim, 2));

    core::Tensor src_sliced = src.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Slice(30, large_dim, core::None)});
    EXPECT_EQ(src_sliced.GetShape(), core::SizeVector({2, large_dim - 30}));
    dst = src_sliced.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({large_dim - 30}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(large_dim - 30, 2));
}

// np.sum(np.ones((2, large_dim)), dim=1)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase1) {
    core::Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    core::SizeVector shape{2, large_dim};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    core::Tensor src(vals, shape, core::Int64, device);
    core::Tensor dst;

    dst = src.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(2, large_dim));

    core::Tensor src_sliced = src.GetItem(
            {core::TensorKey::Slice(core::None, core::None, core::None),
             core::TensorKey::Slice(30, large_dim, core::None)});
    EXPECT_EQ(src_sliced.GetShape(), core::SizeVector({2, large_dim - 30}));
    dst = src_sliced.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(2, large_dim - 30));
}

// np.sum(np.ones((large_dim, 2)), dim=0)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase2) {
    core::Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    core::SizeVector shape{large_dim, 2};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    core::Tensor src(vals, shape, core::Int64, device);
    core::Tensor dst;

    dst = src.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(2, large_dim));

    core::Tensor src_sliced = src.GetItem(
            {core::TensorKey::Slice(30, large_dim, core::None),
             core::TensorKey::Slice(core::None, core::None, core::None)});
    EXPECT_EQ(src_sliced.GetShape(), core::SizeVector({large_dim - 30, 2}));
    dst = src_sliced.Sum({0}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(2, large_dim - 30));
}

// np.sum(np.ones((large_dim, 2)), dim=1)
TEST_P(TensorPermuteDevices, ReduceSum64bit2DCase3) {
    core::Device device = GetParam();
    int64_t large_dim = (1ULL << 27) + 10;
    core::SizeVector shape{large_dim, 2};
    int64_t num_elements = shape.NumElements();
    std::vector<int64_t> vals(num_elements, 1);
    core::Tensor src(vals, shape, core::Int64, device);
    core::Tensor dst;

    dst = src.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({large_dim}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>(large_dim, 2));

    core::Tensor src_sliced = src.GetItem(
            {core::TensorKey::Slice(30, large_dim, core::None),
             core::TensorKey::Slice(core::None, core::None, core::None)});
    EXPECT_EQ(src_sliced.GetShape(), core::SizeVector({large_dim - 30, 2}));
    dst = src_sliced.Sum({1}, /*keepdim=*/false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({large_dim - 30}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>(large_dim - 30, 2));
}

TEST_P(TensorPermuteDevices, ReduceSumLargeArray) {
    core::Device device = GetParam();

    std::vector<int64_t> sizes = TensorSizes::TestCases();
    int64_t max_size = *std::max_element(sizes.begin(), sizes.end());
    std::vector<int> vals(max_size);
    std::transform(vals.begin(), vals.end(), vals.begin(), [](int x) -> int {
        return utility::UniformRandIntGenerator(0, 3)();
    });

    for (int64_t size : sizes) {
        int ref_result = std::accumulate(vals.begin(), vals.begin() + size, 0,
                                         std::plus<int>());
        core::Tensor src(std::vector<int>(vals.begin(), vals.begin() + size),
                         {size}, core::Int32, device);
        core::Tensor dst = src.Sum({0}, false);

        EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
        EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>({ref_result}));
    }
}

TEST_P(TensorPermuteDevices, ReduceProd) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({{{22.f, 23.f, 20.f, 9.f},
                                                   {6.f, 14.f, 18.f, 13.f},
                                                   {15.f, 3.f, 17.f, 0.f}},
                                                  {{7.f, 21.f, 11.f, 1.f},
                                                   {4.f, 2.f, 10.f, 19.f},
                                                   {5.f, 8.f, 16.f, 12.f}}},
                                                 device);
    core::Tensor dst;

    dst = src.Prod({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Prod({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({154.f, 483.f, 220.f, 9.f, 24.f, 28.f, 180.f,
                                  247.f, 75.f, 24.f, 272.f, 0.f}));

    dst = src.Prod({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({1980.f, 966.f, 6120.f, 0.f, 140.f, 336.f,
                                  1760.f, 228.f}));

    dst = src.Prod({2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {91080.f, 19656.f, 0.f, 1617.f, 1520.f, 7680.f}));

    dst = src.Prod({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({277200.f, 324576.f, 10771200.f, 0.f}));

    dst = src.Prod({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({147276360.f, 29877120.f, 0.f}));

    dst = src.Prod({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0.f, 18876211200.f}));

    dst = src.Prod({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f}));
}

TEST_P(TensorPermuteDevices, ReduceMin) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({{{22.f, 23.f, 20.f, 9.f},
                                                   {6.f, 14.f, 18.f, 13.f},
                                                   {15.f, 3.f, 17.f, 0.f}},
                                                  {{7.f, 21.f, 11.f, 1.f},
                                                   {4.f, 2.f, 10.f, 19.f},
                                                   {5.f, 8.f, 16.f, 12.f}}},
                                                 device);
    core::Tensor dst;

    dst = src.Min({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Min({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({7.f, 21.f, 11.f, 1.f, 4.f, 2.f, 10.f, 13.f,
                                  5.f, 3.f, 16.f, 0.f}));

    dst = src.Min({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({6.f, 3.f, 17.f, 0.f, 4.f, 2.f, 10.f, 1.f}));

    dst = src.Min({2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({9.f, 6.f, 0.f, 1.f, 2.f, 5.f}));

    dst = src.Min({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({4.f, 2.f, 10.f, 0.f}));

    dst = src.Min({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1.f, 2.f, 0.f}));

    dst = src.Min({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f, 1.f}));

    dst = src.Min({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({0.f}));
}

TEST_P(TensorPermuteDevices, ReduceMax) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({{{22.f, 23.f, 20.f, 9.f},
                                                   {6.f, 14.f, 18.f, 13.f},
                                                   {15.f, 3.f, 17.f, 0.f}},
                                                  {{7.f, 21.f, 11.f, 1.f},
                                                   {4.f, 2.f, 10.f, 19.f},
                                                   {5.f, 8.f, 16.f, 12.f}}},
                                                 device);
    core::Tensor dst;

    dst = src.Max({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f,  14.f,
                                  18.f, 13.f, 15.f, 3.f, 17.f, 0.f,
                                  7.f,  21.f, 11.f, 1.f, 4.f,  2.f,
                                  10.f, 19.f, 5.f,  8.f, 16.f, 12.f}));

    dst = src.Max({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 9.f, 6.f, 14.f, 18.f, 19.f,
                                  15.f, 8.f, 17.f, 12.f}));

    dst = src.Max({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>(
                      {22.f, 23.f, 20.f, 13.f, 7.f, 21.f, 16.f, 19.f}));

    dst = src.Max({2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({23.f, 18.f, 17.f, 21.f, 19.f, 16.f}));

    dst = src.Max({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({4}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({22.f, 23.f, 20.f, 19.f}));

    dst = src.Max({0, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({23.f, 19.f, 17.f}));

    dst = src.Max({1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({23.f, 21.f}));

    dst = src.Max({0, 1, 2}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({23.f}));
}

TEST_P(TensorPermuteDevices, ReduceMaxFloatLimit) {
    // std::numeric_limits<scalar_t> should use lowest() instead of min().
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>({-2.f, -1.f}, device);

    core::Tensor dst = src.Max({0});
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({-1.f}));

    dst = src.ArgMax({0});
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>({1}));
}

TEST_P(TensorPermuteDevices, ReduceArgMin) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>(
            {{{22, 23, 20, 9}, {6, 14, 18, 13}, {15, 3, 17, 0}},
             {{7, 21, 11, 1}, {4, 2, 10, 19}, {5, 8, 16, 12}}},
            device);
    core::Tensor dst;

    dst = src.ArgMin({0, 1, 2});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>({11}));

    dst = src.ArgMin({0});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0}));

    dst = src.ArgMin({1});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 2, 2, 2, 1, 1, 1, 0}));

    dst = src.ArgMin({2});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({3, 0, 3, 3, 1, 0}));
}

TEST_P(TensorPermuteDevices, ReduceArgMax) {
    core::Device device = GetParam();
    core::Tensor src = core::Tensor::Init<float>(
            {{{22, 23, 20, 9}, {6, 14, 18, 13}, {15, 3, 17, 0}},
             {{7, 21, 11, 1}, {4, 2, 10, 19}, {5, 8, 16, 12}}},
            device);
    core::Tensor dst;

    dst = src.ArgMax({0, 1, 2});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(), std::vector<int64_t>({1}));

    dst = src.ArgMax({0});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1}));

    dst = src.ArgMax({1});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 4}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({0, 0, 0, 1, 0, 0, 2, 1}));

    dst = src.ArgMax({2});
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 2, 2, 1, 3, 2}));
}

TEST_P(TensorPermuteDevices, Sqrt) {
    core::Device device = GetParam();
    core::Tensor src =
            core::Tensor::Init<float>({{0, 1, 4}, {9, 16, 25}}, device);
    core::Tensor dst = src.Sqrt();
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));

    // Sqrt only works for float types, throws exception otherwise.
    src = core::Tensor({2, 3}, core::Int32, device);
    EXPECT_THROW(src.Sqrt(), std::runtime_error);

    // Negative number's sqrt shall be NaN.
    src = core::Tensor::Init<float>({{0, 1, 4}, {9, -16, -25}}, device);
    dst = src.Sqrt();
    std::vector<float> dst_vals = dst.ToFlatVector<float>();
    EXPECT_EQ(dst_vals[0], 0);
    EXPECT_EQ(dst_vals[1], 1);
    EXPECT_EQ(dst_vals[2], 2);
    EXPECT_EQ(dst_vals[3], 3);
    EXPECT_TRUE(std::isnan(dst_vals[4]));
    EXPECT_TRUE(std::isnan(dst_vals[5]));

    // Inplace version.
    src = core::Tensor::Init<float>({{0, 1, 4}, {9, 16, 25}}, device);
    src.Sqrt_();
    EXPECT_EQ(src.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST_P(TensorPermuteDevices, Sin) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::sin(v); });
    core::Tensor dst_ref(dst_vals, {2, 3}, core::Float32, device);

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Sin();
    EXPECT_TRUE(dst.AllClose(dst_ref));

    // Inplace version.
    src.Sin_();
    EXPECT_TRUE(src.AllClose(dst_ref));

    // Only works for float types, throws exception otherwise.
    src = core::Tensor({2, 3}, core::Int32, device);
    EXPECT_THROW(src.Sin(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Cos) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::cos(v); });
    core::Tensor dst_ref(dst_vals, {2, 3}, core::Float32, device);

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Cos();
    EXPECT_TRUE(dst.AllClose(dst_ref));

    // Inplace version.
    src.Cos_();
    EXPECT_TRUE(src.AllClose(dst_ref));

    // Only works for float types, throws exception otherwise.
    src = core::Tensor({2, 3}, core::Int32, device);
    EXPECT_THROW(src.Cos(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Neg) {
    core::Device device = GetParam();

    std::vector<float> dst_vals{2, 1, 0, -1, -2, -3};
    core::Tensor src =
            core::Tensor::Init<float>({{-2, -1, 0}, {1, 2, 3}}, device);
    core::Tensor dst = src.Neg();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Neg_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);

    // Also works for int.
    src = core::Tensor(std::vector<int>{-1, 0, 2}, {1, 3}, core::Int32, device);
    dst = src.Neg();
    EXPECT_EQ(dst.ToFlatVector<int>(), std::vector<int>({1, 0, -2}));
}

TEST_P(TensorPermuteDevices, Exp) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::exp(v); });
    core::Tensor dst_ref(dst_vals, {2, 3}, core::Float32, device);

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Exp();
    EXPECT_TRUE(dst.AllClose(dst_ref));

    // Inplace version.
    src.Exp_();
    EXPECT_TRUE(src.AllClose(dst_ref));

    // Only works for float types, throws exception otherwise.
    src = core::Tensor({2, 3}, core::Int32, device);
    EXPECT_THROW(src.Exp(), std::runtime_error);
}

TEST_P(TensorPermuteDevices, Abs) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2, -1, 0, 1, 2, 3};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::abs(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Abs();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);

    // Inplace version.
    src.Abs_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, IsNan) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-INFINITY, NAN, 0, NAN, 2, INFINITY};
    std::vector<bool> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> bool { return std::isnan(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.IsNan();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_vals);
}

TEST_P(TensorPermuteDevices, IsInf) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-INFINITY, NAN, 0, NAN, 2, INFINITY};
    std::vector<bool> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> bool { return std::isinf(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.IsInf();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_vals);
}

TEST_P(TensorPermuteDevices, IsFinite) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-INFINITY, NAN, 0, NAN, 2, INFINITY};
    std::vector<bool> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> bool { return std::isfinite(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.IsFinite();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Floor) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2.4, -1.6, 0, 1.4, 2.6, 3.5};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::floor(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Floor();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Ceil) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2.4, -1.6, 0, 1.4, 2.6, 3.5};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::ceil(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Ceil();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Round) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2.4, -1.6, 0, 1.4, 2.6, 3.5};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::round(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Round();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, Trunc) {
    core::Device device = GetParam();

    std::vector<float> src_vals{-2.4, -1.6, 0, 1.4, 2.6, 3.5};
    std::vector<float> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](float v) -> float { return std::trunc(v); });

    core::Tensor src(src_vals, {2, 3}, core::Float32, device);
    core::Tensor dst = src.Trunc();
    EXPECT_EQ(dst.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalNot) {
    core::Device device = GetParam();

    std::vector<bool> src_vals{true, false, true, false};
    std::vector<bool> dst_vals;
    std::transform(src_vals.begin(), src_vals.end(),
                   std::back_inserter(dst_vals),
                   [](bool v) -> bool { return !static_cast<bool>(v); });

    core::Tensor src(src_vals, {2, 2}, core::Bool, device);
    core::Tensor dst = src.LogicalNot();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_vals);

    // Inplace version.
    src.LogicalNot_();
    EXPECT_EQ(src.ToFlatVector<bool>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalNotFloat) {
    core::Device device = GetParam();

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

    core::Tensor src(src_vals, {2, 2}, core::Float32, device);
    core::Tensor dst = src.LogicalNot();
    EXPECT_EQ(dst.ToFlatVector<bool>(), dst_bool_vals);

    // Inplace version.
    src.LogicalNot_();
    EXPECT_EQ(src.ToFlatVector<float>(), dst_vals);
}

TEST_P(TensorPermuteDevices, LogicalAnd) {
    core::Device device = GetParam();
    core::Tensor a(std::vector<bool>({true, false, true, false}), {2, 2},
                   core::Bool, device);
    core::Tensor b(std::vector<bool>({true, true, false, false}), {2, 2},
                   core::Bool, device);
    core::Tensor c = a.LogicalAnd(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{-1, 0}, {1, 0}}, device);
    core::Tensor b = core::Tensor::Init<float>({{1, 0}, {0, 0}}, device);
    core::Tensor c = a.LogicalAnd(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, false, false, false}));

    // Inplace version.
    a.LogicalAnd_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 0, 0, 0}));
}

TEST_P(TensorPermuteDevices, LogicalOr) {
    core::Device device = GetParam();
    core::Tensor a(std::vector<bool>({true, false, true, false}), {2, 2},
                   core::Bool, device);
    core::Tensor b(std::vector<bool>({true, true, false, false}), {2, 2},
                   core::Bool, device);
    core::Tensor c = a.LogicalOr(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{-1, 0}, {1, 0}}, device);
    core::Tensor b = core::Tensor::Init<float>({{1, -1}, {0, 0}}, device);
    core::Tensor c = a.LogicalOr(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({true, true, true, false}));

    // Inplace version.
    a.LogicalOr_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 1, 1, 0}));
}

TEST_P(TensorPermuteDevices, LogicalXor) {
    core::Device device = GetParam();
    core::Tensor a(std::vector<bool>({true, false, true, false}), {2, 2},
                   core::Bool, device);
    core::Tensor b(std::vector<bool>({true, true, false, false}), {2, 2},
                   core::Bool, device);
    core::Tensor c = a.LogicalXor(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));

    // Inplace version.
    a.LogicalXor_(b);
    EXPECT_EQ(a.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));
}

TEST_P(TensorPermuteDevices, LogicalXorFloat) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{-1, 0}, {1, 0}}, device);
    core::Tensor b = core::Tensor::Init<float>({{1, -1}, {0, 0}}, device);
    core::Tensor c = a.LogicalXor(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, false}));

    // Inplace version.
    a.LogicalXor_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 1, 0}));
}

TEST_P(TensorPermuteDevices, Gt) {
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Gt(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Lt(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Ge(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Le(b);
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
    core::Device device = GetParam();
    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Eq(b);
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
    core::Device device = GetParam();

    core::Tensor a = core::Tensor::Init<float>({{0, 1}, {-1, 1}}, device);
    core::Tensor b = core::Tensor::Init<float>({{0, 0}, {0, 2}}, device);
    core::Tensor c = a.Ne(b);
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, true}));
    c = a != b;
    EXPECT_EQ(c.ToFlatVector<bool>(),
              std::vector<bool>({false, true, true, true}));

    // Inplace version.
    a.Ne_(b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 1, 1, 1}));
}

TEST_P(TensorPermuteDevices, BooleanIndex) {
    core::Device device = GetParam();

    // a[a < 0] = 0
    core::Tensor a = core::Tensor::Init<float>({1, -1, -2, 3}, device);
    core::Tensor b = core::Tensor::Init<float>({0}, device);
    a.SetItem(core::TensorKey::IndexTensor(a.Le(b)), b);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({1, 0, 0, 3}));

    // x = np.array([[0, 1], [1, 1], [2, 2]])
    // row_sum = np.array([1, 2, 4])
    // y = x[row_sum <= 2, :]
    core::Tensor x =
            core::Tensor::Init<float>({{0, 1}, {1, 1}, {2, 2}}, device);
    core::Tensor row_sum = core::Tensor::Init<float>({1, 2, 4}, device);
    core::Tensor two = core::Tensor::Init<float>({2}, device);
    core::Tensor y = x.GetItem(
            {core::TensorKey::IndexTensor(row_sum.Le(two)),
             core::TensorKey::Slice(core::None, core::None, core::None)});
    EXPECT_EQ(y.ToFlatVector<float>(), std::vector<float>({0, 1, 1, 1}));
    EXPECT_EQ(y.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(y.GetDtype(), core::Float32);
}

TEST_P(TensorPermuteDevices, NonZeroNumpy) {
    core::Device device = GetParam();

    core::Tensor a =
            core::Tensor::Init<float>({{0, 1}, {1, 0}, {1, 0}}, device);
    std::vector<core::Tensor> results = a.NonZeroNumpy();
    EXPECT_EQ(results[0].ToFlatVector<int64_t>(),
              std::vector<int64_t>({0, 1, 2}));
    EXPECT_EQ(results[1].ToFlatVector<int64_t>(),
              std::vector<int64_t>({1, 0, 0}));
    EXPECT_EQ(results[0].GetShape(), core::SizeVector{3});
    EXPECT_EQ(results[1].GetShape(), core::SizeVector{3});
}

TEST_P(TensorPermuteDevices, CreationEmpty) {
    core::Device device = GetParam();

    core::Tensor a = core::Tensor::Empty({}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({}));
    EXPECT_EQ(a.NumElements(), 1);

    a = core::Tensor::Empty({0}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(a.NumElements(), 0);

    a = core::Tensor::Empty({1}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({1}));
    EXPECT_EQ(a.NumElements(), 1);

    a = core::Tensor::Empty({0, 1}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({0, 1}));
    EXPECT_EQ(a.NumElements(), 0);

    a = core::Tensor::Empty({2, 3}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
}

TEST_P(TensorPermuteDevices, CreationFull) {
    core::Device device = GetParam();

    const float fill_value = 100;
    core::Tensor a = core::Tensor::Full({}, fill_value, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({}));
    EXPECT_EQ(a.NumElements(), 1);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = core::Tensor::Full({0}, fill_value, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(a.NumElements(), 0);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = core::Tensor::Full({1}, fill_value, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({1}));
    EXPECT_EQ(a.NumElements(), 1);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = core::Tensor::Full({0, 1}, fill_value, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({0, 1}));
    EXPECT_EQ(a.NumElements(), 0);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));

    a = core::Tensor::Full({2, 3}, fill_value, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(),
              std::vector<float>(a.NumElements(), fill_value));
}

TEST_P(TensorPermuteDevices, CreationZeros) {
    core::Device device = GetParam();

    core::Tensor a = core::Tensor::Zeros({2, 3}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>(a.NumElements(), 0));
}

TEST_P(TensorPermuteDevices, CreationOnes) {
    core::Device device = GetParam();

    core::Tensor a = core::Tensor::Ones({2, 3}, core::Float32, device);
    EXPECT_EQ(a.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(a.NumElements(), 6);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>(a.NumElements(), 1));
}

TEST_P(TensorPermuteDevices, ScalarOperatorOverload) {
    core::Device device = GetParam();
    core::Tensor a;
    core::Tensor b;

    // +
    a = core::Tensor::Ones({2}, core::Float32, device);
    b = a.Add(1);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = a + 1;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = 1 + a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));
    b = a + true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));

    // +=
    a = core::Tensor::Ones({2}, core::Float32, device);
    a.Add_(1);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({2, 2}));
    a += 1;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({3, 3}));
    a += true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({4, 4}));

    // -
    a = core::Tensor::Ones({2}, core::Float32, device);
    b = a.Sub(1);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));
    b = a - 1;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));
    b = 10 - a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({9, 9}));
    b = a - true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0, 0}));

    // -=
    a = core::Tensor::Ones({2}, core::Float32, device);
    a.Sub_(1);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({0, 0}));
    a -= 1;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({-1, -1}));
    a -= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({-2, -2}));

    // *
    a = core::Tensor::Full({2}, 2, core::Float32, device);
    b = a.Mul(10);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = a * 10;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = 10 * a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));
    b = a * true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({2, 2}));

    // *=
    a = core::Tensor::Full({2}, 2, core::Float32, device);
    a.Mul_(10);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({20, 20}));
    a *= 10;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({200, 200}));
    a *= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({200, 200}));

    // /
    a = core::Tensor::Full({2}, 20, core::Float32, device);
    b = a.Div(2);
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({10, 10}));
    b = a / 2;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({10, 10}));
    b = 10 / a;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({0.5, 0.5}));
    b = a / true;
    EXPECT_EQ(b.ToFlatVector<float>(), std::vector<float>({20, 20}));

    // /=
    a = core::Tensor::Full({2}, 20, core::Float32, device);
    a.Div_(2);
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({10, 10}));
    a /= 2;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({5, 5}));
    a /= true;
    EXPECT_EQ(a.ToFlatVector<float>(), std::vector<float>({5, 5}));
}

TEST_P(TensorPermuteDevices, ReduceMean) {
    core::Device device = GetParam();
    core::Tensor src;
    core::Tensor dst;

    // Only Float32 and Float64 supports Mean.
    src = core::Tensor::Ones({2, 3}, core::Int64, device);
    EXPECT_THROW(src.Mean({}), std::runtime_error);

    // Input shape {2, 3}, not keepdim.
    src = core::Tensor(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3},
                       core::Float32, device);
    dst = src.Mean({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));
    dst = src.Mean({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({3}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1.5, 2.5, 3.5}));
    dst = src.Mean({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1, 4}));

    // Input shape {2, 3}, keepdim.
    src = core::Tensor(std::vector<float>({0, 1, 2, 3, 4, 5}), {2, 3},
                       core::Float32, device);
    dst = src.Mean({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(),
              std::vector<float>({0, 1, 2, 3, 4, 5}));
    dst = src.Mean({0}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 3}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1.5, 2.5, 3.5}));
    dst = src.Mean({1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1, 4}));

    // Input shape {}, one element, not keepdim.
    src = core::Tensor::Ones({}, core::Float32, device);
    dst = src.Mean({}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1}));
    EXPECT_THROW(src.Mean({0}, false), std::runtime_error);

    // Input shape {}, one element, keepdim.
    src = core::Tensor::Ones({}, core::Float32, device);
    dst = src.Mean({}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({1}));
    EXPECT_THROW(src.Mean({0}, true), std::runtime_error);

    // Input shape {0}, not keepdim.
    src = core::Tensor::Ones({0}, core::Float32, device);
    dst = src.Mean({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));  // 1D becomes 0D.
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));

    // Input shape {0}, keepdim.
    src = core::Tensor::Ones({0}, core::Float32, device);
    dst = src.Mean({0}, true);
    EXPECT_EQ(dst.GetShape(),
              core::SizeVector({1}));  // 1D, filled with identity.
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));

    // Input shape {0, 2}, not keepdim.
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Mean({0}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({2}));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[1]));
    dst = src.Mean({1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));
    dst = src.Mean({0, 1}, false);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({}));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));

    // Input shape {0, 2}, keepdim.
    src = core::Tensor::Ones({0, 2}, core::Float32, device);
    dst = src.Mean({0}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 2}));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[1]));
    dst = src.Mean({1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({0, 1}));
    EXPECT_EQ(dst.ToFlatVector<float>(), std::vector<float>({}));
    dst = src.Mean({0, 1}, true);
    EXPECT_EQ(dst.GetShape(), core::SizeVector({1, 1}));
    EXPECT_TRUE(std::isnan(dst.ToFlatVector<float>()[0]));
}

TEST_P(TensorPermuteDevices, ToDLPackFromDLPack) {
    core::Device device = GetParam();
    core::Tensor src_t = core::Tensor::Init<float>(
            {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
             {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}},
            device);
    const void *blob_head = src_t.GetBlob()->GetDataPtr();

    // src_t = src_t[1, 0:3:2, 0:4:2], a mix of [] and slice
    src_t = src_t[1].Slice(0, 0, 3, 2).Slice(1, 0, 4, 2);
    EXPECT_EQ(src_t.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(src_t.GetStrides(), core::SizeVector({8, 2}));
    EXPECT_EQ(src_t.GetBlob()->GetDataPtr(), blob_head);
    EXPECT_EQ(src_t.GetDataPtr(), static_cast<const char *>(blob_head) +
                                          core::Float32.ByteSize() * 3 * 4);
    EXPECT_EQ(src_t.ToFlatVector<float>(),
              std::vector<float>({12, 14, 20, 22}));

    DLManagedTensor *dl_t = src_t.ToDLPack();

    core::Tensor dst_t = core::Tensor::FromDLPack(dl_t);
    EXPECT_EQ(dst_t.GetShape(), core::SizeVector({2, 2}));
    EXPECT_EQ(dst_t.GetStrides(), core::SizeVector({8, 2}));
    // Note that the original blob head's info has been discarded.
    EXPECT_EQ(dst_t.GetBlob()->GetDataPtr(),
              static_cast<const char *>(blob_head) +
                      core::Float32.ByteSize() * 3 * 4);
    EXPECT_EQ(dst_t.GetDataPtr(), static_cast<const char *>(blob_head) +
                                          core::Float32.ByteSize() * 3 * 4);
    EXPECT_EQ(dst_t.ToFlatVector<float>(),
              std::vector<float>({12, 14, 20, 22}));
}

TEST_P(TensorPermuteDevices, IsSame) {
    core::Device device = GetParam();

    // "Shallow" copy.
    core::Tensor t0 = core::Tensor::Ones({6, 8}, core::Float32, device);
    core::Tensor t1 = t0;  // "Shallow" copy
    EXPECT_TRUE(t0.IsSame(t1));
    EXPECT_TRUE(t1.IsSame(t0));

    // Copy constructor copies view.
    core::Tensor t0_copy_construct(t0);
    EXPECT_TRUE(t0.IsSame(t0_copy_construct));
    EXPECT_TRUE(t0_copy_construct.IsSame(t0));

    // New tensor of the same value.
    core::Tensor t2 = core::Tensor::Ones({6, 8}, core::Float32, device);
    EXPECT_FALSE(t0.IsSame(t2));
    EXPECT_FALSE(t2.IsSame(t0));

    // Tensor::Contiguous() does not copy if already contiguous.
    core::Tensor t0_contiguous = t0.Contiguous();
    EXPECT_TRUE(t0.IsSame(t0_contiguous));
    EXPECT_TRUE(t0_contiguous.IsSame(t0));

    // Slices are views.
    core::Tensor t0_slice =
            t0.GetItem({core::TensorKey::Slice(0, 5, 2)})[0];  // t0[0:5:2][0]
    core::Tensor t1_slice = t1[0];
    EXPECT_TRUE(t0_slice.IsSame(t1_slice));
    EXPECT_TRUE(t1_slice.IsSame(t0_slice));

    // Explicit copy to the same device.
    core::Tensor t0_copy = t0.Clone();
    EXPECT_FALSE(t0.IsSame(t0_copy));
    EXPECT_FALSE(t0_copy.IsSame(t0));

    // std::vector<Tensor> initializer list and push_back() are views.
    std::vector<core::Tensor> vec{t0};
    vec.push_back(t1);
    EXPECT_TRUE(t0.IsSame(vec[0]));
    EXPECT_TRUE(t1.IsSame(vec[1]));
    EXPECT_TRUE(vec[0].IsSame(vec[1]));
}

TEST_P(TensorPermuteDevices, RValueScalar) {
    const core::Device &device = GetParam();
    core::Tensor t, t_ref;

    // Check with shape {}.
    t = core::Tensor::Init<int32_t>(0, device);
    t_ref = core::Tensor::Init<int32_t>(1000, device);
    t.AsRvalue() = 1000;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with shape {0}.
    t = core::Tensor::Init<bool>({}, device);
    t_ref = core::Tensor::Init<bool>({}, device);
    t.AsRvalue() = 0;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with shape {1, 0}.
    t = core::Tensor::Init<int32_t>({{}}, device);
    t_ref = core::Tensor::Init<int32_t>({{}}, device);
    t.AsRvalue() = 10;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with shape {1}.
    t = core::Tensor::Init<float>({20.30}, device);
    t_ref = core::Tensor::Init<float>({-10.10}, device);
    t.AsRvalue() = -10.10;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with shape {1, 1}.
    t = core::Tensor::Init<uint8_t>({{20}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{10}}, device);
    t.AsRvalue() = 10;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with shape {1, 2}.
    t = core::Tensor::Init<uint8_t>({{20, 10}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{0, 0}}, device);
    t.AsRvalue() = 0;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with indexing.
    t = core::Tensor::Init<bool>({{true, true}, {true, true}}, device);
    t_ref = core::Tensor::Init<bool>({{false, false}, {true, true}}, device);
    t[0] = 0;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with implicit conversion.
    t = core::Tensor::Init<int32_t>({{5, 6}, {7, 8}}, device);
    t_ref = core::Tensor::Init<int32_t>({{10, 10}, {10, 10}}, device);
    t.AsRvalue() = 10.2f;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with Slice.
    t = core::Tensor::Init<uint8_t>({1}, device);
    t_ref = core::Tensor::Init<uint8_t>({255}, device);
    t.Slice(0, 0, 1) = 255;
    EXPECT_TRUE(t.AllClose(t_ref));

    // Datatype implicit conversion with Slice.
    t = core::Tensor::Init<bool>({{false, false}}, device);
    t_ref = core::Tensor::Init<bool>({{true, true}}, device);
    t.Slice(1, 0, 2) = 1.0f;
    EXPECT_TRUE(t.AllClose(t_ref));
}

TEST_P(TensorPermuteDevices, Clip) {
    core::Device device = GetParam();
    core::Tensor t, t_clip, t_ref;

    // Check with float tensor.
    t = core::Tensor::Init<float>({{0, -1, 1, 4, 1000}}, device);
    t_ref = core::Tensor::Init<float>({{0, 0, 1, 4, 5.2}}, device);
    t_clip = t.Clip(0, 5.2);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check with uint8 tensor.
    t = core::Tensor::Init<uint8_t>({{0, 255, 30, 49, 100}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{20, 40, 30, 40, 40}}, device);
    t_clip = t.Clip(20, 40);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check with uint8 tensor and min max values in float.
    t = core::Tensor::Init<uint8_t>({{0, 255, 30, 49, 100}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{20, 40, 30, 40, 40}}, device);
    t_clip = t.Clip(20.3, 40.6);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check with Integer.
    t = core::Tensor::Init<int32_t>({{0, 3000, 30, 49, 500}}, device);
    t_ref = core::Tensor::Init<int32_t>({{21, 49, 30, 49, 49}}, device);
    t_clip = t.Clip(21, 49);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check with Integer and min max values in double.
    t = core::Tensor::Init<int32_t>({{0, 3000, 30, 49, 500}}, device);
    t_ref = core::Tensor::Init<int32_t>({{20, 49, 30, 49, 49}}, device);
    t_clip = t.Clip(20.3, 49.01);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check error with boolean tensor.
    t = core::Tensor::Init<bool>({{false, true, true, false, false}}, device);
    EXPECT_THROW(t_clip = t.Clip(1, 1.3), std::runtime_error);

    // Check when min value is greater than max value.
    t = core::Tensor::Init<float>({{0, -1, 1, 4, 1000}}, device);
    t_ref = core::Tensor::Init<float>({{2.0, 2.0, 2.0, 2.0, 2.0}}, device);
    t_clip = t.Clip(5.2, 2.0);
    EXPECT_TRUE(t_clip.AllClose(t_ref));

    // Check with large int64_t value.
    t = core::Tensor::Init<int64_t>({{9223372036854775807, -1, 1, 4, 1000}},
                                    device);
    t_ref = core::Tensor::Init<int64_t>({{9223372036854775807, 5, 5, 5, 1000}},
                                        device);
    t_clip = t.Clip(5.2, 9223372036854775807);
    EXPECT_TRUE(t_clip.AllClose(t_ref));
}

TEST_P(TensorPermuteDevices, Clip_) {
    core::Device device = GetParam();
    core::Tensor t, t_ref;

    // Check with float tensor.
    t = core::Tensor::Init<float>({{0, -1, 1, 4, 1000}}, device);
    t_ref = core::Tensor::Init<float>({{0, 0, 1, 4, 5.2}}, device);
    t.Clip_(0, 5.2);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with uint8 tensor.
    t = core::Tensor::Init<uint8_t>({{0, 255, 30, 49, 100}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{20, 40, 30, 40, 40}}, device);
    t.Clip_(20, 40);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with uint8 tensor and min max values in float.
    t = core::Tensor::Init<uint8_t>({{0, 255, 30, 49, 100}}, device);
    t_ref = core::Tensor::Init<uint8_t>({{20, 40, 30, 40, 40}}, device);
    t.Clip_(20.3, 40.6);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with Integer.
    t = core::Tensor::Init<int32_t>({{0, 3000, 30, 49, 500}}, device);
    t_ref = core::Tensor::Init<int32_t>({{21, 49, 30, 49, 49}}, device);
    t.Clip_(21, 49);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with Integer and min max values in double.
    t = core::Tensor::Init<int32_t>({{0, 3000, 30, 49, 500}}, device);
    t_ref = core::Tensor::Init<int32_t>({{20, 49, 30, 49, 49}}, device);
    t.Clip_(20.3, 49.01);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check error with boolean tensor.
    t = core::Tensor::Init<bool>({{false, true, true, false, false}}, device);
    EXPECT_THROW(t.Clip_(1, 1.3), std::runtime_error);

    // Check when min value is greater than max value.
    t = core::Tensor::Init<float>({{0, -1, 1, 4, 1000}}, device);
    t_ref = core::Tensor::Init<float>({{2.0, 2.0, 2.0, 2.0, 2.0}}, device);
    t.Clip_(5.2, 2.0);
    EXPECT_TRUE(t.AllClose(t_ref));

    // Check with large int64_t value.
    t = core::Tensor::Init<int64_t>({{9223372036854775807, -1, 1, 4, 1000}},
                                    device);
    t_ref = core::Tensor::Init<int64_t>({{9223372036854775807, 5, 5, 5, 1000}},
                                        device);
    t.Clip_(5.2, 9223372036854775807);
    EXPECT_TRUE(t.AllClose(t_ref));
}

TEST_P(TensorPermuteDevicePairs, AllEqual) {
    core::Device device_a;
    core::Device device_b;
    std::tie(device_a, device_b) = GetParam();

    core::Tensor src;
    core::Tensor dst;

    // Normal case.
    src = core::Tensor::Init<float>({0, 1, 2}, device_a);
    dst = core::Tensor::Init<float>({0, 1, 2.5}, device_a);
    EXPECT_FALSE(src.AllEqual(dst));

    src = core::Tensor::Init<float>({0, 1, 2}, device_a);
    dst = core::Tensor::Init<float>({0, 1, 2}, device_a);
    EXPECT_TRUE(src.AllEqual(dst));

    // Different device.
    src = core::Tensor::Init<float>({0, 1, 2}, device_a);
    dst = core::Tensor::Init<float>({0, 1, 2}, device_b);
    if (device_a != device_b) {
        EXPECT_ANY_THROW(src.AllEqual(dst));
    } else {
        EXPECT_TRUE(src.AllEqual(dst));
    }

    // Different dtype.
    src = core::Tensor::Init<float>({0, 1, 2}, device_a);
    dst = core::Tensor::Init<int>({0, 1, 2}, device_a);
    EXPECT_ANY_THROW(src.AllEqual(dst));

    // Different shape.
    src = core::Tensor::Init<float>({0, 1, 2}, device_a);
    dst = core::Tensor::Init<float>({{0, 1, 2}}, device_a);
    EXPECT_FALSE(src.AllEqual(dst));
}

TEST_P(TensorPermuteDevices, Iterator) {
    core::Device device = GetParam();

    core::Tensor t;
    std::vector<core::Tensor> t_slices;  // Ground-truth slices.
    int index = 0;

    // operator*() -> const core::Tensor &. Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (const core::Tensor &t_slice : t) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }

    // operator*() -> core::Tensor. Assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor t_slice : t) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }
    for (core::Tensor t_slice : t) {
        t_slice.AsRvalue() = 10;
    }
    EXPECT_TRUE(t.AllEqual(core::Tensor::Init<int>({10, 10, 10}, device)));

    // operator*() -> const core::Tensor &&. Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (const core::Tensor &&t_slice : t) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }

    // operator*() -> core::Tensor &&. Assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor &&t_slice : t) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }
    for (core::Tensor &&t_slice : t) {
        t_slice.AsRvalue() = 10;
    }
    EXPECT_TRUE(t.AllEqual(core::Tensor::Init<int>({10, 10, 10}, device)));

    // operator->(). Assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor::Iterator iter = t.begin(); iter != t.end(); ++iter) {
        EXPECT_TRUE(iter->IsSame(t_slices[index]));
        index++;
    }
    for (core::Tensor::Iterator iter = t.begin(); iter != t.end(); ++iter) {
        iter->AsRvalue() = 10;
    }
    EXPECT_TRUE(t.AllEqual(core::Tensor::Init<int>({10, 10, 10}, device)));

    // 0-d.
    t = core::Tensor::Init<int>(10, device);
    EXPECT_ANY_THROW(t.begin());

    // 2-d.
    t = core::Tensor::Init<int>({{0, 1, 2}, {3, 4, 5}}, device);
    t_slices = {t[0], t[1]};
    index = 0;
    for (const core::Tensor &t_slice : t) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }
}

TEST_P(TensorPermuteDevices, ConstIterator) {
    core::Device device = GetParam();

    core::Tensor t;
    std::vector<core::Tensor> t_slices;  // Ground-truth slices.
    int index = 0;

    // operator*() -> const core::Tensor &. Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (const core::Tensor &t_slice : AsConst(t)) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }

    // operator*() -> core::Tensor. Assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor t_slice : AsConst(t)) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }
    for (core::Tensor t_slice : AsConst(t)) {
        t_slice.AsRvalue() = 10;
    }
    EXPECT_TRUE(t.AllEqual(core::Tensor::Init<int>({10, 10, 10}, device)));

    // operator*() -> const core::Tensor &&. Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (const core::Tensor &&t_slice : AsConst(t)) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }

    // operator->() with cbegin() and cend(). Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor::ConstIterator iter = t.cbegin(); iter != t.cend();
         ++iter) {
        EXPECT_TRUE(iter->IsSame(t_slices[index]));
        index++;
    }

    // operator->() with overloaded begin() and end(). Not assignable.
    t = core::Tensor::Init<int>({0, 1, 2}, device);
    const core::Tensor &t_const = t;
    t_slices = {t[0], t[1], t[2]};
    index = 0;
    for (core::Tensor::ConstIterator iter = t_const.begin();
         iter != t_const.end(); ++iter) {
        EXPECT_TRUE(iter->IsSame(t_slices[index]));
        index++;
    }

    // 0-d.
    t = core::Tensor::Init<int>(10, device);
    EXPECT_ANY_THROW(t.begin());

    // 2-d.
    t = core::Tensor::Init<int>({{0, 1, 2}, {3, 4, 5}}, device);
    t_slices = {t[0], t[1]};
    index = 0;
    for (const core::Tensor &t_slice : AsConst(t)) {
        EXPECT_TRUE(t_slice.IsSame(t_slices[index]));
        index++;
    }
}

TEST_P(TensorPermuteDevices, TakeOwnership) {
    core::Device device = GetParam();
    if (!device.IsCPU()) {
        GTEST_SKIP();
    }
    std::vector<int> values{1, 2, 3, 4, 5, 6};
    std::vector<int> vec(values);
    void *vec_data = (void *)vec.data();
    int64_t vec_size = (int64_t)vec.size();
    core::Tensor t(std::move(vec));
    EXPECT_TRUE(t.GetDataPtr<int>() == vec_data);
    EXPECT_TRUE(t.GetShape() == core::SizeVector({vec_size}));
    EXPECT_EQ(t.ToFlatVector<int>(), values);
}
}  // namespace tests
}  // namespace open3d
