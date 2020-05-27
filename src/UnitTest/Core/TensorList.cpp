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

#include "Open3D/Core/TensorList.h"

#include <vector>

#include "Core/CoreTest.h"
#include "TestUtility/UnitTest.h"

using namespace std;
using namespace open3d;

class TensorListPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TensorList,
                         TensorListPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TensorListPermuteDevices, ConstructFromIterators) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2};
    TensorList tensor_list(tensors.begin(), tensors.end(), device);

    SizeVector shape({3, 2, 3});
    EXPECT_EQ(tensor_list.AsTensor().GetShape(), shape);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
}

TEST_P(TensorListPermuteDevices, ConstructFromVector) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2};
    TensorList tensor_list(tensors, device);

    SizeVector shape({3, 2, 3});
    EXPECT_EQ(tensor_list.AsTensor().GetShape(), shape);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
}

TEST_P(TensorListPermuteDevices, ConstructFromInitList) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    TensorList tensor_list({t0, t1, t2}, device);

    SizeVector shape({3, 2, 3});
    EXPECT_EQ(tensor_list.AsTensor().GetShape(), shape);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
}

TEST_P(TensorListPermuteDevices, TensorConstructFromTensor) {
    Device device = GetParam();

    Tensor t(std::vector<float>(3 * 2 * 3, 1), {3, 2, 3}, Dtype::Float32,
             device);

    TensorList tensor_list(t);
    SizeVector shape({2, 3});
    EXPECT_EQ(tensor_list.GetShape(), shape);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(3 * 2 * 3, 1));
}

TEST_P(TensorListPermuteDevices, CopyConstruct) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2};
    TensorList tensor_list(tensors, device);
    TensorList tensor_list_new(tensor_list);

    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              tensor_list_new.AsTensor().ToFlatVector<float>());

    /// Change of the copy should NOT affect the origin
    tensor_list.AsTensor()[0][0][0] = 1;
    EXPECT_NE(tensor_list.AsTensor().ToFlatVector<float>(),
              tensor_list_new.AsTensor().ToFlatVector<float>());
}

TEST_P(TensorListPermuteDevices, AssignOperator) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    /// Right value assignment
    TensorList tensor_list_gt = {t0, t1, t2};
    TensorList tensor_list_a = {t2, t2, t2};
    TensorList tensor_list_b = {t0, t1};

    tensor_list_a.AsTensor().Slice(0, 0, 2) = tensor_list_b.AsTensor();
    EXPECT_EQ(tensor_list_a.AsTensor().ToFlatVector<float>(),
              tensor_list_gt.AsTensor().ToFlatVector<float>());
}

TEST_P(TensorListPermuteDevices, Resize) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2};
    TensorList tensor_list(tensors, device);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}));

    tensor_list.Resize(5);
    EXPECT_EQ(tensor_list.GetSize(), 5);
    EXPECT_EQ(tensor_list.GetReservedSize(), 16);
    EXPECT_EQ(
            tensor_list.AsTensor().ToFlatVector<float>(),
            std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    tensor_list.Resize(2);
    EXPECT_EQ(tensor_list.GetSize(), 2);
    EXPECT_EQ(tensor_list.GetReservedSize(), 16);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));
}

TEST_P(TensorListPermuteDevices, PushBack) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    TensorList tensor_list({2, 3}, Dtype::Float32, device);
    EXPECT_EQ(tensor_list.GetSize(), 0);
    EXPECT_EQ(tensor_list.GetReservedSize(), 1);

    tensor_list.PushBack(t0);
    EXPECT_EQ(tensor_list.GetSize(), 1);
    EXPECT_EQ(tensor_list.GetReservedSize(), 2);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0}));

    tensor_list.PushBack(t1);
    EXPECT_EQ(tensor_list.GetSize(), 2);
    EXPECT_EQ(tensor_list.GetReservedSize(), 4);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}));

    tensor_list.PushBack(t2);
    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, AccessOperator) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2};
    TensorList tensor_list(tensors, device);

    EXPECT_EQ(tensor_list.GetSize(), 3);
    EXPECT_EQ(tensor_list[0].ToFlatVector<float>(), t0.ToFlatVector<float>());
    EXPECT_EQ(tensor_list[1].ToFlatVector<float>(), t1.ToFlatVector<float>());
    EXPECT_EQ(tensor_list[2].ToFlatVector<float>(), t2.ToFlatVector<float>());

    tensor_list[0] = t2;
    tensor_list[1] = t1;
    tensor_list[2] = t0;

    EXPECT_EQ(tensor_list.AsTensor().ToFlatVector<float>(),
              std::vector<float>(
                      {2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}));
}

TEST_P(TensorListPermuteDevices, Slice) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);
    Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2, t3};
    TensorList tensor_list(tensors, device);

    Tensor tensor = tensor_list.AsTensor().Slice(0, 0, 3, 2);
    EXPECT_EQ(tensor.ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, IndexGet) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);
    Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, Dtype::Float32, device);

    std::vector<Tensor> tensors = {t0, t1, t2, t3};
    TensorList tensor_list(tensors, device);

    std::vector<Tensor> indices = {Tensor(std::vector<int64_t>({0, -1, 2}), {3},
                                          Dtype::Int64, device)};
    Tensor tensor = tensor_list.AsTensor().IndexGet(indices);
    EXPECT_EQ(tensor.ToFlatVector<float>(),
              std::vector<float>(
                      {0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2}));
}

TEST_P(TensorListPermuteDevices, Concatenate) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    std::vector<Tensor> tensors0 = {t0};

    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);
    Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, Dtype::Float32, device);
    std::vector<Tensor> tensors1 = {t1, t2, t3};

    TensorList tensor_list0(tensors0, device);
    TensorList tensor_list1(tensors1, device);

    TensorList tensor_list2 = tensor_list0 + tensor_list1;
    EXPECT_EQ(tensor_list2.GetSize(), 4);
    EXPECT_EQ(tensor_list2.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list2.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));

    TensorList tensor_list3 =
            TensorList::Concatenate(tensor_list1, tensor_list0);
    EXPECT_EQ(tensor_list3.GetSize(), 4);
    EXPECT_EQ(tensor_list3.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list3.AsTensor().ToFlatVector<float>(),
              std::vector<float>({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                  3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0}));
}

TEST_P(TensorListPermuteDevices, Extend) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    std::vector<Tensor> tensors0 = {t0};

    Tensor t1(std::vector<float>(2 * 3, 1), {2, 3}, Dtype::Float32, device);
    Tensor t2(std::vector<float>(2 * 3, 2), {2, 3}, Dtype::Float32, device);
    Tensor t3(std::vector<float>(2 * 3, 3), {2, 3}, Dtype::Float32, device);
    std::vector<Tensor> tensors1 = {t1, t2, t3};

    TensorList tensor_list0(tensors0, device);
    TensorList tensor_list1(tensors1, device);

    tensor_list0.Extend(tensor_list1);
    EXPECT_EQ(tensor_list0.GetSize(), 4);
    EXPECT_EQ(tensor_list0.GetReservedSize(), 8);
    EXPECT_EQ(tensor_list0.AsTensor().ToFlatVector<float>(),
              std::vector<float>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));

    tensor_list1 += tensor_list1;
    EXPECT_EQ(tensor_list1.GetSize(), 6);
    EXPECT_EQ(tensor_list1.GetReservedSize(), 16);
    EXPECT_EQ(tensor_list1.AsTensor().ToFlatVector<float>(),
              std::vector<float>({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                  3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1,
                                  2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));
}

TEST_P(TensorListPermuteDevices, Clear) {
    Device device = GetParam();

    Tensor t0(std::vector<float>(2 * 3, 0), {2, 3}, Dtype::Float32, device);
    std::vector<Tensor> tensors = {t0};

    TensorList tensor_list(tensors, device);
    tensor_list.Clear();
    EXPECT_EQ(tensor_list.GetSize(), 0);
    EXPECT_EQ(tensor_list.GetReservedSize(), 1);
}
