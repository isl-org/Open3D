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
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/TensorArray.h"
#include "TestUtility/UnitTest.h"

using namespace std;
using namespace open3d;

TEST(Container, Registry) {
    EXPECT_TRUE(MemoryManagerBackendRegistry()->Has("cpu"));
    EXPECT_NE(MemoryManagerBackendRegistry()->GetSingletonObject("cpu"),
              nullptr);
}

TEST(Container, CPU_Tensor) {
    Shape shape;

    // Create tensor
    shape = {4, 4};
    Tensor<float> matrix4f(shape, "cpu");

    // Create tensor with init values
    shape = {3};
    std::vector<float> init_val({1, 2, 3});
    Tensor<float> vector3f(init_val, shape, "cpu");

    // Check that the values are actually copied
    std::vector<float> out_val = vector3f.ToStdVector();
    unit_test::ExpectEQ(out_val, {1, 2, 3});
}

TEST(Container, CPU_Array) {
    Shape tensor_shape = {3};
    size_t max_size = 10000;
    TensorArray<float> points(tensor_shape, max_size, "cpu");
}

TEST(Container, GPU_CONDITIONAL_TEST(GPU_Tensor)) {
    Shape shape;

    // Create tensor
    shape = {4, 4};
    Tensor<float> matrix4f(shape, "gpu");

    // Create tensor with init values
    shape = {3};
    std::vector<float> init_val({1, 2, 3});
    Tensor<float> vector3f(init_val, shape, "gpu");

    // Check that the values are actually copied
    std::vector<float> out_val = vector3f.ToStdVector();
    unit_test::ExpectEQ(out_val, {1, 2, 3});
}

TEST(Container, GPU_CONDITIONAL_TEST(GPU_Array)) {
    Shape tensor_shape = {3};
    size_t max_size = 100000;
    TensorArray<float> points(tensor_shape, max_size, "gpu");
}
