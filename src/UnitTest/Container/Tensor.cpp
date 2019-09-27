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
#include "Open3D/Container/Shape.h"
#include "Open3D/Container/Tensor.h"
#include "TestUtility/UnitTest.h"

#include <vector>

using namespace std;
using namespace open3d;

TEST(Tensor, CPU) {
    Shape shape{2, 3};
    Dtype dtype = Dtype::f32;
    Device device("CPU:0");
    Tensor t(shape, dtype, device);

    EXPECT_EQ(t.GetShape(), shape);
    EXPECT_EQ(t.GetBlob()->byte_size_, 4 * 2 * 3);
    EXPECT_EQ(t.GetBlob()->device_, device);
}

TEST(Tensor, GPU) {
    Shape shape{2, 3};
    Dtype dtype = Dtype::f32;
    Device device("GPU:0");
    Tensor t(shape, dtype, device);

    EXPECT_EQ(t.GetShape(), shape);
    EXPECT_EQ(t.GetBlob()->byte_size_, 4 * 2 * 3);
    EXPECT_EQ(t.GetBlob()->device_, device);
}

TEST(Tensor, WithInitValue) {
    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor t(vals, {2, 3}, Dtype::f32, Device("CPU:0"));
}

TEST(Tensor, WithInitValueTypeMismatch) {
    std::vector<int> vals{0, 1, 2, 3, 4, 5};
    EXPECT_THROW(Tensor(vals, {2, 3}, Dtype::f32, Device("CPU:0")),
                 std::runtime_error);
}

TEST(Tensor, WithInitValueSizeMismatch) {
    std::vector<float> vals{0, 1, 2, 3, 4};
    EXPECT_THROW(Tensor(vals, {2, 3}, Dtype::f32, Device("CPU:0")),
                 std::runtime_error);
}

TEST(Tensor, CopyTo) {
    Device src_device("CPU:0");
    Device dst_device("GPU:0");
    Shape shape{2, 3};
    Dtype dtype(Dtype::f32);

    std::vector<float> vals{0, 1, 2, 3, 4, 5};
    Tensor src_t(vals, shape, dtype, dst_device);

    Tensor dst_t = src_t.CopyTo(dst_device);

    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
    EXPECT_EQ(dst_t.GetBlob()->byte_size_,
              shape.NumElements() * DtypeUtil::ByteSize(dtype));
    EXPECT_EQ(dst_t.GetDevice(), dst_device);
    EXPECT_EQ(dst_t.GetShape(), src_t.GetShape());
}
