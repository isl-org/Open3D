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

#include "open3d/t/geometry/TensorListMap.h"

#include <vector>

#include "tests/UnitTest.h"
#include "tests/core/CoreTest.h"

namespace open3d {
namespace tests {

TEST(TensorListMap, Constructor_GetPrimaryKey) {
    t::geometry::TensorListMap tm("points");
    EXPECT_EQ(tm.GetPrimaryKey(), "points");
}

TEST(TensorListMap, Assign) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm("points");
    tm["points"] = core::TensorList({3}, dtype, device);
    tm["dummy"] = core::TensorList({3}, dtype, device);
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("dummy"));

    std::unordered_map<std::string, core::TensorList> replacement{
            {"points", core::TensorList::FromTensor(
                               core::Tensor::Ones({5, 3}, dtype, device))},
            {"colors", core::TensorList::FromTensor(
                               core::Tensor::Ones({5, 3}, dtype, device))},
    };
    tm.Assign(replacement);
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("colors"));
    EXPECT_FALSE(tm.Contains("dummy"));

    // Underlying memory are the same.
    EXPECT_TRUE(
            tm["points"].AsTensor().IsSame(replacement["points"].AsTensor()));
    EXPECT_TRUE(
            tm["colors"].AsTensor().IsSame(replacement["colors"].AsTensor()));
}

TEST(TensorListMap, SynchronizedPushBack) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))}});
    EXPECT_EQ(tm["points"].GetSize(), 5);
    EXPECT_EQ(tm["colors"].GetSize(), 5);

    // Good.
    core::Tensor a_point = core::Tensor::Ones({3}, dtype, device);
    core::Tensor a_color = core::Tensor::Ones({3}, dtype, device);
    tm.SynchronizedPushBack({{"points", a_point}, {"colors", a_color}});
    EXPECT_EQ(tm["points"].GetSize(), 6);
    EXPECT_EQ(tm["colors"].GetSize(), 6);
    EXPECT_TRUE(tm["points"][5].AllClose(a_point));
    EXPECT_FALSE(tm["points"][5].IsSame(a_point));  // PushBack copies memory.
    EXPECT_TRUE(tm["colors"][5].AllClose(a_color));
    EXPECT_FALSE(tm["colors"][5].IsSame(a_color));  // PushBack copies memory.

    // Missing key.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack({{"colors", a_color}}));

    // Unexpected key.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack({{"points", a_point},
                                              {"colors", a_color},
                                              {"more_colors", a_color}}));

    // Wrong dtype.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack(
            {{"points", core::Tensor::Ones({3}, core::Dtype::Float64, device)},
             {"colors", a_color}}));

    // Wrong shape.
    EXPECT_ANY_THROW(tm.SynchronizedPushBack(
            {{"points", core::Tensor::Ones({5}, core::Dtype::Float32, device)},
             {"colors", a_color}}));
}

TEST(TensorListMap, IsSizeSynchronized) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_FALSE(tm.IsSizeSynchronized());

    tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
    EXPECT_TRUE(tm.IsSizeSynchronized());
}

TEST(TensorListMap, AssertSizeSynchronized) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_ANY_THROW(tm.AssertSizeSynchronized());

    tm["colors"].PushBack(core::Tensor::Ones({3}, dtype, device));
    EXPECT_NO_THROW(tm.AssertSizeSynchronized());
}

TEST(TensorListMap, Contains) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = core::Device("CPU:0");

    t::geometry::TensorListMap tm(
            "points",
            {{"points", core::TensorList::FromTensor(
                                core::Tensor::Ones({5, 3}, dtype, device))},
             {"colors", core::TensorList::FromTensor(
                                core::Tensor::Ones({4, 3}, dtype, device))}});
    EXPECT_TRUE(tm.Contains("points"));
    EXPECT_TRUE(tm.Contains("colors"));
    EXPECT_FALSE(tm.Contains("normals"));
}

}  // namespace tests
}  // namespace open3d
