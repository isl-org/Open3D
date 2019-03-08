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

#include "TestUtility/UnitTest.h"

#include "Open3D/Types/Types.h"

#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// The data type size varies based on alignment:
// - without custom alignment the size matches N x sizeof(TYPE)
// - with custom alignment the size is different than N x sizeof(TYPE)
// ----------------------------------------------------------------------------
TEST(Types, sizeof_type)
{
    EXPECT_EQ(3 * sizeof(double),     sizeof(Eigen::Vector3d));

    EXPECT_EQ(3 * 3 * sizeof(double), sizeof(open3d::Matrix3d));
    EXPECT_EQ(3 * 3 * sizeof(float),  sizeof(open3d::Matrix3f));

    EXPECT_EQ(4 * 4 * sizeof(float),  sizeof(open3d::Matrix4f));
    EXPECT_EQ(4 * 4 * sizeof(double), sizeof(open3d::Matrix4d));

    EXPECT_EQ(6 * 6 * sizeof(float),  sizeof(open3d::Matrix6f));
    EXPECT_EQ(6 * 6 * sizeof(double), sizeof(open3d::Matrix6d));

    EXPECT_EQ(    3 * sizeof(double), sizeof(open3d::Vector3d));
    EXPECT_EQ(    3 * sizeof(float),  sizeof(open3d::Vector3f));
    EXPECT_EQ(    3 * sizeof(int),    sizeof(open3d::Vector3i));
}

