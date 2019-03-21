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

#include "Open3D/Types/Mat.h"

#include <iostream>
using namespace std;

// ----------------------------------------------------------------------------
// Make sure the types are PODs.
// ----------------------------------------------------------------------------
TEST(BasicTypes, is_POD) {
    EXPECT_TRUE(is_pod<open3d::Mat3d>());
    EXPECT_TRUE(is_pod<open3d::Mat3f>());
    EXPECT_TRUE(is_pod<open3d::Mat4d>());
    EXPECT_TRUE(is_pod<open3d::Mat4f>());
    EXPECT_TRUE(is_pod<open3d::Mat6d>());
    EXPECT_TRUE(is_pod<open3d::Mat6f>());

    EXPECT_TRUE(is_pod<open3d::Vec3d>());
    EXPECT_TRUE(is_pod<open3d::Vec3f>());
    EXPECT_TRUE(is_pod<open3d::Vec3i>());
}

// ----------------------------------------------------------------------------
// Test reading/writing using the subscript operator.
// ----------------------------------------------------------------------------
TEST(BasicTypes, subscript_ops) {
    open3d::Mat3f m;

    for (uint r = 0; r < m.Rows; r++)
        for (uint c = 0; c < m.Cols; c++) {
            m[r][c] = r * 1.0f + c * 0.1f;
            EXPECT_FLOAT_EQ(m[r][c], r * 1.0f + c * 0.1f);
        }

    // will fail assert in debug mode due to out of bounds accesses
    // m[3][2] = 1.0f;
    // m[2][3] = 2.0f;

    open3d::Vec3f v;

    for (uint c = 0; c < m.Cols; c++) {
        v[c] = c * 0.12f;
        EXPECT_FLOAT_EQ(v[c], c * 0.12f);
    }
}

// ----------------------------------------------------------------------------
// Test the cast T* operator.
// Note: there's no need for an explicit cast operator, just use member s[...].
// ----------------------------------------------------------------------------
TEST(BasicTypes, cast) {
    open3d::Mat3f m;
    float* mf = (float*)m.s;

    for (uint r = 0; r < m.Rows; r++)
        for (uint c = 0; c < m.Cols; c++) {
            mf[r * m.Cols + c] = r * 1.0f + c * 0.1f;
            EXPECT_FLOAT_EQ(mf[r * m.Cols + c], r * 1.0f + c * 0.1f);
            EXPECT_FLOAT_EQ(m[r][c], r * 1.0f + c * 0.1f);
        }

    // test memcpy
    float mcpy[9];
    memcpy(mcpy, mf, 9 * sizeof(float));
    for (uint r = 0; r < m.Rows; r++)
        for (uint c = 0; c < m.Cols; c++)
            EXPECT_FLOAT_EQ(mcpy[r * m.Cols + c], r * 1.0f + c * 0.1f);

    open3d::Vec3f v;
    float* vf = (float*)v.s;

    for (uint c = 0; c < m.Cols; c++) {
        vf[c] = c * 0.12f;
        EXPECT_FLOAT_EQ(vf[c], c * 0.12f);
        EXPECT_FLOAT_EQ(v[c], c * 0.12f);
    }
}

// ----------------------------------------------------------------------------
// The data type size varies based on alignment:
// - without custom alignment the size matches N x sizeof(TYPE)
// - with custom alignment the size is different than N x sizeof(TYPE)
// ----------------------------------------------------------------------------
TEST(BasicTypes, sizeof_type) {
    EXPECT_EQ(3 * sizeof(double), sizeof(Eigen::Vector3d));

    EXPECT_EQ(3 * 3 * sizeof(double), sizeof(open3d::Mat3d));
    EXPECT_EQ(3 * 3 * sizeof(float), sizeof(open3d::Mat3f));

    EXPECT_EQ(4 * 4 * sizeof(float), sizeof(open3d::Mat4f));
    EXPECT_EQ(4 * 4 * sizeof(double), sizeof(open3d::Mat4d));

    EXPECT_EQ(6 * 6 * sizeof(float), sizeof(open3d::Mat6f));
    EXPECT_EQ(6 * 6 * sizeof(double), sizeof(open3d::Mat6d));

    EXPECT_EQ(3 * sizeof(double), sizeof(open3d::Vec3d));
    EXPECT_EQ(3 * sizeof(float), sizeof(open3d::Vec3f));
    EXPECT_EQ(3 * sizeof(int), sizeof(open3d::Vec3i));
}

// ----------------------------------------------------------------------------
// Test ==, !=, <=, >=.
// ----------------------------------------------------------------------------
TEST(BasicTypes, comparison_ops_float) {
    open3d::Mat3f m0 = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    open3d::Mat3f m1 = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    EXPECT_TRUE(m0 == m1);
    EXPECT_TRUE(m0 <= m1);

    m0[1][0] = 3.1f;
    EXPECT_TRUE(m0 != m1);
    EXPECT_TRUE(m0 >= m1);

    open3d::Vec3f v0 = {0.0f, 0.1f, 0.2f};
    open3d::Vec3f v1 = {0.0f, 0.1f, 0.2f};
    EXPECT_TRUE(v0 == v1);
    EXPECT_TRUE(v0 <= v1);

    v0[2] = 1.2f;
    EXPECT_TRUE(v0 != v1);
    EXPECT_TRUE(v0 >= v1);
}

// ----------------------------------------------------------------------------
// Test ==, !=, <=, >=.
// ----------------------------------------------------------------------------
TEST(BasicTypes, comparison_ops_double) {
    open3d::Mat3d m0 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    open3d::Mat3d m1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    EXPECT_TRUE(m0 == m1);
    EXPECT_TRUE(m0 <= m1);

    m0[1][0] = 3.1;
    EXPECT_TRUE(m0 != m1);
    EXPECT_TRUE(m0 >= m1);

    open3d::Vec3d v0 = {0.0, 0.1, 0.2};
    open3d::Vec3d v1 = {0.0, 0.1, 0.2};
    EXPECT_TRUE(v0 == v1);
    EXPECT_TRUE(v0 <= v1);

    v0[2] = 1.2;
    EXPECT_TRUE(v0 != v1);
    EXPECT_TRUE(v0 >= v1);
}
