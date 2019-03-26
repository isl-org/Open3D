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

#include <Eigen/Geometry>

#include <iostream>
using namespace std;
using namespace unit_test;

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

// ----------------------------------------------------------------------------
// Test clearing the basic data types.
// ----------------------------------------------------------------------------
TEST(BasicTypes, zero) {
    open3d::Mat3d m{};
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(m[r][c], 0.0);

    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++)
            m[r][c] = r * open3d::Mat3d::Cols + c;

    m.setZero();
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(m[r][c], 0.0);

    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++)
            m[r][c] = r * open3d::Mat3d::Cols + c;

    m = open3d::Mat3d::Zero();
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(m[r][c], 0.0);

    open3d::Vec3d v{};
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(v[c], 0.0);

    for (uint c = 0; c < open3d::Mat3d::Cols; c++) v[c] = c;
    v.setZero();
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(v[c], 0.0);

    for (uint c = 0; c < open3d::Mat3d::Cols; c++) v[c] = c;
    v = open3d::Vec3d::Zero();
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(v[c], 0.0);
}

// ----------------------------------------------------------------------------
// Test initializing to ones.
// ----------------------------------------------------------------------------
TEST(BasicTypes, ones) {
    open3d::Mat3d m{};
    m = open3d::Mat3d::Ones();
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(m[r][c], 1.0);

    open3d::Vec3d v{};
    v = open3d::Vec3d::Ones();
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) EXPECT_EQ(v[c], 1.0);
}

// ----------------------------------------------------------------------------
// Test initializing with identity.
// ----------------------------------------------------------------------------
TEST(BasicTypes, setIdentity) {
    open3d::Mat3d m{};
    m.setIdentity();
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
            if (r == c)
                EXPECT_EQ(m[r][c], 1.0);
            else
                EXPECT_EQ(m[r][c], 0.0);
        }

    m.setZero();
    m = open3d::Mat3d::Identity();
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
            if (r == c)
                EXPECT_EQ(m[r][c], 1.0);
            else
                EXPECT_EQ(m[r][c], 0.0);
        }
}

// ----------------------------------------------------------------------------
// Test initializing to random values between [-1, 1].
// ----------------------------------------------------------------------------
TEST(BasicTypes, random) {
    open3d::Mat3d m{};
    m = open3d::Mat3d::Random(-10.0, 15);
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
            EXPECT_TRUE(-10.0 <= m[r][c]);
            EXPECT_TRUE(m[r][c] <= 15.0);
        }

    open3d::Mat3i mi{};
    mi = open3d::Mat3i::Random(-10, 15);
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
            EXPECT_TRUE(-10 <= mi[r][c]);
            EXPECT_TRUE(mi[r][c] <= 15);
        }

    open3d::Vec3d v{};
    v = open3d::Vec3d::Random(-10.0, 15.0);
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
        EXPECT_TRUE(-10.0 <= v[c]);
        EXPECT_TRUE(v[c] <= 15.0);
    }

    open3d::Vec3i vi{};
    vi = open3d::Vec3i::Random(-10, 15);
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
        EXPECT_TRUE(-10 <= vi[c]);
        EXPECT_TRUE(vi[c] <= 15);
    }
}

// ----------------------------------------------------------------------------
// Test normalize.
// ----------------------------------------------------------------------------
TEST(BasicTypes, normalize) {
    open3d::Vec3d v0{};

    v0 = open3d::Vec3d::Random();

    Eigen::Vector3d ev0 = Eigen::Vector3d(v0[0], v0[1], v0[2]);

    v0.normalize();
    ev0.normalize();

    EXPECT_EQ(v0[0], ev0[0]);
    EXPECT_EQ(v0[1], ev0[1]);
    EXPECT_EQ(v0[2], ev0[2]);
}

// ----------------------------------------------------------------------------
// Test dot.
// ----------------------------------------------------------------------------
TEST(BasicTypes, dot) {
    open3d::Vec3d v0{};
    open3d::Vec3d v1{};

    v0 = open3d::Vec3d::Random();
    v1 = open3d::Vec3d::Random();

    Eigen::Vector3d ev0 = Eigen::Vector3d(v0[0], v0[1], v0[2]);
    Eigen::Vector3d ev1 = Eigen::Vector3d(v1[0], v1[1], v1[2]);

    auto open3d_dot = v0.dot(v1);
    auto eigen_dot = ev0.dot(ev1);

    EXPECT_EQ(eigen_dot, open3d_dot);
}

// ----------------------------------------------------------------------------
// Test cross.
// ----------------------------------------------------------------------------
TEST(BasicTypes, cross) {
    open3d::Vec3d v0{};
    open3d::Vec3d v1{};

    v0 = open3d::Vec3d::Random();
    v1 = open3d::Vec3d::Random();

    Eigen::Vector3d ev0 = Eigen::Vector3d(v0[0], v0[1], v0[2]);
    Eigen::Vector3d ev1 = Eigen::Vector3d(v1[0], v1[1], v1[2]);

    auto open3d_cross = v0.cross(v1);
    auto eigen_cross = ev0.cross(ev1);

    EXPECT_EQ(eigen_cross[0], open3d_cross[0]);
    EXPECT_EQ(eigen_cross[1], open3d_cross[1]);
    EXPECT_EQ(eigen_cross[2], open3d_cross[2]);
}

// ----------------------------------------------------------------------------
// Test indexing using the parenthesis operator.
// ----------------------------------------------------------------------------
TEST(BasicTypes, parenthesis) {
    open3d::Mat3d m{};
    m = open3d::Mat3d::Random(-10.0, 15);
    for (uint r = 0; r < open3d::Mat3d::Rows; r++)
        for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
            EXPECT_EQ(m[r][c], m(r, c));
            m(r, c);
            EXPECT_EQ(m[r][c], m(r, c));
        }

    open3d::Vec3d v{};
    v = open3d::Vec3d::Random(-10.0, 15.0);
    for (uint c = 0; c < open3d::Mat3d::Cols; c++) {
        EXPECT_EQ(v[c], v(c));
        v(c) = c;
        EXPECT_EQ(v[c], v(c));
    }
}

// ----------------------------------------------------------------------------
// Test the block<>() operator.
// ----------------------------------------------------------------------------
TEST(BasicTypes, block) {
    open3d::Mat3d m = open3d::Mat3d::Random(-10.0, 15);
    Eigen::Matrix3d em{};
    em << m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0],
            m[2][1], m[2][2];

    Eigen::Vector3d ev = em.block<3, 1>(0, 0);

    // Print<double, 3, 3>(m);
    // Print(em);
    // Print(ev);
}
