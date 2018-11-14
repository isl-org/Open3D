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

#include "UnitTest.h"

#include <Eigen/Core>
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
void unit_test::NotImplemented()
{
    cout << "\033[0;32m"
         << "[          ] "
         << "\033[0;0m";
    cout << "\033[0;31m"
         << "Not implemented."
         << "\033[0;0m" << endl;

    // FAIL();
    // ADD_FAILURE();
    GTEST_NONFATAL_FAILURE_("Not implemented");
}

// ----------------------------------------------------------------------------
// Equal test over Eigen::Vector2d components.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const Eigen::Vector2d& v0, const Eigen::Vector2d& v1)
{
    EXPECT_NEAR(v0(0, 0), v1(0, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(1, 0), v1(1, 0), unit_test::THRESHOLD_1E_6);
}
void unit_test::ExpectEQ(const double& v00, const double& v01, const Eigen::Vector2d& v1)
{
    EXPECT_NEAR(v00, v1(0, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v01, v1(1, 0), unit_test::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// Equal test over Eigen::Vector3d components.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1)
{
    EXPECT_NEAR(v0(0, 0), v1(0, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(1, 0), v1(1, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(2, 0), v1(2, 0), unit_test::THRESHOLD_1E_6);
}
void unit_test::ExpectEQ(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1)
{
    EXPECT_NEAR(v00, v1(0, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v01, v1(1, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v02, v1(2, 0), unit_test::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// Equal test over Eigen::Matrix3d components.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const Eigen::Matrix3d& v0, const Eigen::Matrix3d& v1)
{
    EXPECT_NEAR(v0(0, 0), v1(0, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(1, 0), v1(1, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(2, 0), v1(2, 0), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(0, 1), v1(0, 1), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(1, 1), v1(1, 1), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(2, 1), v1(2, 1), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(0, 2), v1(0, 2), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(1, 2), v1(1, 2), unit_test::THRESHOLD_1E_6);
    EXPECT_NEAR(v0(2, 2), v1(2, 2), unit_test::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// Equal test over Eigen::Vector2i components.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const Eigen::Vector2i& v0, const Eigen::Vector2i& v1)
{
    EXPECT_EQ(v0(0, 0), v1(0, 0));
    EXPECT_EQ(v0(1, 0), v1(1, 0));
}
void unit_test::ExpectEQ(const int& v00, const int& v01, const Eigen::Vector2i& v1)
{
    EXPECT_EQ(v00, v1(0, 0));
    EXPECT_EQ(v01, v1(1, 0));
}

// ----------------------------------------------------------------------------
// Equal test over Eigen::Vector3i components.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const Eigen::Vector3i& v0, const Eigen::Vector3i& v1)
{
    EXPECT_EQ(v0(0, 0), v1(0, 0));
    EXPECT_EQ(v0(1, 0), v1(1, 0));
    EXPECT_EQ(v0(2, 0), v1(2, 0));
}
void unit_test::ExpectEQ(const int& v00, const int& v01, const int& v02, const Eigen::Vector3i& v1)
{
    EXPECT_EQ(v00, v1(0, 0));
    EXPECT_EQ(v01, v1(1, 0));
    EXPECT_EQ(v02, v1(2, 0));
}

// ----------------------------------------------------------------------------
// Less than or Equal test over Eigen::Vector3d components.
// ----------------------------------------------------------------------------
void unit_test::ExpectLE(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1)
{
    EXPECT_LE(v0(0, 0), v1(0, 0));
    EXPECT_LE(v0(1, 0), v1(1, 0));
    EXPECT_LE(v0(2, 0), v1(2, 0));
}
void unit_test::ExpectLE(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1)
{
    EXPECT_LE(v00, v1(0, 0));
    EXPECT_LE(v01, v1(1, 0));
    EXPECT_LE(v02, v1(2, 0));
}

// ----------------------------------------------------------------------------
// Greater than or Equal test over Eigen::Vector3d components.
// ----------------------------------------------------------------------------
void unit_test::ExpectGE(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1)
{
    EXPECT_GE(v0(0, 0), v1(0, 0));
    EXPECT_GE(v0(1, 0), v1(1, 0));
    EXPECT_GE(v0(2, 0), v1(2, 0));
}
void unit_test::ExpectGE(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1)
{
    EXPECT_GE(v00, v1(0, 0));
    EXPECT_GE(v01, v1(1, 0));
    EXPECT_GE(v02, v1(2, 0));
}

// ----------------------------------------------------------------------------
// Test equality of two arrays of int.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const int* const v0,
              const int* const v1,
              const size_t& size)
{
    for (int i = 0; i < size; i++)
        EXPECT_EQ(v0[i], v1[i]);
}

// ----------------------------------------------------------------------------
// Test equality of two vectors of int.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const vector<int>& v0, const vector<int>& v1)
{
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(&v0[0], &v1[0], v0.size());
}

// ----------------------------------------------------------------------------
// Test equality of two arrays of double.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const double* const v0,
              const double* const v1,
              const size_t& size)
{
    for (int i = 0; i < size; i++)
        EXPECT_NEAR(v0[i], v1[i], THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
// Test equality of two vectors of double.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const vector<double>& v0, const vector<double>& v1)
{
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(&v0[0], &v1[0], v0.size());
}
