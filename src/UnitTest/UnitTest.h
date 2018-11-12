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

#pragma once

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>

#include "Print.h"
#include "Rand.h"
#include "Sort.h"

namespace unit_test
{
    // thresholds for comparing floating point values
    const double THRESHOLD_1E_6 = 1e-6;

    // Mechanism for reporting unit tests for which there is no implementation yet.
    void NotImplemented();

    // Equal test.
    template<class T, int M, int N>
    void ExpectEQ(const Eigen::Matrix<T, M, N>& v0,
                  const Eigen::Matrix<T, M, N>& v1)
    {
        EXPECT_EQ(v0.size(), v1.size());
        for (int i = 0; i < v0.size(); i++)
            EXPECT_NEAR(v0.coeff(i), v1.coeff(i), THRESHOLD_1E_6);
    }

    // Equal test over Eigen::Vector2d components.
    void ExpectEQ(const Eigen::Vector2d& v0, const Eigen::Vector2d& v1);
    void ExpectEQ(const double& v00, const double& v01, const Eigen::Vector2d& v1);

    // Equal test over Eigen::Vector3d components.
    void ExpectEQ(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1);
    void ExpectEQ(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1);

    // Equal test over Eigen::Matrix3d components.
    void ExpectEQ(const Eigen::Matrix3d& v0, const Eigen::Matrix3d& v1);

    // Equal test over Eigen::Vector2i components.
    void ExpectEQ(const Eigen::Vector2i& v0, const Eigen::Vector2i& v1);
    void ExpectEQ(const int& v00, const int& v01, const Eigen::Vector2i& v1);

    // Equal test over Eigen::Vector3i components.
    void ExpectEQ(const Eigen::Vector3i& v0, const Eigen::Vector3i& v1);
    void ExpectEQ(const int& v00, const int& v01, const int& v02, const Eigen::Vector3i& v1);

    // Less than or Equal test over Eigen::Vector3d components.
    void ExpectLE(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1);
    void ExpectLE(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1);

    // Greater than or Equal test over Eigen::Vector3d components.
    void ExpectGE(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1);
    void ExpectGE(const double& v00, const double& v01, const double& v02, const Eigen::Vector3d& v1);

    // Reinterpret cast from uint8_t* to float*.
    template<class T>
    T* const Cast(uint8_t* data)
    {
        return reinterpret_cast<T* const>(data);
    }
}
