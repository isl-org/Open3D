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

    // Mechanism for reporting unit tests for which there is no implementation.
    void NotImplemented();

    // Equal test.
    template<class T>
    void ExpectEQ(const T& v0, const T& v1)
    {
        EXPECT_EQ(v0.size(), v1.size());
        for (int i = 0; i < v0.size(); i++)
            EXPECT_NEAR(v0.coeff(i), v1.coeff(i), THRESHOLD_1E_6);
    }

    // Less than or Equal test.
    template<class T>
    void ExpectLE(const T& v0, const T& v1)
    {
        EXPECT_EQ(v0.size(), v1.size());
        for (int i = 0; i < v0.size(); i++)
            EXPECT_LE(v0.coeff(i), v1.coeff(i));
    }

    // Greater than or Equal test.
    template<class T>
    void ExpectGE(const T& v0, const T& v1)
    {
        EXPECT_EQ(v0.size(), v1.size());
        for (int i = 0; i < v0.size(); i++)
            EXPECT_GE(v0.coeff(i), v1.coeff(i));
    }
}
