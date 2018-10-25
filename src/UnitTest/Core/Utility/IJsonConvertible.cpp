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

#include "Core/Utility/IJsonConvertible.h"
#include <json/json.h>

using namespace open3d;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(IJsonConvertible, EigenVector3dToFromJsonArray)
{
    int loops = 10000;
    srand((unsigned int) time(0));
    for (int i = 0; i < loops; i++)
    {
        Eigen::Vector3d v3d = Eigen::Vector3d::Random();

        bool status = false;
        Json::Value json_value;
        Eigen::Vector3d ref;

        status = IJsonConvertible::EigenVector3dToJsonArray(v3d, json_value);
        EXPECT_TRUE(status);

        status = IJsonConvertible::EigenVector3dFromJsonArray(ref, json_value);
        EXPECT_TRUE(status);

        EXPECT_NEAR(ref(0, 0), v3d(0, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 0), v3d(1, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 0), v3d(2, 0), THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(IJsonConvertible, EigenVector4dToFromJsonArray)
{
    int loops = 10000;
    srand((unsigned int) time(0));
    for (int i = 0; i < loops; i++)
    {
        Eigen::Vector4d v4d = Eigen::Vector4d::Random();

        bool status = false;
        Json::Value json_value;
        Eigen::Vector4d ref;

        status = IJsonConvertible::EigenVector4dToJsonArray(v4d, json_value);
        EXPECT_TRUE(status);

        status = IJsonConvertible::EigenVector4dFromJsonArray(ref, json_value);
        EXPECT_TRUE(status);

        EXPECT_NEAR(ref(0, 0), v4d(0, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 0), v4d(1, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 0), v4d(2, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(3, 0), v4d(3, 0), THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(IJsonConvertible, EigenMatrix3dToFromJsonArray)
{
    int loops = 10000;
    srand((unsigned int) time(0));
    for (int i = 0; i < loops; i++)
    {
        Eigen::Matrix3d m3d = Eigen::Matrix3d::Random();

        bool status = false;
        Json::Value json_value;
        Eigen::Matrix3d ref;

        status = IJsonConvertible::EigenMatrix3dToJsonArray(m3d, json_value);
        EXPECT_TRUE(status);

        status = IJsonConvertible::EigenMatrix3dFromJsonArray(ref, json_value);
        EXPECT_TRUE(status);

        EXPECT_NEAR(ref(0, 0), m3d(0, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 0), m3d(1, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 0), m3d(2, 0), THRESHOLD_1E_6);

        EXPECT_NEAR(ref(0, 1), m3d(0, 1), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 1), m3d(1, 1), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 1), m3d(2, 1), THRESHOLD_1E_6);

        EXPECT_NEAR(ref(0, 2), m3d(0, 2), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 2), m3d(1, 2), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 2), m3d(2, 2), THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(IJsonConvertible, EigenMatrix4dToFromJsonArray)
{
    int loops = 10000;
    srand((unsigned int) time(0));
    for (int i = 0; i < loops; i++)
    {
        Eigen::Matrix4d m4d = Eigen::Matrix4d::Random();

        bool status = false;
        Json::Value json_value;
        Eigen::Matrix4d ref;

        status = IJsonConvertible::EigenMatrix4dToJsonArray(m4d, json_value);
        EXPECT_TRUE(status);

        status = IJsonConvertible::EigenMatrix4dFromJsonArray(ref, json_value);
        EXPECT_TRUE(status);

        EXPECT_NEAR(ref(0, 0), m4d(0, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 0), m4d(1, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 0), m4d(2, 0), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(3, 0), m4d(3, 0), THRESHOLD_1E_6);

        EXPECT_NEAR(ref(0, 1), m4d(0, 1), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 1), m4d(1, 1), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 1), m4d(2, 1), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(3, 1), m4d(3, 1), THRESHOLD_1E_6);

        EXPECT_NEAR(ref(0, 2), m4d(0, 2), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 2), m4d(1, 2), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 2), m4d(2, 2), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(3, 2), m4d(3, 2), THRESHOLD_1E_6);

        EXPECT_NEAR(ref(0, 3), m4d(0, 3), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(1, 3), m4d(1, 3), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(2, 3), m4d(2, 3), THRESHOLD_1E_6);
        EXPECT_NEAR(ref(3, 3), m4d(3, 3), THRESHOLD_1E_6);
    }
}
