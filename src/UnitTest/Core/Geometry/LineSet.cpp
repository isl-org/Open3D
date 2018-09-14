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

#include "Core/Geometry/LineSet.h"

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, Constructor)
{
    open3d::LineSet ls;

    // inherited from Geometry2D
    EXPECT_EQ(open3d::Geometry::GeometryType::LineSet, ls.GetGeometryType());
    EXPECT_EQ(3, ls.Dimension());

    // public member variables
    EXPECT_EQ(0, ls.points_.size());
    EXPECT_EQ(0, ls.lines_.size());
    EXPECT_EQ(0, ls.colors_.size());

    // public members
    EXPECT_TRUE(ls.IsEmpty());

    Eigen::Vector3d minBound = ls.GetMinBound();
    Eigen::Vector3d maxBound = ls.GetMaxBound();

    EXPECT_FLOAT_EQ(0.0, minBound(0, 0));
    EXPECT_FLOAT_EQ(0.0, minBound(1, 0));
    EXPECT_FLOAT_EQ(0.0, minBound(2, 0));

    EXPECT_FLOAT_EQ(0.0, maxBound(0, 0));
    EXPECT_FLOAT_EQ(0.0, maxBound(1, 0));
    EXPECT_FLOAT_EQ(0.0, maxBound(2, 0));

    EXPECT_FALSE(ls.HasPoints());
    EXPECT_FALSE(ls.HasLines());
    EXPECT_FALSE(ls.HasColors());
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_MemberData)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_Clear)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_IsEmpty)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_GetMinBound)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_GetMaxBound)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_Transform)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_HasPoints)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_HasLines)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_HasColors)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_GetLineCoordinate)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSet, DISABLED_CreateLineSetFromPointCloudCorrespondences)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
// 
// ----------------------------------------------------------------------------
TEST(LineSetFactory, DISABLED_CreateLineSetFromPointCloudCorrespondences)
{
    UnitTest::NotImplemented();
}
