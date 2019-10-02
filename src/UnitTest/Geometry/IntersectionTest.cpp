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

#include "Open3D/Geometry/IntersectionTest.h"
#include "TestUtility/Raw.h"
#include "TestUtility/UnitTest.h"

using namespace Eigen;
using namespace open3d;
using namespace std;
using namespace unit_test;

TEST(IntersectionTest, PointsCoplanar) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d p2(0, 1, 0);
    Eigen::Vector3d p3(1, 1, 0);

    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p1, p2, p3));
    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p0, p2, p3));
    EXPECT_TRUE(geometry::IntersectionTest::PointsCoplanar(p0, p1, p2, p2));
}

TEST(IntersectionTest, LinesMinimumDistance) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d q0(0, 1, 0);
    Eigen::Vector3d q1(1, 1, 0);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p0, q0, q1),
              -1.);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, q0, q0),
              -2.);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, q0, q1),
              -3.);

    Eigen::Vector3d u0(1, 0, 0);
    Eigen::Vector3d u1(1, 1, 0);
    EXPECT_EQ(geometry::IntersectionTest::LinesMinimumDistance(p0, p1, u0, u1),
              0.);
}

TEST(IntersectionTest, LineSegmentsMinimumDistance) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d q0(0, 0, 1);
    Eigen::Vector3d q1(1, 0, 1);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);

    p0 = Eigen::Vector3d(0, 0, 0);
    p1 = Eigen::Vector3d(1, 0, 0);
    q0 = Eigen::Vector3d(2, 0, 0);
    q1 = Eigen::Vector3d(4, 0, 0);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);

    p0 = Eigen::Vector3d(0, 0, 0);
    p1 = Eigen::Vector3d(1, 0, 0);
    q0 = Eigen::Vector3d(0, 1, 0);
    q1 = Eigen::Vector3d(1, 4, 0);
    EXPECT_EQ(geometry::IntersectionTest::LineSegmentsMinimumDistance(p0, p1,
                                                                      q0, q1),
              1.);
}
