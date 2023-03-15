// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/IntersectionTest.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

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

}  // namespace tests
}  // namespace open3d
