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

#include "open3d/geometry/IntersectionTest.h"

#include "tests/UnitTest.h"

using namespace open3d::geometry;
using namespace ::testing;
using Eigen::Vector3d;

using pnt_t = std::tuple<Vector3d, Vector3d>;
using pln_t = std::tuple<Vector3d, Vector3d, bool>;
#define ROOT2 1.0 / sqrt(2.0)
#define ROOT3 1.0 / sqrt(3.0)

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

TEST(IntersectionTest, ClosestDistanceAABBPoint) {
    // Tests the closest distance from a point to an AABB. This single test is
    // adequate under the assumption that the ClosestDistanceAABB uses the
    // ClosestPointAABB method under the hood, and the parameterized tests
    // in ClosestPointAABBTests pass.
    AxisAlignedBoundingBox box{{0, 0, 0}, {1, 1, 1}};
    auto value = IntersectionTest::ClosestDistanceAABB({4, 0.5, 0.5}, box);
    EXPECT_DOUBLE_EQ(3.0, value);
}

// Closest point to AABB tests
class ClosestPointAABBTests : public TestWithParam<pnt_t> {};

TEST_P(ClosestPointAABBTests, CheckClosestPoints) {
    AxisAlignedBoundingBox box{{0, 0, 0}, {1, 1, 1}};
    auto test = std::get<0>(GetParam());
    auto expected = std::get<1>(GetParam());

    auto closest = IntersectionTest::ClosestPointAABB(test, box);
    ExpectEQ(expected, closest);
}

INSTANTIATE_TEST_CASE_P(IntersectionTest,
                        ClosestPointAABBTests,
                        Values(  // Faces
                                pnt_t({-1, 0.5, 0.5}, {0, 0.5, 0.5}),
                                pnt_t({2, 0.5, 0.5}, {1, 0.5, 0.5}),
                                pnt_t({0.5, -1, 0.5}, {0.5, 0, 0.5}),
                                pnt_t({0.5, 2, 0.5}, {0.5, 1, 0.5}),
                                pnt_t({0.5, 0.5, -1}, {0.5, 0.5, 0}),
                                pnt_t({0.5, 0.5, 2}, {0.5, 0.5, 1}),

                                // Edges
                                pnt_t({0.5, -1, -1}, {0.5, 0, 0}),
                                pnt_t({0.5, 2, 2}, {0.5, 1, 1}),
                                pnt_t({-1, 0.5, -1}, {0, 0.5, 0}),
                                pnt_t({2, 0.5, 2}, {1, 0.5, 1}),
                                pnt_t({-1, -1, 0.5}, {0, 0, 0.5}),
                                pnt_t({2, 2, 0.5}, {1, 1, 0.5}),

                                // Corners
                                pnt_t({-1, -1, -1}, {0, 0, 0}),
                                pnt_t({2, 2, 2}, {1, 1, 1}),

                                // Points already inside the AABB
                                pnt_t({0, 1, 1}, {0, 1, 1}),
                                pnt_t({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5})));

// Farthest point to AABB tests
class FarthestPointAABBTests : public TestWithParam<pnt_t> {};

TEST_P(FarthestPointAABBTests, CheckFarthestPoints) {
    AxisAlignedBoundingBox box{{-1, -1, -1}, {1, 1, 1}};
    auto test = std::get<0>(GetParam());
    auto expected = std::get<1>(GetParam());

    auto closest = IntersectionTest::FarthestPointAABB(test, box);
    ExpectEQ(expected, closest);
}

INSTANTIATE_TEST_CASE_P(IntersectionTest,
                        FarthestPointAABBTests,
                        Values(  // Inside
                                pnt_t({0.5, 0.5, 0.5}, {-1, -1, -1}),
                                pnt_t({-0.5, 0.5, 0.5}, {1, -1, -1}),
                                pnt_t({0.5, -0.5, 0.5}, {-1, 1, -1}),
                                pnt_t({0.5, 0.5, -0.5}, {-1, -1, 1}),
                                pnt_t({-0.5, -0.5, 0.5}, {1, 1, -1}),
                                pnt_t({0.5, -0.5, -0.5}, {-1, 1, 1}),
                                pnt_t({-0.5, 0.5, -0.5}, {1, -1, 1}),
                                pnt_t({-0.5, -0.5, -0.5}, {1, 1, 1}),

                                // Outside
                                pnt_t({3, 3, 3}, {-1, -1, -1}),
                                pnt_t({-3, 3, 3}, {1, -1, -1}),
                                pnt_t({3, -3, 3}, {-1, 1, -1}),
                                pnt_t({3, 3, -3}, {-1, -1, 1}),
                                pnt_t({-3, -3, 3}, {1, 1, -1}),
                                pnt_t({3, -3, -3}, {-1, 1, 1}),
                                pnt_t({-3, 3, -3}, {1, -1, 1}),
                                pnt_t({-3, -3, -3}, {1, 1, 1})));

// Hyperplane to AABB Intersection Tests
class PlaneAABBTests : public TestWithParam<pln_t> {};

TEST_P(PlaneAABBTests, CheckFarthestPoints) {
    AxisAlignedBoundingBox box{{-1, -1, -1}, {1, 1, 1}};
    auto point = std::get<0>(GetParam());
    auto normal = std::get<1>(GetParam());
    auto expected = std::get<2>(GetParam());

    Eigen::Hyperplane<double, 3> plane(normal, point);
    auto intersects = IntersectionTest::PlaneAABB(plane, box);
    EXPECT_EQ(expected, intersects);
}

INSTANTIATE_TEST_CASE_P(
        IntersectionTest,
        PlaneAABBTests,
        Values(pln_t({2, 0, 0}, {1, 0, 0}, false),
               pln_t({-2, 0, 0}, {1, 0, 0}, false),
               pln_t({2, 0, 0}, {-1, 0, 0}, false),
               pln_t({-2, 0, 0}, {-1, 0, 0}, false),
               pln_t({0, 2, 0}, {0, 1, 0}, false),
               pln_t({0, -2, 0}, {0, 1, 0}, false),
               pln_t({0, 0, 2}, {0, 0, 1}, false),
               pln_t({0, 0, -2}, {0, 0, 1}, false),
               pln_t({2, 2, 2}, {ROOT3, ROOT3, ROOT3}, false),
               pln_t({-2, 2, 2}, {-ROOT3, ROOT3, ROOT3}, false),
               pln_t({2, -2, 2}, {ROOT3, -ROOT3, ROOT3}, false),
               pln_t({2, 2, -2}, {ROOT3, ROOT3, -ROOT3}, false),
               pln_t({-2, -2, -2}, {ROOT3, ROOT3, ROOT3}, false),

               pln_t({0, 0, 0}, {1, 0, 0}, true),
               pln_t({0, 0, 0}, {-1, 0, 0}, true),
               pln_t({0, 0, 0}, {0, 1, 0}, true),
               pln_t({0, 0, 0}, {0, -1, 0}, true),
               pln_t({0, 0, 0}, {0, 0, 1}, true),
               pln_t({0, 0, 0}, {0, 0, -1}, true),

               pln_t({1, 0, 0}, {1, 0, 0}, true),
               pln_t({0, 1, 0}, {0, 1, 0}, true),
               pln_t({0, 0, 1}, {0, 0, 1}, true),

               pln_t({2, 2, 2}, {-ROOT2, ROOT2, 0}, true),
               pln_t({-1, -1, -1}, {-ROOT3, -ROOT3, -ROOT3}, true),
               pln_t({1, -1, -1}, {ROOT3, -ROOT3, -ROOT3}, true),
               pln_t({-1, 1, -1}, {-ROOT3, ROOT3, -ROOT3}, true),
               pln_t({-1, -1, 1}, {-ROOT3, -ROOT3, ROOT3}, true),
               pln_t({1, 1, 1}, {ROOT3, ROOT3, ROOT3}, true)));

}  // namespace tests
}  // namespace open3d
