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

#include "open3d/geometry/BoundingVolume.h"
#include "tests/UnitTest.h"

using v3d_t = Eigen::Vector3d;

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

// Intersection tests for lines/rays against AABB
// ==========================================================================

// These basic line cases are all straightforward intersections which should
// work for both the slab and the exact methods
auto basic_line_cases = ::testing::Values(
        // Basic directional tests
        std::make_tuple(v3d_t(-6, 0, 0), v3d_t(-1, 0, 0), -7.),
        std::make_tuple(v3d_t(4, 0, 0), v3d_t(1, 0, 0), -5.),
        std::make_tuple(v3d_t(7, 0, 0), v3d_t(-1, 0, 0), 6.),
        std::make_tuple(v3d_t(0, 4, 0), v3d_t(0, -1, 0), 3.),
        std::make_tuple(v3d_t(0, -10, 0), v3d_t(0, -1, 0), -11.),
        std::make_tuple(v3d_t(0, -6, 0), v3d_t(0, 1, 0), 5.),
        std::make_tuple(v3d_t(0, 0, -9), v3d_t(0, 0, -1), -10.),
        std::make_tuple(v3d_t(0, 0, -5), v3d_t(0, 0, 1), 4.),
        std::make_tuple(v3d_t(0, 0, 8), v3d_t(0, 0, -1), 7.),

        // Interior tests
        std::make_tuple(v3d_t(0, 0, 0.5), v3d_t(0, 0, 1), -1.5),
        std::make_tuple(v3d_t(0, 0, 0), v3d_t(-1, 0, 0), -1.),

        // face tests
        std::make_tuple(v3d_t(1, 0, 0), v3d_t(-1, 0, 0), 0.),
        std::make_tuple(v3d_t(1, 0, 0), v3d_t(1, 0, 0), -2.),

        // Corner tests (direction isn't normalized here so that the return
        // value will be 1.)
        std::make_tuple(v3d_t(2, 2, 0), v3d_t(-1, -1, 1), 1.),
        std::make_tuple(v3d_t(0, 0, 2), v3d_t(1, 1, -1), 1.),

        // Non-intersecting tests
        std::make_tuple(v3d_t(-6, 0, 0), v3d_t(0, 1, 0), std::nan("")),
        std::make_tuple(v3d_t(0, 3, 0), v3d_t(0, 0, 1), std::nan("")),
        std::make_tuple(v3d_t(0, 0, 1.5), v3d_t(1, 0, 0), std::nan("")));

// These cases are degenerate for the slab method but should work for the
// exact method. These cases are designed to work for both line and ray
// intersections.
auto degenerate_line_cases =
        ::testing::Values(std::make_tuple(v3d_t(1, 2, 0), v3d_t(0, -1, 0), 1.),
                          std::make_tuple(v3d_t(-2, 0, 1), v3d_t(1, 0, 0), 1.));

// These are the basic line test cases adapated for ray intersections. Many of
// the intersecting cases become non-intersections.
auto basic_ray_cases = ::testing::Values(
        // Basic directional tests
        std::make_tuple(v3d_t(7, 0, 0), v3d_t(-1, 0, 0), 6.),
        std::make_tuple(v3d_t(0, 4, 0), v3d_t(0, -1, 0), 3.),
        std::make_tuple(v3d_t(0, -6, 0), v3d_t(0, 1, 0), 5.),
        std::make_tuple(v3d_t(0, 0, -5), v3d_t(0, 0, 1), 4.),
        std::make_tuple(v3d_t(0, 0, 8), v3d_t(0, 0, -1), 7.),

        // Interior tests
        std::make_tuple(v3d_t(0, 0, 0.5), v3d_t(0, 0, 1), 0.),
        std::make_tuple(v3d_t(0, 0, 0), v3d_t(-1, 0, 0), 0.),

        // face tests
        std::make_tuple(v3d_t(1, 0, 0), v3d_t(-1, 0, 0), 0.),
        std::make_tuple(v3d_t(1, 0, 0), v3d_t(1, 0, 0), 0.),

        // Corner tests (direction isn't normalized here so that the return
        // value will be 1.)
        std::make_tuple(v3d_t(2, 2, 0), v3d_t(-1, -1, 1), 1.),
        std::make_tuple(v3d_t(0, 0, 2), v3d_t(1, 1, -1), 1.),

        // Non-intersecting tests
        std::make_tuple(v3d_t(-6, 0, 0), v3d_t(-1, 0, 0), std::nan("")),
        std::make_tuple(v3d_t(4, 0, 0), v3d_t(1, 0, 0), std::nan("")),
        std::make_tuple(v3d_t(0, -10, 0), v3d_t(0, -1, 0), std::nan("")),
        std::make_tuple(v3d_t(0, 0, -9), v3d_t(0, 0, -1), std::nan("")),
        std::make_tuple(v3d_t(-6, 0, 0), v3d_t(0, 1, 0), std::nan("")),
        std::make_tuple(v3d_t(0, 3, 0), v3d_t(0, 0, 1), std::nan("")),
        std::make_tuple(v3d_t(0, 0, 1.5), v3d_t(1, 0, 0), std::nan("")));

// Test fixture for line/aabb checks
class LineAABBParameterChecks
    : public ::testing::TestWithParam<std::tuple<v3d_t, v3d_t, double>> {
protected:
    geometry::AxisAlignedBoundingBox box{{-1, -1, -1}, {1, 1, 1}};
};

// Test fixture for line/aabb checks for the exact method's handling of
// cases that are degenerate for the slab method
class LineAABBDegenParameterChecks : public LineAABBParameterChecks {};

// These tests verify that the slab method works against the basic line cases
TEST_P(LineAABBParameterChecks, SlabCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> line{std::get<0>(GetParam()),
                                            std::get<1>(GetParam())};

    double result = IntersectionTest::LineAABBSlabParam(line, box);
    if (std::isnan(std::get<2>(GetParam()))) {
        // This is because EXPECT_DOUBLE_EQ doesn't handle NaN correctly
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
    }
}

// These tests verify that the exact method works against the basic line cases
TEST_P(LineAABBParameterChecks, ExactCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> line{std::get<0>(GetParam()),
                                            std::get<1>(GetParam())};

    double result = IntersectionTest::LineAABBExactParam(line, box);
    if (std::isnan(std::get<2>(GetParam()))) {
        // This is because EXPECT_DOUBLE_EQ doesn't handle NaN correctly
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
    }
}

// These tests check the exact intersection method against conditions that
// would be degenerate for the slab method
TEST_P(LineAABBDegenParameterChecks, ExactCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> line{std::get<0>(GetParam()),
                                            std::get<1>(GetParam())};

    double result = IntersectionTest::LineAABBExactParam(line, box);
    EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
}

INSTANTIATE_TEST_CASE_P(LineAABBTests,
                        LineAABBParameterChecks,
                        basic_line_cases);

INSTANTIATE_TEST_CASE_P(LineAABBTests,
                        LineAABBDegenParameterChecks,
                        degenerate_line_cases);

// Test fixture for ray/aabb checks
class RayAABBParameterChecks
    : public ::testing::TestWithParam<std::tuple<v3d_t, v3d_t, double>> {
protected:
    geometry::AxisAlignedBoundingBox box{{-1, -1, -1}, {1, 1, 1}};
};

// Test fixture for ray/aabb checks that are degenerate for the slab method
class RayAABBDegenParameterChecks : public RayAABBParameterChecks {};

// These tests are to check the slab method against the basic ray test cases
TEST_P(RayAABBParameterChecks, SlabCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> ray{std::get<0>(GetParam()),
                                           std::get<1>(GetParam())};

    double result = IntersectionTest::RayAABBSlabParam(ray, box);
    if (std::isnan(std::get<2>(GetParam()))) {
        // This is because EXPECT_DOUBLE_EQ doesn't handle NaN correctly
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
    }
}

// These tests are to check the exact method against the basic ray test cases
TEST_P(RayAABBParameterChecks, ExactCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> ray{std::get<0>(GetParam()),
                                           std::get<1>(GetParam())};

    double result = IntersectionTest::RayAABBExactParam(ray, box);
    if (std::isnan(std::get<2>(GetParam()))) {
        // This is because EXPECT_DOUBLE_EQ doesn't handle NaN correctly
        EXPECT_TRUE(std::isnan(result));
    } else {
        EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
    }
}

// These tests are to check the exact method against the ray test cases which
// are degenerate for the slab method
TEST_P(RayAABBDegenParameterChecks, ExactCheckIntersections) {
    using namespace geometry;
    Eigen::ParametrizedLine<double, 3> ray{std::get<0>(GetParam()),
                                           std::get<1>(GetParam())};

    double result = IntersectionTest::RayAABBExactParam(ray, box);
    EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), result);
}

INSTANTIATE_TEST_CASE_P(RayAABBTests, RayAABBParameterChecks, basic_ray_cases);

INSTANTIATE_TEST_CASE_P(RayAABBTests,
                        RayAABBDegenParameterChecks,
                        degenerate_line_cases);

}  // namespace tests
}  // namespace open3d
