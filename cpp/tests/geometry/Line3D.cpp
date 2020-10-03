// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/geometry/Line3D.h"

#include "tests/UnitTest.h"

using namespace open3d::geometry;
using namespace ::testing;
using opt_d_t = open3d::utility::optional<double>;

using v_t = Eigen::Vector3d;
using plane_t = Eigen::Hyperplane<double, 3>;

// Test case parameter types
using lt_t = Line3D::LineType;

// Test data for failed intersection parameter tests
// line-type, plane-normal, line-origin, line-dir/seg-endpoint
using fl_intr_t = std::tuple<lt_t, v_t, v_t, v_t>;

// Test data for intersection parameter tests
// line-type, plane-normal, line-origin, line-dir/seg-endpoint, expected-param
using intr_t = std::tuple<lt_t, v_t, v_t, v_t, double>;

// Test data for projection parameter tests
// line-type, test-point, line-origin, line-dir/seg-endpoint, expected-param
using proj_t = std::tuple<lt_t, v_t, v_t, v_t, double>;

// Test data for intersections against AABBs
// line-type, line-origin, line-dir/seg-endpoint, optional expected parameter
using ab_t = std::tuple<lt_t, v_t, v_t, opt_d_t>;

// Test data for closest parameter tests
// line0-type, line0-origin, line0-dir/endpoint, line1-type, line1-origin,
// line1-dir/endpoint, line0-expected, line1-expected
using cp_t = std::tuple<lt_t, v_t, v_t, lt_t, v_t, v_t, double, double>;

// Factory function to build appropriate type from enum
std::shared_ptr<Line3D> LineFactory(lt_t type, const v_t& v0, const v_t& v1) {
    if (type == lt_t::Line) {
        return std::make_shared<Line3D>(v0, v1);
    } else if (type == lt_t::Ray) {
        return std::make_shared<Ray3D>(v0, v1);
    } else if (type == lt_t::Segment) {
        return std::make_shared<Segment3D>(v0, v1);
    } else {
        throw std::exception();
    }
}

namespace open3d {
namespace tests {

// Line Transformation Tests
// ============================================================================
TEST(Line3D, Transform) {
    Line3D line{{0, 1, 0}, {1, 0, 0}};
    auto t = Eigen::Translation<double, 3>(1, 0, 0) *
             Eigen::AngleAxis<double>(EIGEN_PI / 2., Eigen::Vector3d::UnitZ());
    line.Transform(t);

    // For some reason this does not work with ExpectEQ, even though the
    // difference is ~1e-17 and the threshold is 1e-6
    EXPECT_TRUE((line.Origin() - Eigen::Vector3d{0, 0, 0}).norm() < 1e-10);
    ExpectEQ(line.Direction(), Eigen::Vector3d{0, 1, 0});
}

// Segment Transformation Tests
// ============================================================================
TEST(Segment3D, Transform) {
    Segment3D seg{{0, 1, 0}, {1, 1, 0}};
    auto t = Eigen::Translation<double, 3>(1, 0, 0) *
             Eigen::AngleAxis<double>(EIGEN_PI / 2., Eigen::Vector3d::UnitZ());
    seg.Transform(t);

    // For some reason this does not work with ExpectEQ, even though the
    // difference is ~1e-17 and the threshold is 1e-6
    EXPECT_TRUE((seg.Origin() - Eigen::Vector3d{0, 0, 0}).norm() < 1e-10);
    EXPECT_TRUE((seg.EndPoint() - Eigen::Vector3d{0, 1, 0}).norm() < 1e-10);
    ExpectEQ(seg.Direction(), Eigen::Vector3d{0, 1, 0});
}

// Special Segment3D Tests
// ============================================================================
TEST(Segment3D, ConstructedNormalized) {
    // In order to override ambiguity with Eigen's handling of direction
    // normalization, the Segment3D constructor normalizes the direction on
    // creation. This allows the length to be directly compared to line
    // parameters without fear of breaking behavior.  This is only necessary for
    // segments, since lines and rays are by definition infinite.
    Segment3D seg{{0, 0, 0}, {10, 0, 0}};

    EXPECT_EQ(10., seg.Length());
}

TEST(Segment3D, GetBoundingBox0) {
    Segment3D seg{{-3, -2, -1}, {3, 2, 1}};
    AxisAlignedBoundingBox expected{{-3, -2, -1}, {3, 2, 1}};

    ExpectEQ(expected.min_bound_, seg.GetBoundingBox().min_bound_);
    ExpectEQ(expected.max_bound_, seg.GetBoundingBox().max_bound_);
}

TEST(Segment3D, GetBoundingBox1) {
    Segment3D seg{{3, 2, 1}, {-3, -2, -1}};
    AxisAlignedBoundingBox expected{{-3, -2, -1}, {3, 2, 1}};

    ExpectEQ(expected.min_bound_, seg.GetBoundingBox().min_bound_);
    ExpectEQ(expected.max_bound_, seg.GetBoundingBox().max_bound_);
}

// Intersection Parameter Tests
// ============================================================================
class FailedIntersectionParamTests : public TestWithParam<fl_intr_t> {};

TEST_P(FailedIntersectionParamTests, CheckFailedPlaneIntersections) {
    auto line_type = std::get<0>(GetParam());
    plane_t plane{std::get<1>(GetParam()), 0};
    const auto& l0 = std::get<2>(GetParam());
    const auto& l1 = std::get<3>(GetParam());

    auto line = LineFactory(line_type, l0, l1);

    auto result = line->IntersectionParameter(plane);
    EXPECT_FALSE(result.has_value());
}

INSTANTIATE_TEST_CASE_P(
        Fail,
        FailedIntersectionParamTests,
        Values(
                // Test parameters contain in this order:
                // line type, plane normal, line origin, line-dir/seg-endpoint

                // Lines should fail the plane intersections only when they are
                // parallel to the plane. Test plane normals both ways to make
                // sure that the code handles Eigen's +inf and -inf
                fl_intr_t{lt_t::Line, {0, 0, -1}, {0, 0, 1}, {1, 0, 0}},
                fl_intr_t{lt_t::Line, {0, 0, 1}, {0, 0, 1}, {1, 0, 0}},
                fl_intr_t{lt_t::Line, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}},
                fl_intr_t{lt_t::Line, {-1, 0, 0}, {1, 0, 0}, {0, 1, 0}},

                // Rays should fail where lines fail and also when the plane
                // lies behind the origin
                fl_intr_t{lt_t::Ray, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}},
                fl_intr_t{lt_t::Ray, {-1, 0, 0}, {1, 0, 0}, {0, 1, 0}},
                fl_intr_t{lt_t::Ray, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}},

                // Segments should fail in the same cases as rays and lines,
                // plus when the plane lies past the end point
                fl_intr_t{lt_t::Segment, {1, 0, 0}, {1, 0, 0}, {1, 1, 0}},
                fl_intr_t{lt_t::Segment, {-1, 0, 0}, {1, 0, 0}, {1, 1, 0}},

                fl_intr_t{lt_t::Segment, {-1, 0, 0}, {1, 0, 0}, {2, 0, 0}},
                fl_intr_t{lt_t::Segment, {1, 0, 0}, {-2, 0, 0}, {-1, 0, 0}}));

class IntersectionParamTests : public TestWithParam<intr_t> {};

TEST_P(IntersectionParamTests, CheckPlaneIntersections) {
    auto line_type = std::get<0>(GetParam());
    plane_t plane{std::get<1>(GetParam()), 0};
    const auto& l0 = std::get<2>(GetParam());
    const auto& l1 = std::get<3>(GetParam());
    double expected = std::get<4>(GetParam());

    auto line = LineFactory(line_type, l0, l1);

    auto result = line->IntersectionParameter(plane);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(expected, result.value());
}

INSTANTIATE_TEST_CASE_P(
        Pass,
        IntersectionParamTests,
        Values(
                // Test parameters contain in this order:
                // line type, plane normal, line origin, line-dir/seg-end,
                // expected param
                intr_t{lt_t::Line, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, -1},
                intr_t{lt_t::Line, {0, 1, 0}, {-2, -2, 0}, {0, 1, 0}, 2},

                intr_t{lt_t::Ray, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}, 1},
                intr_t{lt_t::Ray, {0, 1, 0}, {-2, -2, 0}, {0, 1, 0}, 2},

                intr_t{lt_t::Segment, {1, 0, 0}, {1, 0, 0}, {-2, 0, 0}, 1},
                intr_t{lt_t::Segment, {0, 1, 0}, {-2, -2, 0}, {-2, 1, 0}, 2}));

// Projection Parameter Tests
// ============================================================================
class ProjectionParamTests : public TestWithParam<proj_t> {};

TEST_P(ProjectionParamTests, CheckProjectionParameters) {
    auto line_type = std::get<0>(GetParam());
    const auto& test_point = std::get<1>(GetParam());
    const auto& l0 = std::get<2>(GetParam());
    const auto& l1 = std::get<3>(GetParam());
    double expected = std::get<4>(GetParam());

    auto line = LineFactory(line_type, l0, l1);
    double result = line->ProjectionParameter(test_point);

    EXPECT_EQ(expected, result);
}

INSTANTIATE_TEST_CASE_P(
        Tests,
        ProjectionParamTests,
        Values(
                // Lines will project to any point, positive or negative
                proj_t{lt_t::Line, {1, 1, 0}, {0, 0, 0}, {1, 0, 0}, 1},
                proj_t{lt_t::Line, {1, 1, 0}, {-4, 0, 0}, {1, 0, 0}, 5},
                proj_t{lt_t::Line, {1, 1, 0}, {4, 0, 0}, {1, 0, 0}, -3},

                // Rays will project to any positive point, but negative points
                // will return 0
                proj_t{lt_t::Ray, {1, 1, 0}, {0, 0, 0}, {0, 1, 0}, 1},
                proj_t{lt_t::Ray, {1, 1, 0}, {0, -4, 0}, {0, 1, 0}, 5},
                proj_t{lt_t::Ray, {1, 1, 0}, {0, 4, 0}, {0, 1, 0}, 0},

                // Line segments will only project between 0 and their length
                proj_t{lt_t::Segment, {1, 1, 0}, {0, 0, 0}, {0, 2, 0}, 1},
                proj_t{lt_t::Segment, {1, 1, 0}, {0, 0, 0}, {0, 1, 0}, 1},
                proj_t{lt_t::Segment, {1, 1, 0}, {0, -4, 0}, {0, 0, 0}, 4},
                proj_t{lt_t::Segment, {1, 1, 0}, {0, 4, 0}, {0, 8, 0}, 0}));

// AxisAlignedBoundingBox Tests
// ============================================================================

// These are cases which should work for both the exact method and the slab
// method
auto basic_cases = Values(
        // Basic directional tests
        ab_t{lt_t::Line, {-6, 0, 0}, {-1, 0, 0}, -7.},
        ab_t{lt_t::Line, {4, 0, 0}, {1, 0, 0}, -5.},
        ab_t{lt_t::Line, {7, 0, 0}, {-1, 0, 0}, 6.},
        ab_t{lt_t::Line, {0, 4, 0}, {0, -1, 0}, 3.},
        ab_t{lt_t::Line, {0, -10, 0}, {0, -1, 0}, -11.},
        ab_t{lt_t::Line, {0, -6, 0}, {0, 1, 0}, 5.},
        ab_t{lt_t::Line, {0, 0, -9}, {0, 0, -1}, -10.},
        ab_t{lt_t::Line, {0, 0, -5}, {0, 0, 1}, 4.},
        ab_t{lt_t::Line, {0, 0, 8}, {0, 0, -1}, 7.},

        ab_t{lt_t::Ray, {7, 0, 0}, {-1, 0, 0}, 6.},
        ab_t{lt_t::Ray, {0, 4, 0}, {0, -1, 0}, 3.},
        ab_t{lt_t::Ray, {0, -6, 0}, {0, 1, 0}, 5.},
        ab_t{lt_t::Ray, {0, 0, -5}, {0, 0, 1}, 4.},
        ab_t{lt_t::Ray, {0, 0, 8}, {0, 0, -1}, 7.},

        ab_t{lt_t::Segment, {7, 0, 0}, {0, 0, 0}, 6.},
        ab_t{lt_t::Segment, {0, 4, 0}, {0, 0, 0}, 3.},
        ab_t{lt_t::Segment, {0, -6, 0}, {0, 6, 0}, 5.},
        ab_t{lt_t::Segment, {0, 0, -5}, {0, 0, 1}, 4.},
        ab_t{lt_t::Segment, {0, 0, 8}, {0, 0, -1}, 7.},

        // Interior tests
        ab_t{lt_t::Line, {0, 0, 0.5}, {0, 0, 1}, -1.5},
        ab_t{lt_t::Line, {0, 0, 0}, {-1, 0, 0}, -1.},
        ab_t{lt_t::Ray, {0, 0, 0.5}, {0, 0, 1}, 0.},
        ab_t{lt_t::Ray, {0, 0, 0}, {-1, 0, 0}, 0.},
        ab_t{lt_t::Segment, {0, 0, 0.5}, {0, 0, 3}, 0.},
        ab_t{lt_t::Segment, {0, 0, 0}, {-3, 0, 0}, 0.},

        // face tests
        ab_t{lt_t::Line, {1, 0, 0}, {-1, 0, 0}, 0.},
        ab_t{lt_t::Line, {1, 0, 0}, {1, 0, 0}, -2.},
        ab_t{lt_t::Ray, {1, 0, 0}, {-1, 0, 0}, 0.},
        ab_t{lt_t::Segment, {1, 0, 0}, {2, 0, 0}, 0.},

        // Corner tests (direction isn't normalized here so that the return
        // value will be 1.}
        ab_t{lt_t::Line, {2, 2, 0}, {-1, -1, 1}, 1.},
        ab_t{lt_t::Line, {0, 0, 2}, {1, 1, -1}, 1.},
        ab_t{lt_t::Ray, {2, 2, 0}, {-1, -1, 1}, 1.},
        ab_t{lt_t::Ray, {0, 0, 2}, {1, 1, -1}, 1.},

        // Segment non-normalization doesn't work, so we'll start the segment
        // at the corner point
        ab_t{lt_t::Segment, {1, 1, 1}, {2, 2, 0}, 0.},
        ab_t{lt_t::Segment, {-1, 1, 1}, {-2, 2, 0}, 0.},

        // Non-intersecting tests
        ab_t{lt_t::Segment, {3, 0, 0}, {4, 0, 0}, {}},
        ab_t{lt_t::Segment, {4, 0, 0}, {3, 0, 0}, {}},
        ab_t{lt_t::Segment, {0, 3, -1}, {0, 3, 1}, {}},

        ab_t{lt_t::Line, {-6, 0, 0}, {0, 1, 0}, {}},
        ab_t{lt_t::Line, {0, 3, 0}, {0, 0, 1}, {}},
        ab_t{lt_t::Line, {0, 0, 1.5}, {1, 0, 0}, {}},
        ab_t{lt_t::Ray, {-6, 0, 0}, {-1, 0, 0}, {}},
        ab_t{lt_t::Ray, {4, 0, 0}, {1, 0, 0}, {}},
        ab_t{lt_t::Ray, {0, -10, 0}, {0, -1, 0}, {}},
        ab_t{lt_t::Ray, {0, 0, -9}, {0, 0, -1}, {}});

class LineAABBParamTests : public TestWithParam<ab_t> {
protected:
    AxisAlignedBoundingBox box{{-1, -1, -1}, {1, 1, 1}};
};

TEST_P(LineAABBParamTests, CheckExact) {
    auto line_type = std::get<0>(GetParam());
    const auto& l0 = std::get<1>(GetParam());
    const auto& l1 = std::get<2>(GetParam());
    const auto& expected = std::get<3>(GetParam());

    auto line = LineFactory(line_type, l0, l1);
    auto result = line->ExactAABB(box);

    if (expected.has_value()) {
        EXPECT_TRUE(result.has_value());
        if (result.has_value()) {
            EXPECT_EQ(expected.value(), result.value());
        }
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

TEST_P(LineAABBParamTests, CheckSlab) {
    auto line_type = std::get<0>(GetParam());
    const auto& l0 = std::get<1>(GetParam());
    const auto& l1 = std::get<2>(GetParam());
    const auto& expected = std::get<3>(GetParam());

    auto line = LineFactory(line_type, l0, l1);
    auto result = line->SlabAABB(box);

    if (expected.has_value()) {
        EXPECT_TRUE(result.has_value());
        if (result.has_value()) {
            EXPECT_EQ(expected.value(), result.value());
        }
    } else {
        EXPECT_FALSE(result.has_value());
    }
}

INSTANTIATE_TEST_CASE_P(Checks, LineAABBParamTests, basic_cases);

class DegenerateLineAABBTests : public LineAABBParamTests {};
TEST_P(DegenerateLineAABBTests, CheckExact) {
    auto line_type = std::get<0>(GetParam());
    const auto& l0 = std::get<1>(GetParam());
    const auto& l1 = std::get<2>(GetParam());
    const auto& expected = std::get<3>(GetParam());

    auto line = LineFactory(line_type, l0, l1);
    auto result = line->ExactAABB(box);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(expected.value(), result.value());
}

INSTANTIATE_TEST_CASE_P(Checks,
                        DegenerateLineAABBTests,
                        Values(ab_t{lt_t::Line, {1, 2, 0}, {0, -1, 0}, 1.},
                               ab_t{lt_t::Line, {-2, 0, 1}, {1, 0, 0}, 1.},
                               ab_t{lt_t::Ray, {1, 2, 0}, {0, -1, 0}, 1.},
                               ab_t{lt_t::Ray, {-2, 0, 1}, {1, 0, 0}, 1.},
                               ab_t{lt_t::Segment, {1, 2, 0}, {1, 0, 0}, 1.},
                               ab_t{lt_t::Segment, {-2, 0, 1}, {0, 0, 1}, 1.}));

// Line/Ray/Segment Closest Point Tests
// ============================================================================
class ClosestPointTests : public TestWithParam<cp_t> {};

TEST_P(ClosestPointTests, CheckClosestPoints) {
    auto l0_type = std::get<0>(GetParam());
    auto l0_origin = std::get<1>(GetParam());
    auto l0_dir = std::get<2>(GetParam());

    auto l1_type = std::get<3>(GetParam());
    auto l1_origin = std::get<4>(GetParam());
    auto l1_dir = std::get<5>(GetParam());

    double e0 = std::get<6>(GetParam());
    double e1 = std::get<7>(GetParam());

    auto l0 = LineFactory(l0_type, l0_origin, l0_dir);
    auto l1 = LineFactory(l1_type, l1_origin, l1_dir);

    auto result = l0->ClosestParameters(*l1);

    EXPECT_DOUBLE_EQ(std::get<0>(result), e0);
    EXPECT_DOUBLE_EQ(std::get<1>(result), e1);
}

INSTANTIATE_TEST_CASE_P(LineTests,
                        ClosestPointTests,
                        Values(
                                // Line to line
                                cp_t(lt_t::Line,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     lt_t::Line,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Line,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {0, 1, 0},
                                     {0, 1, 0},
                                     lt_t::Line,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     -1,
                                     -1),
                                cp_t(lt_t::Line,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Line,
                                     {0, 1, 0},
                                     {0, 1, 0},
                                     -1,
                                     -1),

                                // Line to line parallel
                                cp_t(lt_t::Line,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Line,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     0,
                                     -1),
                                cp_t(lt_t::Line,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     lt_t::Line,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     0,
                                     1),

                                // Line to ray
                                cp_t(lt_t::Line,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     lt_t::Ray,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Ray,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {0, 1, 0},
                                     {0, 1, 0},
                                     lt_t::Ray,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     -1,
                                     0),
                                cp_t(lt_t::Line,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Ray,
                                     {0, 1, 0},
                                     {0, 1, 0},
                                     -1,
                                     0),

                                // Line to ray parallel
                                cp_t(lt_t::Ray,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Line,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     0,
                                     -1),
                                cp_t(lt_t::Line,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Ray,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     1,
                                     0),
                                cp_t(lt_t::Line,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     lt_t::Ray,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     0,
                                     1),

                                // Line to segment
                                cp_t(lt_t::Line,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     lt_t::Segment,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {-1, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Segment,
                                     {0, -1, 0},
                                     {0, 1, 0},
                                     1,
                                     1),
                                cp_t(lt_t::Segment,
                                     {0, 1, 0},
                                     {0, 2, 0},
                                     lt_t::Line,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     0,
                                     -1),
                                cp_t(lt_t::Segment,
                                     {0, 2, 0},
                                     {0, 1, 0},
                                     lt_t::Line,
                                     {1, 0, 0},
                                     {1, 0, 0},
                                     1,
                                     -1),

                                // Line to segment parallel
                                cp_t(lt_t::Line,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Segment,
                                     {1, 0, 1},
                                     {2, 0, 1},
                                     1,
                                     0),
                                cp_t(lt_t::Line,
                                     {0, 0, 0},
                                     {1, 0, 0},
                                     lt_t::Segment,
                                     {2, 0, 1},
                                     {1, 0, 1},
                                     1,
                                     1),
                                cp_t(lt_t::Line,
                                     {1, 0, 1},
                                     {1, 0, 0},
                                     lt_t::Segment,
                                     {0, 0, 0},
                                     {0.5, 0, 0},
                                     -0.5,
                                     0.5)));

const double root2 = 1.41421356237309504880168;
INSTANTIATE_TEST_CASE_P(
        RayTests,
        ClosestPointTests,
        Values(
                // Ray to ray
                cp_t(lt_t::Ray,
                     {0, -1, 0},
                     {0, 1, 0},
                     lt_t::Ray,
                     {-1, 0, 0},
                     {1, 0, 0},
                     1,
                     1),
                cp_t(lt_t::Ray,
                     {-1, 0, 0},
                     {1, 0, 0},
                     lt_t::Ray,
                     {0, -1, 0},
                     {0, 1, 0},
                     1,
                     1),
                cp_t(lt_t::Ray,
                     {0, 1, 0},
                     {0, 1, 0},
                     lt_t::Ray,
                     {1, 0, 0},
                     {1, 0, 0},
                     0,
                     0),
                cp_t(lt_t::Ray,
                     {1, 0, 0},
                     {1, 0, 0},
                     lt_t::Ray,
                     {0, 1, 0},
                     {0, 1, 0},
                     0,
                     0),

                // Ray to ray out of bounds behind origin, these cases test the
                // clamp/project/clamp/project procedure
                cp_t(lt_t::Ray,
                     {0, 1, 0},
                     {1, 0, 0},
                     lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     2,
                     0),
                cp_t(lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     lt_t::Ray,
                     {0, 1, 0},
                     {1, 0, 0},
                     0,
                     2),
                cp_t(lt_t::Ray,
                     {1.5, 1, 0},
                     {1, 0, 0},
                     lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     .5,
                     0),
                cp_t(lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     lt_t::Ray,
                     {1.5, 1, 0},
                     {1, 0, 0},
                     0,
                     .5),

                // Ray to ray parallel
                cp_t(lt_t::Ray,
                     {0, 0, 0},
                     {1, 0, 0},
                     lt_t::Ray,
                     {1, 0, 1},
                     {1, 0, 0},
                     1,
                     0),
                cp_t(lt_t::Ray,
                     {1, 0, 1},
                     {1, 0, 0},
                     lt_t::Ray,
                     {0, 0, 0},
                     {1, 0, 0},
                     0,
                     1),

                // Ray to segment
                cp_t(lt_t::Ray,
                     {0, -1, 0},
                     {0, 1, 0},
                     lt_t::Segment,
                     {-1, 0, 0},
                     {1, 0, 0},
                     1,
                     1),
                cp_t(lt_t::Ray,
                     {-1, 0, 0},
                     {1, 0, 0},
                     lt_t::Segment,
                     {0, -1, 0},
                     {0, 1, 0},
                     1,
                     1),
                cp_t(lt_t::Ray,
                     {0, 1, 0},
                     {0, 1, 0},
                     lt_t::Segment,
                     {1, 0, 0},
                     {2, 0, 0},
                     0,
                     0),
                cp_t(lt_t::Ray,
                     {1, 0, 0},
                     {1, 0, 0},
                     lt_t::Segment,
                     {0, 1, 0},
                     {0, 2, 0},
                     0,
                     0),

                // Ray to ray out of bounds behind origin, these cases test the
                // clamp/project/clamp/project procedure
                cp_t(lt_t::Ray,
                     {0, 1, 0},
                     {1, 0, 0},
                     lt_t::Segment,
                     {2, 2, 0},
                     {3, 3, 0},
                     2,
                     0),
                cp_t(lt_t::Ray,
                     {0, 1, 0},
                     {1, 0, 0},
                     lt_t::Segment,
                     {3, 3, 0},
                     {2, 2, 0},
                     2,
                     root2),
                cp_t(lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     lt_t::Segment,
                     {0, 1, 0},
                     {0.5, 1, 0},
                     0,
                     0.5),
                cp_t(lt_t::Ray,
                     {2, 2, 0},
                     {1, 1, 0},
                     lt_t::Segment,
                     {0.5, 1, 0},
                     {0, 1, 0},
                     0,
                     0)));

INSTANTIATE_TEST_CASE_P(
        SegmentTests,
        ClosestPointTests,
        Values(
                // Regular segment to segment intersections
                cp_t(lt_t::Segment,
                     {0, -1, 0},
                     {0, 1, 0},
                     lt_t::Segment,
                     {-1, 0, 0},
                     {1, 0, 0},
                     1,
                     1),
                cp_t(lt_t::Segment,
                     {-1, 0, 0},
                     {1, 0, 0},
                     lt_t::Segment,
                     {0, -1, 0},
                     {0, 1, 0},
                     1,
                     1),

                // Tests corresponding with special case b from "Real-Time
                // Collision Detection" Figure 5.9b, p148
                cp_t(lt_t::Segment,
                     {0, 1, 0},
                     {3, 1, 0},
                     lt_t::Segment,
                     {2, 2, 0},
                     {3, 3, 0},
                     2,
                     0),
                cp_t(lt_t::Segment,
                     {0, 1, 0},
                     {3, 1, 0},
                     lt_t::Segment,
                     {3, 3, 0},
                     {2, 2, 0},
                     2,
                     root2),
                cp_t(lt_t::Segment,
                     {2, 2, 0},
                     {3, 3, 0},
                     lt_t::Segment,
                     {0, 1, 0},
                     {3, 1, 0},
                     0,
                     2),
                cp_t(lt_t::Segment,
                     {3, 3, 0},
                     {2, 2, 0},
                     lt_t::Segment,
                     {0, 1, 0},
                     {3, 1, 0},
                     root2,
                     2),

                // Tests corresponding with special case b from "Real-Time
                // Collision Detection" Figure 5.9c, p148
                cp_t(lt_t::Segment,
                     {1.5, 1, 0},
                     {3, 1, 0},
                     lt_t::Segment,
                     {2, 2, 0},
                     {3, 3, 0},
                     0.5,
                     0),
                cp_t(lt_t::Segment,
                     {1.5, 1, 0},
                     {3, 1, 0},
                     lt_t::Segment,
                     {3, 3, 0},
                     {2, 2, 0},
                     0.5,
                     root2),
                cp_t(lt_t::Segment,
                     {2, 2, 0},
                     {3, 3, 0},
                     lt_t::Segment,
                     {1.5, 1, 0},
                     {3, 1, 0},
                     0,
                     0.5),
                cp_t(lt_t::Segment,
                     {3, 3, 0},
                     {2, 2, 0},
                     lt_t::Segment,
                     {1.5, 1, 0},
                     {3, 1, 0},
                     root2,
                     0.5)));
}  // namespace tests
}  // namespace open3d
