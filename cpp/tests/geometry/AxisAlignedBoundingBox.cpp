// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/geometry/BoundingVolume.h"
#include "tests/Tests.h"

using namespace open3d::geometry;
using namespace ::testing;

using Eigen::Vector3d;

using emp_t = std::tuple<Vector3d, Vector3d, bool>;
using dbl_t = std::tuple<Vector3d, Vector3d, double>;

namespace open3d {
namespace tests {

TEST(AxisAlignedBoundingBox, EmptyConstructor) {
    AxisAlignedBoundingBox box;

    EXPECT_TRUE(box.IsEmpty());
    ExpectEQ(box.min_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.color_, Vector3d{1, 1, 1});
}

TEST(AxisAlignedBoundingBox, ConstructorBounds) {
    AxisAlignedBoundingBox box({1, 2, 3}, {4, 5, 6});
    ExpectEQ(box.min_bound_, Vector3d{1, 2, 3});
    ExpectEQ(box.max_bound_, Vector3d{4, 5, 6});
    ExpectEQ(box.color_, Vector3d{1, 1, 1});
}

TEST(AxisAlignedBoundingBox, BoundAccessors) {
    AxisAlignedBoundingBox box({1, 2, 3}, {4, 5, 6});
    ExpectEQ(box.GetMinBound(), Vector3d{1, 2, 3});
    ExpectEQ(box.GetMaxBound(), Vector3d{4, 5, 6});
}

TEST(AxisAlignedBoundingBox, GetCenter) {
    AxisAlignedBoundingBox box({1, 2, 3}, {5, 6, 7});
    ExpectEQ(box.GetCenter(), Vector3d{3, 4, 5});
}

TEST(AxisAlignedBoundingBox, GetExtent) {
    AxisAlignedBoundingBox box({1, 2, 3}, {4, 6, 8});
    ExpectEQ(box.GetExtent(), Vector3d{3, 4, 5});
}

TEST(AxisAlignedBoundingBox, GetHalfExtent) {
    AxisAlignedBoundingBox box({1, 2, 3}, {7, 10, 13});
    ExpectEQ(box.GetHalfExtent(), Vector3d{3, 4, 5});
}

TEST(AxisAlignedBoundingBox, Clear) {
    AxisAlignedBoundingBox box({1, 2, 3}, {7, 10, 13});
    box.Clear();
    EXPECT_TRUE(box.IsEmpty());
    ExpectEQ(box.min_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{0, 0, 0});
}

TEST(AxisAlignedBoundingBox, GetXPercentage) {
    AxisAlignedBoundingBox box({1, 2, 3}, {2, 3, 4});
    auto value = box.GetXPercentage(1.6);
    EXPECT_NEAR(0.6, value, 1e-6);
}

TEST(AxisAlignedBoundingBox, GetYPercentage) {
    AxisAlignedBoundingBox box({1, 2, 3}, {2, 3, 4});
    auto value = box.GetYPercentage(2.2);
    EXPECT_NEAR(0.2, value, 1e-6);
}

TEST(AxisAlignedBoundingBox, GetZPercentage) {
    AxisAlignedBoundingBox box({1, 2, 3}, {2, 3, 4});
    auto value = box.GetZPercentage(3.7);
    EXPECT_NEAR(0.7, value, 1e-6);
}

TEST(AxisAlignedBoundingBox, GetBoxPoints) {
    AxisAlignedBoundingBox box({1, 2, 3}, {4, 5, 6});
    auto points = box.GetBoxPoints();
    std::vector<Vector3d> expected{{1, 2, 3}, {1, 2, 6}, {1, 5, 6}, {1, 5, 3},
                                   {4, 2, 3}, {4, 2, 6}, {4, 5, 3}, {4, 5, 6}};

    for (auto exp : expected) {
        auto locate = [&exp](const Vector3d& v) {
            return (v - exp).norm() < 1e-6;
        };
        auto found = std::find_if(points.begin(), points.end(), locate);
        EXPECT_NE(found, points.end());
    }

    EXPECT_EQ(8, points.size());
}

// Point Indices within Bounds
TEST(AxisAlignedBoundingBox, PointIndiciesInBounds) {
    AxisAlignedBoundingBox box({1, 2, 3}, {4, 5, 6});
    std::vector<Vector3d> points{{2, 3, 4},  {1, 2, 3}, {1, 2, 6},  {1, 5, 6},
                                 {1, 5, 3},  {4, 2, 3}, {4, 2, 6},  {4, 5, 3},
                                 {4, 5, 6},  {0, 3, 4}, {10, 3, 4}, {2, 0, 4},
                                 {2, 10, 4}, {2, 3, 0}, {2, 3, 10}};
    std::vector<size_t> expected{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto indices = box.GetPointIndicesWithinBoundingBox(points);
    std::sort(indices.begin(), indices.end());

    EXPECT_EQ(expected.size(), indices.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], indices[i]);
    }
}

TEST(AxisAlignedBoundingBox, CreateFromPoints) {
    std::vector<Vector3d> points{{2, 3, 4}, {1, 2, 6}, {1, 5, 6}, {1, 5, 3},
                                 {4, 2, 3}, {4, 2, 6}, {4, 5, 3}};
    auto box = AxisAlignedBoundingBox::CreateFromPoints(points);
    ExpectEQ(box.min_bound_, Vector3d{1, 2, 3});
    ExpectEQ(box.max_bound_, Vector3d{4, 5, 6});
}

TEST(AxisAlignedBoundingBox, ScaleBox1) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 4, 5});
    box.Scale(2.0, {2, 3, 4});
    ExpectEQ(box.min_bound_, Vector3d{0, 1, 2});
    ExpectEQ(box.max_bound_, Vector3d{4, 5, 6});
}

TEST(AxisAlignedBoundingBox, ScaleBox2) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 4, 5});
    box.Scale(2.0, {1, 2, 3});
    ExpectEQ(box.min_bound_, Vector3d{1, 2, 3});
    ExpectEQ(box.max_bound_, Vector3d{5, 6, 7});
}

TEST(AxisAlignedBoundingBox, TranslateRelative) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 4, 5});
    box.Translate({3, 4, 5}, true);
    ExpectEQ(box.min_bound_, Vector3d{4, 6, 8});
    ExpectEQ(box.max_bound_, Vector3d{6, 8, 10});
}

TEST(AxisAlignedBoundingBox, TranslateAbsolute) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 4, 5});
    box.Translate({-10, -15, -20}, false);
    ExpectEQ(box.min_bound_, Vector3d{-11, -16, -21});
    ExpectEQ(box.max_bound_, Vector3d{-9, -14, -19});
}

TEST(AxisAlignedBoundingBox, PlusEqualsEmpty1) {
    AxisAlignedBoundingBox box({1, 2, 3}, {1, 4, 5});
    AxisAlignedBoundingBox box1({2, 2, 2}, {3, 3, 3});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{2, 2, 2});
    ExpectEQ(box.max_bound_, Vector3d{3, 3, 3});
}

TEST(AxisAlignedBoundingBox, PlusEqualsEmpty2) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 2, 5});
    AxisAlignedBoundingBox box1({2, 2, 2}, {3, 3, 3});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{2, 2, 2});
    ExpectEQ(box.max_bound_, Vector3d{3, 3, 3});
}

TEST(AxisAlignedBoundingBox, PlusEqualsEmpty3) {
    AxisAlignedBoundingBox box({1, 2, 3}, {3, 4, 3});
    AxisAlignedBoundingBox box1({2, 2, 2}, {3, 3, 3});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{2, 2, 2});
    ExpectEQ(box.max_bound_, Vector3d{3, 3, 3});
}

TEST(AxisAlignedBoundingBox, PlusEquals1) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({-1, 0.25, 0.25}, {0.5, 0.5, 0.5});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{-1, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{1, 1, 1});
}

TEST(AxisAlignedBoundingBox, PlusEquals2) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({0.25, -1, 0.25}, {0.5, 0.5, 0.5});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{0, -1, 0});
    ExpectEQ(box.max_bound_, Vector3d{1, 1, 1});
}

TEST(AxisAlignedBoundingBox, PlusEquals3) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({0.25, 0.25, -1}, {0.5, 0.5, 0.5});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{0, 0, -1});
    ExpectEQ(box.max_bound_, Vector3d{1, 1, 1});
}

TEST(AxisAlignedBoundingBox, PlusEquals4) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({0.25, 0.25, 0.25}, {2, 0.5, 0.5});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{2, 1, 1});
}

TEST(AxisAlignedBoundingBox, PlusEquals5) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({0.25, 0.25, 0.25}, {0.5, 2, 0.5});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{1, 2, 1});
}

TEST(AxisAlignedBoundingBox, PlusEquals6) {
    AxisAlignedBoundingBox box({0, 0, 0}, {1, 1, 1});
    AxisAlignedBoundingBox box1({0.25, 0.25, 0.25}, {0.5, 0.5, 2});
    box += box1;
    ExpectEQ(box.min_bound_, Vector3d{0, 0, 0});
    ExpectEQ(box.max_bound_, Vector3d{1, 1, 2});
}

// IsEmpty permutations
class IsEmptyTests : public TestWithParam<emp_t> {};

TEST_P(IsEmptyTests, CheckBoxEmpty) {
    AxisAlignedBoundingBox box(std::get<0>(GetParam()),
                               std::get<1>(GetParam()));
    EXPECT_EQ(std::get<2>(GetParam()), box.IsEmpty());
}

INSTANTIATE_TEST_SUITE_P(AABBTests,
                         IsEmptyTests,
                         Values(emp_t{{4, 2, 3}, {4, 5, 6}, true},
                                emp_t{{1, 5, 3}, {4, 5, 6}, true},
                                emp_t{{1, 2, 6}, {4, 5, 6}, true},
                                emp_t{{1, 2, 3}, {4, 5, 6}, false}));

// Get Max Extent
class MaxExtentTests : public TestWithParam<dbl_t> {};

TEST_P(MaxExtentTests, CheckMaxExtent) {
    AxisAlignedBoundingBox box(std::get<0>(GetParam()),
                               std::get<1>(GetParam()));
    EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), box.GetMaxExtent());
}

INSTANTIATE_TEST_SUITE_P(AABBTests,
                         MaxExtentTests,
                         Values(dbl_t{{-1, -2, -3}, {1, 2, 10}, 13},
                                dbl_t{{-1, -2, -3}, {10, 2, 3}, 11},
                                dbl_t{{-1, -2, -3}, {1, 10, 3}, 12}));

// Volume
class VolumeTests : public TestWithParam<dbl_t> {};

TEST_P(VolumeTests, CheckMaxExtent) {
    AxisAlignedBoundingBox box(std::get<0>(GetParam()),
                               std::get<1>(GetParam()));
    EXPECT_DOUBLE_EQ(std::get<2>(GetParam()), box.Volume());
}

INSTANTIATE_TEST_SUITE_P(AABBTests,
                         VolumeTests,
                         Values(dbl_t{{-1, -2, -3}, {1, 2, 3}, 2 * 4 * 6},
                                dbl_t{{0, 0, 0}, {1, 1, 1}, 1},
                                dbl_t{{0, 0, 0}, {3, 1, 1}, 3},
                                dbl_t{{0, 0, 0}, {1, 4, 1}, 4},
                                dbl_t{{0, 0, 0}, {1, 1, 5}, 5}));

}  // namespace tests
}  // namespace open3d
