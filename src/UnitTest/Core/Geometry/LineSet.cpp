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

using namespace std;

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
TEST(LineSet, Clear)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector2i imin(0, 0);
    Eigen::Vector2i imax(1000, 1000);

    open3d::LineSet ls;

    ls.points_.resize(size);
    ls.lines_.resize(size);
    ls.colors_.resize(size);

    UnitTest::Rand(ls.points_, dmin, dmax, 0);
    UnitTest::Rand(ls.lines_, imin, imax, 0);
    UnitTest::Rand(ls.colors_, dmin, dmax, 0);

    Eigen::Vector3d minBound = ls.GetMinBound();
    Eigen::Vector3d maxBound = ls.GetMaxBound();

    EXPECT_FALSE(ls.IsEmpty());
    EXPECT_NEAR( 19.607843, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  0.000000, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  0.000000, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(996.078431, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(996.078431, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(996.078431, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_TRUE(ls.HasPoints());
    EXPECT_TRUE(ls.HasLines());
    EXPECT_TRUE(ls.HasColors());

    ls.Clear();

    minBound = ls.GetMinBound();
    maxBound = ls.GetMaxBound();

    // public members
    EXPECT_TRUE(ls.IsEmpty());
    EXPECT_NEAR(0.0, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_FALSE(ls.HasPoints());
    EXPECT_FALSE(ls.HasLines());
    EXPECT_FALSE(ls.HasColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(LineSet, IsEmpty)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::LineSet ls;

    EXPECT_TRUE(ls.IsEmpty());

    ls.points_.resize(size);

    UnitTest::Rand(ls.points_, vmin, vmax, 0);

    EXPECT_FALSE(ls.IsEmpty());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(LineSet, GetMinBound)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::LineSet ls;

    ls.points_.resize(size);

    UnitTest::Rand(ls.points_, vmin, vmax, 0);

    Eigen::Vector3d minBound = ls.GetMinBound();

    EXPECT_NEAR( 19.607843, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  0.000000, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  0.000000, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(LineSet, GetMaxBound)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::LineSet ls;

    ls.points_.resize(size);

    UnitTest::Rand(ls.points_, vmin, vmax, 0);

    Eigen::Vector3d maxBound = ls.GetMaxBound();

    EXPECT_NEAR(996.078431, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(996.078431, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(996.078431, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(LineSet, Transform)
{
    vector<Eigen::Vector3d> ref_points =
    {
        {  396.870588, 1201.976471,  880.472941 },\
        {  320.792157, 1081.976471,  829.139608 },\
        {  269.027451,  818.447059,  406.786667 },\
        {  338.831373, 1001.192157,  614.237647 },\
        {  423.537255, 1153.349020,  483.727843 },\
        {  432.949020, 1338.447059,  964.512157 },\
        {  140.007843,  444.721569,  189.296471 },\
        {  292.164706,  763.152941,  317.178824 },\
        {  134.517647,  407.858824,  192.002353 },\
        {  274.909804,  802.368627,  218.747451 } \
    };

    vector<Eigen::Vector2i> ref_lines =
    {
        {   839,   392 },\
        {   780,   796 },\
        {   909,   196 },\
        {   333,   764 },\
        {   274,   552 },\
        {   474,   627 },\
        {   364,   509 },\
        {   949,   913 },\
        {   635,   713 },\
        {   141,   603 } \
    };

    int size = 10;
    open3d::LineSet ls;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector2i imin(0, 0);
    Eigen::Vector2i imax(1000, 1000);

    ls.points_.resize(size);
    UnitTest::Rand(ls.points_, dmin, dmax, 0);

    ls.lines_.resize(size);
    UnitTest::Rand(ls.lines_, imin, imax, 0);

    Eigen::Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40,
                      0.50, 0.60, 0.70, 0.80,
                      0.90, 0.10, 0.11, 0.12,
                      0.13, 0.14, 0.15, 0.16;

    ls.Transform(transformation);

    for (size_t i = 0; i < ls.points_.size(); i++)
    {
        EXPECT_NEAR(ref_points[i](0, 0), ls.points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](1, 0), ls.points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](2, 0), ls.points_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_EQ(ref_lines[i](0, 0), ls.lines_[i](0, 0));
        EXPECT_EQ(ref_lines[i](1, 0), ls.lines_[i](1, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(LineSet, HasPoints)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::LineSet ls;

    EXPECT_FALSE(ls.HasPoints());

    ls.points_.resize(size);

    UnitTest::Rand(ls.points_, vmin, vmax, 0);

    EXPECT_TRUE(ls.HasPoints());
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
