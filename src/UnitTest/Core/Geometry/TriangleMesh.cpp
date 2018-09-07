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

#include "Core/Geometry/TriangleMesh.h"

using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Constructor)
{
    open3d::TriangleMesh tm;

    // inherited from Geometry2D
    EXPECT_EQ(open3d::Geometry::GeometryType::TriangleMesh, tm.GetGeometryType());
    EXPECT_EQ(3, tm.Dimension());

    // public member variables
    EXPECT_EQ(0, tm.vertices_.size());
    EXPECT_EQ(0, tm.vertex_normals_.size());
    EXPECT_EQ(0, tm.vertex_colors_.size());
    EXPECT_EQ(0, tm.triangles_.size());
    EXPECT_EQ(0, tm.triangle_normals_.size());

    // public members
    EXPECT_TRUE(tm.IsEmpty());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), tm.GetMinBound());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), tm.GetMaxBound());
    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, DISABLED_Destructor)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, DISABLED_MemberData)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Clear)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    UnitTest::Rand(tm.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm.vertex_colors_,    dmin, dmax, 0);
    UnitTest::Rand(tm.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm.triangle_normals_, dmin, dmax, 0);

    Eigen::Vector3d minBound = tm.GetMinBound();
    Eigen::Vector3d maxBound = tm.GetMaxBound();

    EXPECT_FALSE(tm.IsEmpty());
    EXPECT_NEAR( 20.023049, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  3.231460, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  3.578574, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(997.798999, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(998.924518, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(999.993571, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_TRUE(tm.HasVertices());
    EXPECT_TRUE(tm.HasVertexNormals());
    EXPECT_TRUE(tm.HasVertexColors());
    EXPECT_TRUE(tm.HasTriangles());
    EXPECT_TRUE(tm.HasTriangleNormals());

    tm.Clear();

    minBound = tm.GetMinBound();
    maxBound = tm.GetMaxBound();

    // public members
    EXPECT_TRUE(tm.IsEmpty());
    EXPECT_NEAR(0.0, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(0.0, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, IsEmpty)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_TRUE(tm.IsEmpty());

    tm.vertices_.resize(size);

    EXPECT_FALSE(tm.IsEmpty());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, GetMinBound)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    UnitTest::Rand(tm.vertices_, dmin, dmax, 0);

    Eigen::Vector3d minBound = tm.GetMinBound();

    EXPECT_NEAR( 20.023049, minBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  3.231460, minBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  3.578574, minBound(2, 0), UnitTest::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, GetMaxBound)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    UnitTest::Rand(tm.vertices_, dmin, dmax, 0);

    Eigen::Vector3d maxBound = tm.GetMaxBound();

    EXPECT_NEAR(997.798999, maxBound(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(998.924518, maxBound(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(999.993571, maxBound(2, 0), UnitTest::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Transform)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        { 398.225124,1205.693071, 881.868153 },\
        { 321.838886,1085.294390, 831.611417 },\
        { 270.900608, 823.791432, 409.198658 },\
        { 339.937683,1004.432856, 615.608467 },\
        { 425.227547,1157.793590, 484.511386 },\
        { 434.350931,1342.432421, 967.169396 },\
        { 140.844202, 447.193004, 190.052250 },\
        { 293.388019, 767.506059, 320.900694 },\
        { 135.193922, 410.559494, 195.502569 },\
        { 276.542855, 807.338946, 221.948633 } \
    };

    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        { 397.825124,1204.893071, 881.748153 },\
        { 321.438886,1084.494390, 831.491417 },\
        { 270.500608, 822.991432, 409.078658 },\
        { 339.537683,1003.632856, 615.488467 },\
        { 424.827547,1156.993590, 484.391386 },\
        { 433.950931,1341.632421, 967.049396 },\
        { 140.444202, 446.393004, 189.932250 },\
        { 292.988019, 766.706059, 320.780694 },\
        { 134.793922, 409.759494, 195.382569 },\
        { 276.142855, 806.538946, 221.828633 } \
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        { 397.825124,1204.893071, 881.748153 },\
        { 321.438886,1084.494390, 831.491417 },\
        { 270.500608, 822.991432, 409.078658 },\
        { 339.537683,1003.632856, 615.488467 },\
        { 424.827547,1156.993590, 484.391386 },\
        { 433.950931,1341.632421, 967.049396 },\
        { 140.444202, 446.393004, 189.932250 },\
        { 292.988019, 766.706059, 320.780694 },\
        { 134.793922, 409.759494, 195.382569 },\
        { 276.142855, 806.538946, 221.828633 } \
    };

    int size = 10;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);

    UnitTest::Rand(tm.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm.triangle_normals_, dmin, dmax, 0);

    Eigen::Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40,
                      0.50, 0.60, 0.70, 0.80,
                      0.90, 0.10, 0.11, 0.12,
                      0.13, 0.14, 0.15, 0.16;

    tm.Transform(transformation);

    EXPECT_EQ(ref_vertices.size(), tm.vertices_.size());
    EXPECT_EQ(ref_vertex_normals.size(), tm.vertex_normals_.size());
    EXPECT_EQ(ref_triangle_normals.size(), tm.triangle_normals_.size());
    for (size_t i = 0; i < tm.vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), tm.vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), tm.vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), tm.vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_vertex_normals[i](0, 0), tm.vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), tm.vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), tm.vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_triangle_normals[i](0, 0), tm.triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), tm.triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), tm.triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, OperatorAppend)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm0;
    open3d::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    UnitTest::Rand(tm0.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_colors_,    dmin, dmax, 0);
    UnitTest::Rand(tm0.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm0.triangle_normals_, dmin, dmax, 0);

    UnitTest::Rand(tm1.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_colors_,    dmin, dmax, 1);
    UnitTest::Rand(tm1.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm1.triangle_normals_, dmin, dmax, 0);

    open3d::TriangleMesh tm(tm0);
    tm += tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertices_[i](0, 0), tm.vertices_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertices_[i](1, 0), tm.vertices_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertices_[i](2, 0), tm.vertices_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertices_[i](0, 0), tm.vertices_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertices_[i](1, 0), tm.vertices_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertices_[i](2, 0), tm.vertices_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertex_normals_[i](0, 0), tm.vertex_normals_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_normals_[i](1, 0), tm.vertex_normals_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_normals_[i](2, 0), tm.vertex_normals_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertex_normals_[i](0, 0), tm.vertex_normals_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_normals_[i](1, 0), tm.vertex_normals_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_normals_[i](2, 0), tm.vertex_normals_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertex_colors_[i](0, 0), tm.vertex_colors_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_colors_[i](1, 0), tm.vertex_colors_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_colors_[i](2, 0), tm.vertex_colors_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertex_colors_[i](0, 0), tm.vertex_colors_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_colors_[i](1, 0), tm.vertex_colors_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_colors_[i](2, 0), tm.vertex_colors_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.triangles_[i](0, 0) +    0, tm.triangles_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangles_[i](1, 0) +    0, tm.triangles_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangles_[i](2, 0) +    0, tm.triangles_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.triangles_[i](0, 0) + size, tm.triangles_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangles_[i](1, 0) + size, tm.triangles_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangles_[i](2, 0) + size, tm.triangles_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.triangle_normals_[i](0, 0), tm.triangle_normals_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangle_normals_[i](1, 0), tm.triangle_normals_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangle_normals_[i](2, 0), tm.triangle_normals_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.triangle_normals_[i](0, 0), tm.triangle_normals_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangle_normals_[i](1, 0), tm.triangle_normals_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangle_normals_[i](2, 0), tm.triangle_normals_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, OperatorADD)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm0;
    open3d::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    UnitTest::Rand(tm0.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_colors_,    dmin, dmax, 0);
    UnitTest::Rand(tm0.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm0.triangle_normals_, dmin, dmax, 0);

    UnitTest::Rand(tm1.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_colors_,    dmin, dmax, 1);
    UnitTest::Rand(tm1.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm1.triangle_normals_, dmin, dmax, 0);

    open3d::TriangleMesh tm = tm0 + tm1;

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertices_[i](0, 0), tm.vertices_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertices_[i](1, 0), tm.vertices_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertices_[i](2, 0), tm.vertices_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertices_[i](0, 0), tm.vertices_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertices_[i](1, 0), tm.vertices_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertices_[i](2, 0), tm.vertices_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertex_normals_[i](0, 0), tm.vertex_normals_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_normals_[i](1, 0), tm.vertex_normals_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_normals_[i](2, 0), tm.vertex_normals_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertex_normals_[i](0, 0), tm.vertex_normals_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_normals_[i](1, 0), tm.vertex_normals_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_normals_[i](2, 0), tm.vertex_normals_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.vertex_colors_[i](0, 0), tm.vertex_colors_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_colors_[i](1, 0), tm.vertex_colors_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.vertex_colors_[i](2, 0), tm.vertex_colors_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.vertex_colors_[i](0, 0), tm.vertex_colors_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_colors_[i](1, 0), tm.vertex_colors_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.vertex_colors_[i](2, 0), tm.vertex_colors_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.triangles_[i](0, 0) +    0, tm.triangles_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangles_[i](1, 0) +    0, tm.triangles_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangles_[i](2, 0) +    0, tm.triangles_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.triangles_[i](0, 0) + size, tm.triangles_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangles_[i](1, 0) + size, tm.triangles_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangles_[i](2, 0) + size, tm.triangles_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++)
    {
        EXPECT_NEAR(tm0.triangle_normals_[i](0, 0), tm.triangle_normals_[   0 + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangle_normals_[i](1, 0), tm.triangle_normals_[   0 + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm0.triangle_normals_[i](2, 0), tm.triangle_normals_[   0 + i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(tm1.triangle_normals_[i](0, 0), tm.triangle_normals_[size + i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangle_normals_[i](1, 0), tm.triangle_normals_[size + i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(tm1.triangle_normals_[i](2, 0), tm.triangle_normals_[size + i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, ComputeTriangleNormals)
{
    vector<Eigen::Vector3d> ref =
    {
        {  -0.626953,  -0.774222,   0.086657 },\
        {  -0.318138,  -0.930480,   0.181647 },\
        {   0.112014,   0.485734,   0.866900 },\
        {   0.402759,   0.818194,  -0.410297 },\
        {   0.864002,  -0.503048,  -0.021061 },\
        {   0.184755,  -0.444282,   0.876629 },\
        {   0.261678,  -0.559228,  -0.786631 },\
        {   0.447148,  -0.802989,  -0.394040 },\
        {   0.000000,   0.000000,   1.000000 },\
        {  -0.412314,  -0.865640,   0.284014 },\
        {   0.619907,  -0.741403,   0.256977 },\
        {  -0.967754,   0.187049,  -0.168715 },\
        {  -0.796156,  -0.228800,  -0.560166 },\
        {   0.000000,   0.000000,   1.000000 },\
        {   0.660489,   0.386835,  -0.643516 },\
        {   0.534962,   0.441787,  -0.720166 },\
        {  -0.346221,  -0.574765,  -0.741469 },\
        {  -0.998830,  -0.038534,   0.029220 },\
        {   0.406639,  -0.402150,  -0.820317 },\
        {  -0.585888,  -0.627625,  -0.512662 },\
        {  -0.938966,   0.000587,   0.344011 },\
        {   0.597906,  -0.762337,   0.247691 },\
        {  -0.198742,  -0.445803,   0.872789 },\
        {  -0.306144,  -0.715797,  -0.627622 },\
        {   0.000000,   0.000000,   1.000000 } \
    };

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);
    UnitTest::Rand(tm.vertices_, dmin, dmax, 0);
    UnitTest::Rand(tm.triangles_, imin, imax, 1);

    tm.ComputeTriangleNormals();

    EXPECT_EQ(ref.size(), tm.triangle_normals_.size());
    for (size_t i = 0; i < tm.triangle_normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), tm.triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), tm.triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), tm.triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, ComputeVertexNormals)
{
    vector<Eigen::Vector3d> ref =
    {
        {  -0.328666,  -0.691845,  -0.642907 },\
        {   0.015647,  -0.566687,  -0.823785 },\
        {  -0.726138,  -0.651171,   0.220680 },\
        {   0.383423,  -0.725978,  -0.570914 },\
        {  -0.545622,  -0.821390,   0.166180 },\
        {  -0.524571,  -0.830035,   0.189387 },\
        {   0.995896,  -0.077948,  -0.046001 },\
        {  -0.935695,   0.008680,  -0.352704 },\
        {   0.741147,  -0.312961,  -0.593933 },\
        {  -0.405892,  -0.912841,   0.044408 },\
        {  -0.585888,  -0.627625,  -0.512662 },\
        {  -0.535388,   0.273789,  -0.798999 },\
        {  -0.254314,  -0.964256,  -0.074401 },\
        {   0.402759,   0.818194,  -0.410297 },\
        {   0.469877,  -0.745575,  -0.472581 },\
        {  -0.836572,   0.042436,   0.546211 },\
        {  -0.306144,  -0.715797,  -0.627622 },\
        {   0.184755,  -0.444282,   0.876629 },\
        {  -0.548120,  -0.668317,   0.502909 },\
        {  -0.090128,  -0.995192,  -0.038349 },\
        {  -0.188512,  -0.963922,   0.187932 },\
        {   0.181986,  -0.980940,  -0.068096 },\
        {   0.697897,  -0.451192,  -0.556207 },\
        {  -0.869681,  -0.493283,  -0.018062 },\
        {   0.000000,   0.000000,   1.000000 } \
    };

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);
    UnitTest::Rand(tm.vertices_, dmin, dmax, 0);
    UnitTest::Rand(tm.triangles_, imin, imax, 1);

    tm.ComputeVertexNormals();

    EXPECT_EQ(ref.size(), tm.vertex_normals_.size());
    for (size_t i = 0; i < tm.vertex_normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), tm.vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), tm.vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), tm.vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, Purge)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        { 840.187717, 394.382927, 783.099224 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 335.222756, 768.229595, 277.774711 },\
        { 553.969956, 477.397052, 628.870925 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 916.195068, 635.711728, 717.296929 },\
        { 141.602555, 606.968876,  16.300572 },\
        { 242.886771, 137.231577, 804.176754 },\
        { 156.679089, 400.944394, 129.790447 },\
        { 108.808802, 998.924518, 218.256905 },\
        { 512.932394, 839.112235, 612.639833 },\
        { 296.031618, 637.552268, 524.287190 },\
        { 493.582987, 972.775024, 292.516784 },\
        { 771.357698, 526.744979, 769.913836 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 352.458347, 807.724520, 919.026474 },\
        {  69.755276, 949.327075, 525.995350 },\
        {  86.055848, 192.213846, 663.226927 },\
        { 890.232603, 348.892935,  64.171321 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 238.279954, 970.634132, 902.208073 },\
        { 850.919787, 266.665749, 539.760341 },\
        { 375.206976, 760.248736, 512.535364 },\
        { 667.723761, 531.606434,  39.280343 } \
    };

    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        { 840.187717, 394.382927, 783.099224 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 335.222756, 768.229595, 277.774711 },\
        { 553.969956, 477.397052, 628.870925 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 916.195068, 635.711728, 717.296929 },\
        { 141.602555, 606.968876,  16.300572 },\
        { 242.886771, 137.231577, 804.176754 },\
        { 156.679089, 400.944394, 129.790447 },\
        { 108.808802, 998.924518, 218.256905 },\
        { 512.932394, 839.112235, 612.639833 },\
        { 296.031618, 637.552268, 524.287190 },\
        { 493.582987, 972.775024, 292.516784 },\
        { 771.357698, 526.744979, 769.913836 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 352.458347, 807.724520, 919.026474 },\
        {  69.755276, 949.327075, 525.995350 },\
        {  86.055848, 192.213846, 663.226927 },\
        { 890.232603, 348.892935,  64.171321 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 238.279954, 970.634132, 902.208073 },\
        { 850.919787, 266.665749, 539.760341 },\
        { 375.206976, 760.248736, 512.535364 },\
        { 667.723761, 531.606434,  39.280343 } \
    };

    vector<Eigen::Vector3d> ref_vertex_colors =
    {
        { 840.187717, 394.382927, 783.099224 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 335.222756, 768.229595, 277.774711 },\
        { 553.969956, 477.397052, 628.870925 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 916.195068, 635.711728, 717.296929 },\
        { 141.602555, 606.968876,  16.300572 },\
        { 242.886771, 137.231577, 804.176754 },\
        { 156.679089, 400.944394, 129.790447 },\
        { 108.808802, 998.924518, 218.256905 },\
        { 512.932394, 839.112235, 612.639833 },\
        { 296.031618, 637.552268, 524.287190 },\
        { 493.582987, 972.775024, 292.516784 },\
        { 771.357698, 526.744979, 769.913836 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 352.458347, 807.724520, 919.026474 },\
        {  69.755276, 949.327075, 525.995350 },\
        {  86.055848, 192.213846, 663.226927 },\
        { 890.232603, 348.892935,  64.171321 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 238.279954, 970.634132, 902.208073 },\
        { 850.919787, 266.665749, 539.760341 },\
        { 375.206976, 760.248736, 512.535364 },\
        { 667.723761, 531.606434,  39.280343 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {    20,     9,    18 },\
        {    19,    21,     4 },\
        {     8,    18,     6 },\
        {    13,    11,    15 },\
        {     8,    12,    22 },\
        {    21,    15,    17 },\
        {     3,    14,     0 },\
        {     5,     3,    19 },\
        {     2,    23,     5 },\
        {    12,    20,    14 },\
        {     7,    15,    12 },\
        {    11,    23,     7 },\
        {     9,    21,     6 },\
        {     8,    19,    22 },\
        {     1,    22,    12 },\
        {     2,     4,    15 },\
        {    21,     8,     1 },\
        {     0,    10,     1 },\
        {     5,    23,    21 },\
        {    20,     6,    12 },\
        {     9,    18,    12 },\
        {    16,    12,     0 } \
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        { 840.187717, 394.382927, 783.099224 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 335.222756, 768.229595, 277.774711 },\
        { 553.969956, 477.397052, 628.870925 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 916.195068, 635.711728, 717.296929 },\
        { 141.602555, 606.968876,  16.300572 },\
        { 242.886771, 137.231577, 804.176754 },\
        { 108.808802, 998.924518, 218.256905 },\
        { 512.932394, 839.112235, 612.639833 },\
        { 296.031618, 637.552268, 524.287190 },\
        { 493.582987, 972.775024, 292.516784 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 352.458347, 807.724520, 919.026474 },\
        {  69.755276, 949.327075, 525.995350 },\
        {  86.055848, 192.213846, 663.226927 },\
        { 890.232603, 348.892935,  64.171321 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 238.279954, 970.634132, 902.208073 },\
        { 850.919787, 266.665749, 539.760341 },\
        { 375.206976, 760.248736, 512.535364 },\
        { 667.723761, 531.606434,  39.280343 } \
    };

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm0;
    open3d::TriangleMesh tm1;

    tm0.vertices_.resize(size);
    tm0.vertex_normals_.resize(size);
    tm0.vertex_colors_.resize(size);
    tm0.triangles_.resize(size);
    tm0.triangle_normals_.resize(size);

    tm1.vertices_.resize(size);
    tm1.vertex_normals_.resize(size);
    tm1.vertex_colors_.resize(size);
    tm1.triangles_.resize(size);
    tm1.triangle_normals_.resize(size);

    UnitTest::Rand(tm0.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm0.vertex_colors_,    dmin, dmax, 0);
    UnitTest::Rand(tm0.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm0.triangle_normals_, dmin, dmax, 0);

    UnitTest::Rand(tm1.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_normals_,   dmin, dmax, 0);
    UnitTest::Rand(tm1.vertex_colors_,    dmin, dmax, 1);
    UnitTest::Rand(tm1.triangles_,        imin, imax, 0);
    UnitTest::Rand(tm1.triangle_normals_, dmin, dmax, 0);

    open3d::TriangleMesh tm = tm0 + tm1;

    tm.Purge();

    EXPECT_EQ(ref_vertices.size(), tm.vertices_.size());
    for (size_t i = 0; i < tm.vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), tm.vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), tm.vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), tm.vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_normals.size(), tm.vertex_normals_.size());
    for (size_t i = 0; i < tm.vertex_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_normals[i](0, 0), tm.vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), tm.vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), tm.vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_colors.size(), tm.vertex_colors_.size());
    for (size_t i = 0; i < tm.vertex_colors_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_colors[i](0, 0), tm.vertex_colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](1, 0), tm.vertex_colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](2, 0), tm.vertex_colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), tm.triangles_.size());
    for (size_t i = 0; i < tm.triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), tm.triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), tm.triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), tm.triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangle_normals.size(), tm.triangle_normals_.size());
    for (size_t i = 0; i < tm.triangle_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_triangle_normals[i](0, 0), tm.triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), tm.triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), tm.triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertices)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    tm.vertices_.resize(size);

    EXPECT_TRUE(tm.HasVertices());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasTriangles)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangles());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);

    EXPECT_TRUE(tm.HasTriangles());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertexNormals)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexNormals());

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);

    EXPECT_TRUE(tm.HasVertexNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasVertexColors)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexColors());

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    EXPECT_TRUE(tm.HasVertexColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, HasTriangleNormals)
{
    int size = 100;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    open3d::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangleNormals());

    tm.vertices_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    EXPECT_TRUE(tm.HasTriangleNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, NormalizeNormals)
{
    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        {   0.691871,   0.324763,   0.644860 },\
        {   0.650271,   0.742470,   0.160891 },\
        {   0.379636,   0.870011,   0.314576 },\
        {   0.574357,   0.494966,   0.652014 },\
        {   0.319521,   0.449696,   0.834074 },\
        {   0.690989,   0.479450,   0.540981 },\
        {   0.227116,   0.973517,   0.026144 },\
        {   0.285349,   0.161223,   0.944766 },\
        {   0.348477,   0.891758,   0.288673 },\
        {   0.105818,   0.971468,   0.212258 },\
        {   0.442687,   0.724197,   0.528740 },\
        {   0.337582,   0.727038,   0.597875 },\
        {   0.437042,   0.861341,   0.259008 },\
        {   0.637242,   0.435160,   0.636049 },\
        {   0.393351,   0.876209,   0.278446 },\
        {   0.276810,   0.634363,   0.721776 },\
        {   0.064140,   0.872907,   0.483653 },\
        {   0.123668,   0.276225,   0.953103 },\
        {   0.928961,   0.364071,   0.066963 },\
        {   0.043296,   0.989703,   0.136434 },\
        {   0.176971,   0.720892,   0.670072 },\
        {   0.816339,   0.255829,   0.517825 },\
        {   0.378736,   0.767399,   0.517356 },\
        {   0.781510,   0.622197,   0.045974 },\
        {   0.315325,   0.671402,   0.670663 } \
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        {   0.691871,   0.324763,   0.644860 },\
        {   0.650271,   0.742470,   0.160891 },\
        {   0.379636,   0.870011,   0.314576 },\
        {   0.574357,   0.494966,   0.652014 },\
        {   0.319521,   0.449696,   0.834074 },\
        {   0.690989,   0.479450,   0.540981 },\
        {   0.227116,   0.973517,   0.026144 },\
        {   0.285349,   0.161223,   0.944766 },\
        {   0.348477,   0.891758,   0.288673 },\
        {   0.105818,   0.971468,   0.212258 },\
        {   0.442687,   0.724197,   0.528740 },\
        {   0.337582,   0.727038,   0.597875 },\
        {   0.437042,   0.861341,   0.259008 },\
        {   0.637242,   0.435160,   0.636049 },\
        {   0.393351,   0.876209,   0.278446 },\
        {   0.276810,   0.634363,   0.721776 },\
        {   0.064140,   0.872907,   0.483653 },\
        {   0.123668,   0.276225,   0.953103 },\
        {   0.928961,   0.364071,   0.066963 },\
        {   0.043296,   0.989703,   0.136434 },\
        {   0.176971,   0.720892,   0.670072 },\
        {   0.816339,   0.255829,   0.517825 },\
        {   0.378736,   0.767399,   0.517356 },\
        {   0.781510,   0.622197,   0.045974 },\
        {   0.315325,   0.671402,   0.670663 } \
    };

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    open3d::TriangleMesh tm;

    tm.vertex_normals_.resize(size);
    tm.triangle_normals_.resize(size);
    UnitTest::Rand(tm.vertex_normals_, dmin, dmax, 0);
    UnitTest::Rand(tm.triangle_normals_, dmin, dmax, 1);

    tm.NormalizeNormals();

    EXPECT_EQ(ref_vertex_normals.size(), tm.vertex_normals_.size());
    for (size_t i = 0; i < tm.vertex_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_normals[i](0, 0), tm.vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), tm.vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), tm.vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangle_normals.size(), tm.triangle_normals_.size());
    for (size_t i = 0; i < tm.triangle_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_triangle_normals[i](0, 0), tm.triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), tm.triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), tm.triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, PaintUniformColor)
{
    vector<Eigen::Vector3d> ref =
    {
        {   0.691871,   0.324763,   0.644860 },\
        {   0.650271,   0.742470,   0.160891 },\
        {   0.379636,   0.870011,   0.314576 },\
        {   0.574357,   0.494966,   0.652014 },\
        {   0.319521,   0.449696,   0.834074 },\
        {   0.690989,   0.479450,   0.540981 },\
        {   0.227116,   0.973517,   0.026144 },\
        {   0.285349,   0.161223,   0.944766 },\
        {   0.348477,   0.891758,   0.288673 },\
        {   0.105818,   0.971468,   0.212258 },\
        {   0.442687,   0.724197,   0.528740 },\
        {   0.337582,   0.727038,   0.597875 },\
        {   0.437042,   0.861341,   0.259008 },\
        {   0.637242,   0.435160,   0.636049 },\
        {   0.393351,   0.876209,   0.278446 },\
        {   0.276810,   0.634363,   0.721776 },\
        {   0.064140,   0.872907,   0.483653 },\
        {   0.123668,   0.276225,   0.953103 },\
        {   0.928961,   0.364071,   0.066963 },\
        {   0.043296,   0.989703,   0.136434 },\
        {   0.176971,   0.720892,   0.670072 },\
        {   0.816339,   0.255829,   0.517825 },\
        {   0.378736,   0.767399,   0.517356 },\
        {   0.781510,   0.622197,   0.045974 },\
        {   0.315325,   0.671402,   0.670663 } \
    };

    int size = 25;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(10.0, 10.0, 10.0);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_colors_.resize(size);

    tm.PaintUniformColor(Eigen::Vector3d(31, 120, 205));

    for (size_t i = 0; i < tm.vertex_colors_.size(); i++)
    {
        EXPECT_NEAR( 31, tm.vertex_colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(120, tm.vertex_colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(205, tm.vertex_colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, SelectDownSample)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        { 881.062170, 641.080596, 431.953418 },\
        {  51.938819, 157.807129, 999.993571 },\
        { 992.228461, 576.971113, 877.613778 },\
        { 189.750982, 984.363203,   3.578574 },\
        {  93.919511,  99.559296, 382.895933 },\
        { 305.238817, 261.570322, 655.368107 },\
        { 313.229552, 885.014209, 186.264780 },\
        { 775.421465, 794.910152, 262.784575 },\
        { 210.691561, 606.542169, 865.433501 },\
        { 747.637759, 475.805631, 272.987121 },\
        { 293.177373, 244.068722, 912.390542 },\
        { 646.711828, 341.482374,  50.167885 },\
        { 645.133965, 443.338859, 269.022361 },\
        { 892.323803, 711.905978, 405.267347 },\
        { 101.123717, 584.997879,  60.458876 },\
        { 703.570679, 220.678545,   6.425603 },\
        { 532.218332, 993.527716,  59.163315 },\
        { 500.018598, 467.274167, 251.974033 },\
        { 104.330935, 118.126411, 847.952913 },\
        {  89.609007, 687.545505,  48.738517 },\
        { 351.819269, 554.097076, 330.436215 },\
        { 959.096226, 712.522314, 941.310920 },\
        { 110.427054, 958.022309, 253.276042 },\
        { 758.363768, 424.719606,  62.946831 },\
        { 889.153858, 397.747194, 771.380796 },\
        { 710.806943, 563.262821, 559.248871 },\
        { 491.213758, 200.167924, 765.058087 },\
        { 834.149158, 520.481835, 960.868455 },\
        { 277.892341,  34.380313, 355.916394 },\
        {  59.799811, 923.363849, 813.176790 },\
        { 207.670107, 781.218962, 789.879456 },\
        { 899.686776, 725.663023, 771.367079 } \
    };

    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        { 881.062170, 641.080596, 431.953418 },\
        {  51.938819, 157.807129, 999.993571 },\
        { 992.228461, 576.971113, 877.613778 },\
        { 189.750982, 984.363203,   3.578574 },\
        {  93.919511,  99.559296, 382.895933 },\
        { 305.238817, 261.570322, 655.368107 },\
        { 313.229552, 885.014209, 186.264780 },\
        { 775.421465, 794.910152, 262.784575 },\
        { 210.691561, 606.542169, 865.433501 },\
        { 747.637759, 475.805631, 272.987121 },\
        { 293.177373, 244.068722, 912.390542 },\
        { 646.711828, 341.482374,  50.167885 },\
        { 645.133965, 443.338859, 269.022361 },\
        { 892.323803, 711.905978, 405.267347 },\
        { 101.123717, 584.997879,  60.458876 },\
        { 703.570679, 220.678545,   6.425603 },\
        { 532.218332, 993.527716,  59.163315 },\
        { 500.018598, 467.274167, 251.974033 },\
        { 104.330935, 118.126411, 847.952913 },\
        {  89.609007, 687.545505,  48.738517 },\
        { 351.819269, 554.097076, 330.436215 },\
        { 959.096226, 712.522314, 941.310920 },\
        { 110.427054, 958.022309, 253.276042 },\
        { 758.363768, 424.719606,  62.946831 },\
        { 889.153858, 397.747194, 771.380796 },\
        { 710.806943, 563.262821, 559.248871 },\
        { 491.213758, 200.167924, 765.058087 },\
        { 834.149158, 520.481835, 960.868455 },\
        { 277.892341,  34.380313, 355.916394 },\
        {  59.799811, 923.363849, 813.176790 },\
        { 207.670107, 781.218962, 789.879456 },\
        { 899.686776, 725.663023, 771.367079 } \
    };

    vector<Eigen::Vector3d> ref_vertex_colors =
    {
        { 872.353697, 921.574882, 364.222227 },\
        { 374.064842, 520.984642, 676.609647 },\
        { 739.419252, 964.715571, 399.697079 },\
        { 475.869509, 826.042327, 925.212533 },\
        { 254.644625,  18.654723, 789.405906 },\
        { 798.002539, 818.771859, 229.409614 },\
        { 324.791270, 940.873626, 297.895215 },\
        { 790.654675, 403.864791, 892.215921 },\
        { 374.420460, 752.591686, 685.094544 },\
        { 122.871707, 918.511637, 212.574964 },\
        { 338.911558, 675.326466, 835.483094 },\
        { 998.773116, 173.133770, 372.692220 },\
        { 454.208305, 911.550824, 179.233357 },\
        { 941.230209, 418.375448, 224.077750 },\
        { 537.218715, 432.870337, 297.949526 },\
        { 320.620772, 735.309440, 929.204521 },\
        { 925.781888,  76.066593, 814.017323 },\
        { 923.536112, 200.111795, 891.273699 },\
        { 998.739990, 560.172451, 130.136637 },\
        { 466.325659,  54.329684, 297.883514 },\
        { 475.107953, 870.326067, 857.747750 },\
        { 541.542130, 212.654451, 940.684030 },\
        { 663.014891, 629.614190, 344.331849 },\
        { 173.232250,   2.725684, 769.737481 },\
        { 588.182905,  30.288016, 963.936589 },\
        { 591.713167, 543.058723, 901.792241 },\
        { 860.858037, 196.976879, 389.410962 },\
        { 796.026002, 839.958775, 362.836064 },\
        { 601.512090, 600.243661, 537.947524 },\
        { 470.748191, 980.884444, 146.826171 },\
        { 851.082709, 839.729783, 248.716926 },\
        { 792.718875, 878.036742, 692.790219 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {    25,    24,     3 },\
        {    11,    19,     2 },\
        {    15,     1,    28 },\
        {     0,    13,     6 },\
        {    27,    18,    14 },\
        {    17,     8,    26 },\
        {    30,    31,     4 },\
        {    16,    10,    21 },\
        {    29,     5,     8 },\
        {    23,    20,    12 },\
        {     7,     9,    22 } \
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        { 864.974983, 198.221474, 794.434410 },\
        { 493.417258,  69.882431, 483.180169 },\
        { 845.394137, 764.803436, 142.041420 },\
        { 426.455686,   1.501350, 982.456206 },\
        { 606.857175, 915.631550, 556.517015 },\
        { 629.284069, 833.580199, 774.679618 },\
        { 994.696287, 189.595636, 308.608052 },\
        { 330.045136, 654.137845, 864.041435 },\
        { 284.233405, 662.860565,  80.323557 },\
        { 407.996546, 462.995717, 570.363653 },\
        { 765.373587, 439.733009, 275.567279 } \
    };

    int size = 1000;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    UnitTest::Rand(tm.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm.vertex_normals_,   dmin, dmax, 1);
    UnitTest::Rand(tm.vertex_colors_,    dmin, dmax, 2);
    UnitTest::Rand(tm.triangles_,        imin, imax, 3);
    UnitTest::Rand(tm.triangle_normals_, dmin, dmax, 4);

    vector<size_t> indices(size / 4);
    UnitTest::Rand<size_t>(indices, 0, size - 1, 0);

    auto output_tm = open3d::SelectDownSample(tm, indices);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_normals.size(), output_tm->vertex_normals_.size());
    for (size_t i = 0; i < output_tm->vertex_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_normals[i](0, 0), output_tm->vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), output_tm->vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), output_tm->vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_colors.size(), output_tm->vertex_colors_.size());
    for (size_t i = 0; i < output_tm->vertex_colors_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_colors[i](0, 0), output_tm->vertex_colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](1, 0), output_tm->vertex_colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](2, 0), output_tm->vertex_colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangle_normals.size(), output_tm->triangle_normals_.size());
    for (size_t i = 0; i < output_tm->triangle_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_triangle_normals[i](0, 0), output_tm->triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), output_tm->triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), output_tm->triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CropTriangleMesh)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        { 553.969956, 477.397052, 628.870925 },\
        { 402.378775, 560.784512, 554.448163 },\
        { 514.351694, 426.411559, 343.520037 },\
        { 726.846161, 305.987487, 645.644096 },\
        { 537.030208, 393.097261, 481.124130 },\
        { 603.411111, 351.089900, 485.460061 },\
        { 351.819269, 554.097076, 330.436215 },\
        { 780.282852, 348.881373, 770.682270 },\
        { 744.732158, 538.204681, 743.162931 } \
    };

    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        { 553.969956, 477.397052, 628.870925 },\
        { 402.378775, 560.784512, 554.448163 },\
        { 514.351694, 426.411559, 343.520037 },\
        { 726.846161, 305.987487, 645.644096 },\
        { 537.030208, 393.097261, 481.124130 },\
        { 603.411111, 351.089900, 485.460061 },\
        { 351.819269, 554.097076, 330.436215 },\
        { 780.282852, 348.881373, 770.682270 },\
        { 744.732158, 538.204681, 743.162931 } \
    };

    vector<Eigen::Vector3d> ref_vertex_colors =
    {
        { 642.966306, 990.603279, 295.718284 },\
        { 302.528661,  18.338040,  35.420785 },\
        { 697.147824, 373.670287, 818.731236 },\
        { 524.686117, 121.433780, 888.787534 },\
        { 844.571640, 556.005385,  25.646936 },\
        { 776.903922, 180.321763, 791.871200 },\
        { 475.107953, 870.326067, 857.747750 },\
        { 717.493304, 417.665837, 822.490274 },\
        { 963.686963, 495.552024, 483.926434 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     1,     3 },\
        {     2,     4,     7 },\
        {     8,     6,     5 } \
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        { 369.675181, 659.770840, 472.220823 },\
        { 293.251380, 649.662436, 139.229831 },\
        { 430.969716, 179.392677, 274.579157 } \
    };

    int size = 1000;

    Eigen::Vector3d dmin(0.0, 0.0, 0.0);
    Eigen::Vector3d dmax(1000.0, 1000.0, 1000.0);

    Eigen::Vector3i imin(0, 0, 0);
    Eigen::Vector3i imax(size - 1, size - 1, size - 1);

    open3d::TriangleMesh tm;

    tm.vertices_.resize(size);
    tm.vertex_normals_.resize(size);
    tm.vertex_colors_.resize(size);
    tm.triangles_.resize(size);
    tm.triangle_normals_.resize(size);

    UnitTest::Rand(tm.vertices_,         dmin, dmax, 0);
    UnitTest::Rand(tm.vertex_normals_,   dmin, dmax, 1);
    UnitTest::Rand(tm.vertex_colors_,    dmin, dmax, 2);
    UnitTest::Rand(tm.triangles_,        imin, imax, 3);
    UnitTest::Rand(tm.triangle_normals_, dmin, dmax, 4);

    Eigen::Vector3d cropBoundMin(300.0, 300.0, 300.0);
    Eigen::Vector3d cropBoundMax(800.0, 800.0, 800.0);

    auto output_tm = open3d::CropTriangleMesh(tm, cropBoundMin, cropBoundMax);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_normals.size(), output_tm->vertex_normals_.size());
    for (size_t i = 0; i < output_tm->vertex_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_normals[i](0, 0), output_tm->vertex_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), output_tm->vertex_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), output_tm->vertex_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_vertex_colors.size(), output_tm->vertex_colors_.size());
    for (size_t i = 0; i < output_tm->vertex_colors_.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_colors[i](0, 0), output_tm->vertex_colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](1, 0), output_tm->vertex_colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](2, 0), output_tm->vertex_colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangle_normals.size(), output_tm->triangle_normals_.size());
    for (size_t i = 0; i < output_tm->triangle_normals_.size(); i++)
    {
        EXPECT_NEAR(ref_triangle_normals[i](0, 0), output_tm->triangle_normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), output_tm->triangle_normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), output_tm->triangle_normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshSphere)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        {   0.000000,   0.000000,   1.000000 },\
        {   0.000000,   0.000000,  -1.000000 },\
        {   0.587785,   0.000000,   0.809017 },\
        {   0.475528,   0.345492,   0.809017 },\
        {   0.181636,   0.559017,   0.809017 },\
        {  -0.181636,   0.559017,   0.809017 },\
        {  -0.475528,   0.345492,   0.809017 },\
        {  -0.587785,   0.000000,   0.809017 },\
        {  -0.475528,  -0.345492,   0.809017 },\
        {  -0.181636,  -0.559017,   0.809017 },\
        {   0.181636,  -0.559017,   0.809017 },\
        {   0.475528,  -0.345492,   0.809017 },\
        {   0.951057,   0.000000,   0.309017 },\
        {   0.769421,   0.559017,   0.309017 },\
        {   0.293893,   0.904508,   0.309017 },\
        {  -0.293893,   0.904508,   0.309017 },\
        {  -0.769421,   0.559017,   0.309017 },\
        {  -0.951057,   0.000000,   0.309017 },\
        {  -0.769421,  -0.559017,   0.309017 },\
        {  -0.293893,  -0.904508,   0.309017 },\
        {   0.293893,  -0.904508,   0.309017 },\
        {   0.769421,  -0.559017,   0.309017 },\
        {   0.951057,   0.000000,  -0.309017 },\
        {   0.769421,   0.559017,  -0.309017 },\
        {   0.293893,   0.904508,  -0.309017 },\
        {  -0.293893,   0.904508,  -0.309017 },\
        {  -0.769421,   0.559017,  -0.309017 },\
        {  -0.951057,   0.000000,  -0.309017 },\
        {  -0.769421,  -0.559017,  -0.309017 },\
        {  -0.293893,  -0.904508,  -0.309017 },\
        {   0.293893,  -0.904508,  -0.309017 },\
        {   0.769421,  -0.559017,  -0.309017 },\
        {   0.587785,   0.000000,  -0.809017 },\
        {   0.475528,   0.345492,  -0.809017 },\
        {   0.181636,   0.559017,  -0.809017 },\
        {  -0.181636,   0.559017,  -0.809017 },\
        {  -0.475528,   0.345492,  -0.809017 },\
        {  -0.587785,   0.000000,  -0.809017 },\
        {  -0.475528,  -0.345492,  -0.809017 },\
        {  -0.181636,  -0.559017,  -0.809017 },\
        {   0.181636,  -0.559017,  -0.809017 },\
        {   0.475528,  -0.345492,  -0.809017 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     2,     3 },\
        {     1,    33,    32 },\
        {     0,     3,     4 },\
        {     1,    34,    33 },\
        {     0,     4,     5 },\
        {     1,    35,    34 },\
        {     0,     5,     6 },\
        {     1,    36,    35 },\
        {     0,     6,     7 },\
        {     1,    37,    36 },\
        {     0,     7,     8 },\
        {     1,    38,    37 },\
        {     0,     8,     9 },\
        {     1,    39,    38 },\
        {     0,     9,    10 },\
        {     1,    40,    39 },\
        {     0,    10,    11 },\
        {     1,    41,    40 },\
        {     0,    11,     2 },\
        {     1,    32,    41 },\
        {    12,     3,     2 },\
        {    12,    13,     3 },\
        {    13,     4,     3 },\
        {    13,    14,     4 },\
        {    14,     5,     4 },\
        {    14,    15,     5 },\
        {    15,     6,     5 },\
        {    15,    16,     6 },\
        {    16,     7,     6 },\
        {    16,    17,     7 },\
        {    17,     8,     7 },\
        {    17,    18,     8 },\
        {    18,     9,     8 },\
        {    18,    19,     9 },\
        {    19,    10,     9 },\
        {    19,    20,    10 },\
        {    20,    11,    10 },\
        {    20,    21,    11 },\
        {    21,     2,    11 },\
        {    21,    12,     2 },\
        {    22,    13,    12 },\
        {    22,    23,    13 },\
        {    23,    14,    13 },\
        {    23,    24,    14 },\
        {    24,    15,    14 },\
        {    24,    25,    15 },\
        {    25,    16,    15 },\
        {    25,    26,    16 },\
        {    26,    17,    16 },\
        {    26,    27,    17 },\
        {    27,    18,    17 },\
        {    27,    28,    18 },\
        {    28,    19,    18 },\
        {    28,    29,    19 },\
        {    29,    20,    19 },\
        {    29,    30,    20 },\
        {    30,    21,    20 },\
        {    30,    31,    21 },\
        {    31,    12,    21 },\
        {    31,    22,    12 },\
        {    32,    23,    22 },\
        {    32,    33,    23 },\
        {    33,    24,    23 },\
        {    33,    34,    24 },\
        {    34,    25,    24 },\
        {    34,    35,    25 },\
        {    35,    26,    25 },\
        {    35,    36,    26 },\
        {    36,    27,    26 },\
        {    36,    37,    27 },\
        {    37,    28,    27 },\
        {    37,    38,    28 },\
        {    38,    29,    28 },\
        {    38,    39,    29 },\
        {    39,    30,    29 },\
        {    39,    40,    30 },\
        {    40,    31,    30 },\
        {    40,    41,    31 },\
        {    41,    22,    31 },\
        {    41,    32,    22 } \
    };

    auto output_tm = open3d::CreateMeshSphere(1.0, 5);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCylinder)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        {   0.000000,   0.000000,   1.000000 },\
        {   0.000000,   0.000000,  -1.000000 },\
        {   1.000000,   0.000000,   1.000000 },\
        {   0.309017,   0.951057,   1.000000 },\
        {  -0.809017,   0.587785,   1.000000 },\
        {  -0.809017,  -0.587785,   1.000000 },\
        {   0.309017,  -0.951057,   1.000000 },\
        {   1.000000,   0.000000,   0.500000 },\
        {   0.309017,   0.951057,   0.500000 },\
        {  -0.809017,   0.587785,   0.500000 },\
        {  -0.809017,  -0.587785,   0.500000 },\
        {   0.309017,  -0.951057,   0.500000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   0.309017,   0.951057,   0.000000 },\
        {  -0.809017,   0.587785,   0.000000 },\
        {  -0.809017,  -0.587785,   0.000000 },\
        {   0.309017,  -0.951057,   0.000000 },\
        {   1.000000,   0.000000,  -0.500000 },\
        {   0.309017,   0.951057,  -0.500000 },\
        {  -0.809017,   0.587785,  -0.500000 },\
        {  -0.809017,  -0.587785,  -0.500000 },\
        {   0.309017,  -0.951057,  -0.500000 },\
        {   1.000000,   0.000000,  -1.000000 },\
        {   0.309017,   0.951057,  -1.000000 },\
        {  -0.809017,   0.587785,  -1.000000 },\
        {  -0.809017,  -0.587785,  -1.000000 },\
        {   0.309017,  -0.951057,  -1.000000 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     2,     3 },\
        {     1,    23,    22 },\
        {     0,     3,     4 },\
        {     1,    24,    23 },\
        {     0,     4,     5 },\
        {     1,    25,    24 },\
        {     0,     5,     6 },\
        {     1,    26,    25 },\
        {     0,     6,     2 },\
        {     1,    22,    26 },\
        {     7,     3,     2 },\
        {     7,     8,     3 },\
        {     8,     4,     3 },\
        {     8,     9,     4 },\
        {     9,     5,     4 },\
        {     9,    10,     5 },\
        {    10,     6,     5 },\
        {    10,    11,     6 },\
        {    11,     2,     6 },\
        {    11,     7,     2 },\
        {    12,     8,     7 },\
        {    12,    13,     8 },\
        {    13,     9,     8 },\
        {    13,    14,     9 },\
        {    14,    10,     9 },\
        {    14,    15,    10 },\
        {    15,    11,    10 },\
        {    15,    16,    11 },\
        {    16,     7,    11 },\
        {    16,    12,     7 },\
        {    17,    13,    12 },\
        {    17,    18,    13 },\
        {    18,    14,    13 },\
        {    18,    19,    14 },\
        {    19,    15,    14 },\
        {    19,    20,    15 },\
        {    20,    16,    15 },\
        {    20,    21,    16 },\
        {    21,    12,    16 },\
        {    21,    17,    12 },\
        {    22,    18,    17 },\
        {    22,    23,    18 },\
        {    23,    19,    18 },\
        {    23,    24,    19 },\
        {    24,    20,    19 },\
        {    24,    25,    20 },\
        {    25,    21,    20 },\
        {    25,    26,    21 },\
        {    26,    17,    21 },\
        {    26,    22,    17 } \
    };

    auto output_tm = open3d::CreateMeshCylinder(1.0, 2.0, 5);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCone)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        {   0.000000,   0.000000,   0.000000 },\
        {   0.000000,   0.000000,   2.000000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   0.309017,   0.951057,   0.000000 },\
        {  -0.809017,   0.587785,   0.000000 },\
        {  -0.809017,  -0.587785,   0.000000 },\
        {   0.309017,  -0.951057,   0.000000 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     3,     2 },\
        {     1,     2,     3 },\
        {     0,     4,     3 },\
        {     1,     3,     4 },\
        {     0,     5,     4 },\
        {     1,     4,     5 },\
        {     0,     6,     5 },\
        {     1,     5,     6 },\
        {     0,     2,     6 },\
        {     1,     6,     2 } \
    };

    auto output_tm = open3d::CreateMeshCone(1.0, 2.0, 5);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshArrow)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        {   0.000000,   0.000000,   2.000000 },\
        {   0.000000,   0.000000,   0.000000 },\
        {   1.000000,   0.000000,   2.000000 },\
        {   0.309017,   0.951057,   2.000000 },\
        {  -0.809017,   0.587785,   2.000000 },\
        {  -0.809017,  -0.587785,   2.000000 },\
        {   0.309017,  -0.951057,   2.000000 },\
        {   1.000000,   0.000000,   1.500000 },\
        {   0.309017,   0.951057,   1.500000 },\
        {  -0.809017,   0.587785,   1.500000 },\
        {  -0.809017,  -0.587785,   1.500000 },\
        {   0.309017,  -0.951057,   1.500000 },\
        {   1.000000,   0.000000,   1.000000 },\
        {   0.309017,   0.951057,   1.000000 },\
        {  -0.809017,   0.587785,   1.000000 },\
        {  -0.809017,  -0.587785,   1.000000 },\
        {   0.309017,  -0.951057,   1.000000 },\
        {   1.000000,   0.000000,   0.500000 },\
        {   0.309017,   0.951057,   0.500000 },\
        {  -0.809017,   0.587785,   0.500000 },\
        {  -0.809017,  -0.587785,   0.500000 },\
        {   0.309017,  -0.951057,   0.500000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   0.309017,   0.951057,   0.000000 },\
        {  -0.809017,   0.587785,   0.000000 },\
        {  -0.809017,  -0.587785,   0.000000 },\
        {   0.309017,  -0.951057,   0.000000 },\
        {   0.000000,   0.000000,   2.000000 },\
        {   0.000000,   0.000000,   3.000000 },\
        {   1.500000,   0.000000,   2.000000 },\
        {   0.463525,   1.426585,   2.000000 },\
        {  -1.213525,   0.881678,   2.000000 },\
        {  -1.213525,  -0.881678,   2.000000 },\
        {   0.463525,  -1.426585,   2.000000 } \
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     2,     3 },\
        {     1,    23,    22 },\
        {     0,     3,     4 },\
        {     1,    24,    23 },\
        {     0,     4,     5 },\
        {     1,    25,    24 },\
        {     0,     5,     6 },\
        {     1,    26,    25 },\
        {     0,     6,     2 },\
        {     1,    22,    26 },\
        {     7,     3,     2 },\
        {     7,     8,     3 },\
        {     8,     4,     3 },\
        {     8,     9,     4 },\
        {     9,     5,     4 },\
        {     9,    10,     5 },\
        {    10,     6,     5 },\
        {    10,    11,     6 },\
        {    11,     2,     6 },\
        {    11,     7,     2 },\
        {    12,     8,     7 },\
        {    12,    13,     8 },\
        {    13,     9,     8 },\
        {    13,    14,     9 },\
        {    14,    10,     9 },\
        {    14,    15,    10 },\
        {    15,    11,    10 },\
        {    15,    16,    11 },\
        {    16,     7,    11 },\
        {    16,    12,     7 },\
        {    17,    13,    12 },\
        {    17,    18,    13 },\
        {    18,    14,    13 },\
        {    18,    19,    14 },\
        {    19,    15,    14 },\
        {    19,    20,    15 },\
        {    20,    16,    15 },\
        {    20,    21,    16 },\
        {    21,    12,    16 },\
        {    21,    17,    12 },\
        {    22,    18,    17 },\
        {    22,    23,    18 },\
        {    23,    19,    18 },\
        {    23,    24,    19 },\
        {    24,    20,    19 },\
        {    24,    25,    20 },\
        {    25,    21,    20 },\
        {    25,    26,    21 },\
        {    26,    17,    21 },\
        {    26,    22,    17 },\
        {    27,    30,    29 },\
        {    28,    29,    30 },\
        {    27,    31,    30 },\
        {    28,    30,    31 },\
        {    27,    32,    31 },\
        {    28,    31,    32 },\
        {    27,    33,    32 },\
        {    28,    32,    33 },\
        {    27,    29,    33 },\
        {    28,    33,    29 } \
    };

    auto output_tm = open3d::CreateMeshArrow(1.0, 1.5, 2.0, 1.0, 5);

    EXPECT_EQ(ref_vertices.size(), output_tm->vertices_.size());
    for (size_t i = 0; i < output_tm->vertices_.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(ref_triangles.size(), output_tm->triangles_.size());
    for (size_t i = 0; i < output_tm->triangles_.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(TriangleMesh, CreateMeshCoordinateFrame)
{
    vector<Eigen::Vector3d> ref_vertices =
    {
        {   0.000000,   0.000000,   0.006000 },\
        {   0.000893,  -0.000290,   0.005926 },\
        {   0.001763,  -0.000573,   0.005706 },\
        {   0.002591,  -0.000842,   0.005346 },\
        {   0.003354,  -0.001090,   0.004854 },\
        {   0.004035,  -0.001311,   0.004243 },\
        {   0.004617,  -0.001500,   0.003527 },\
        {   0.005084,  -0.001652,   0.002724 },\
        {   0.005427,  -0.001763,   0.001854 },\
        {   0.005636,  -0.001831,   0.000939 },\
        {   0.005706,  -0.001854,   0.000000 },\
        {   0.005636,  -0.001831,  -0.000939 },\
        {   0.005427,  -0.001763,  -0.001854 },\
        {   0.005084,  -0.001652,  -0.002724 },\
        {   0.004617,  -0.001500,  -0.003527 },\
        {   0.004035,  -0.001311,  -0.004243 },\
        {   0.003354,  -0.001090,  -0.004854 },\
        {   0.002591,  -0.000842,  -0.005346 },\
        {   0.001763,  -0.000573,  -0.005706 },\
        {   0.000893,  -0.000290,  -0.005926 },\
        {   0.060000,   0.001082,  -0.003329 },\
        {   0.020000,   0.001082,  -0.003329 },\
        {   0.080000,  -0.001854,  -0.005706 },\
        {  -0.002057,   0.060000,  -0.002832 },\
        {  -0.002057,   0.020000,  -0.002832 },\
        {   0.000000,   0.080000,  -0.006000 },\
        {  -0.002832,   0.002057,   0.060000 },\
        {  -0.002832,   0.002057,   0.020000 },\
        {  -0.001854,   0.005706,   0.080000 },\
    };

    vector<Eigen::Vector3d> ref_vertex_normals =
    {
        {   0.000000,   0.000000,   1.000000 },\
        {   0.171274,  -0.062054,   0.983267 },\
        {   0.303522,  -0.104788,   0.947045 },\
        {   0.436905,  -0.147738,   0.887292 },\
        {   0.561823,  -0.187794,   0.805660 },\
        {   0.673910,  -0.223553,   0.704180 },\
        {   0.769966,  -0.253989,   0.585357 },\
        {   0.847434,  -0.278292,   0.452117 },\
        {   0.904311,  -0.295833,   0.307743 },\
        {   0.939146,  -0.306162,   0.155790 },\
        {   0.951057,  -0.309017,  -0.000000 },\
        {   0.939742,  -0.304326,  -0.155790 },\
        {   0.905489,  -0.292207,  -0.307743 },\
        {   0.849164,  -0.272966,  -0.452117 },\
        {   0.772206,  -0.247093,  -0.585357 },\
        {   0.676605,  -0.215256,  -0.704180 },\
        {   0.564907,  -0.178302,  -0.805660 },\
        {   0.440301,  -0.137284,  -0.887292 },\
        {   0.307147,  -0.093630,  -0.947045 },\
        {   0.175038,  -0.050470,  -0.983267 },\
        {   0.000000,   0.309017,  -0.951057 },\
        {   0.000000,   0.309017,  -0.951057 },\
        {   0.000000,  -0.309017,  -0.951057 },\
        {  -0.587785,   0.000000,  -0.809017 },\
        {  -0.587785,   0.000000,  -0.809017 },\
        {   0.000000,   0.000000,  -1.000000 },\
        {  -0.809017,   0.587785,   0.000000 },\
        {  -0.809017,   0.587785,   0.000000 },\
        {  -0.309017,   0.951057,   0.000000 },\
    };

    vector<Eigen::Vector3d> ref_vertex_colors =
    {
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   0.500000,   0.500000,   0.500000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   0.000000,   1.000000,   0.000000 },\
        {   0.000000,   1.000000,   0.000000 },\
        {   0.000000,   1.000000,   0.000000 },\
        {   0.000000,   0.000000,   1.000000 },\
        {   0.000000,   0.000000,   1.000000 },\
        {   0.000000,   0.000000,   1.000000 },\
    };

    vector<Eigen::Vector3d> ref_triangles =
    {
        {     0,     2,     3 },\
        {     0,    22,    23 },\
        {    42,     3,     2 },\
        {    62,    23,    22 },\
        {    82,    43,    42 },\
        {   102,    63,    62 },\
        {   122,    83,    82 },\
        {   142,   103,   102 },\
        {   162,   123,   122 },\
        {   182,   143,   142 },\
        {   202,   163,   162 },\
        {   222,   183,   182 },\
        {   242,   203,   202 },\
        {   262,   223,   222 },\
        {   282,   243,   242 },\
        {   302,   263,   262 },\
        {   322,   283,   282 },\
        {   342,   303,   302 },\
        {   362,   323,   322 },\
        {   382,   343,   342 },\
        {   402,   363,   362 },\
        {   422,   383,   382 },\
        {   442,   403,   402 },\
        {   462,   423,   422 },\
        {   482,   443,   442 },\
        {   502,   463,   462 },\
        {   522,   483,   482 },\
        {   542,   503,   502 },\
        {   562,   523,   522 },\
        {   582,   543,   542 },\
        {   602,   563,   562 },\
        {   622,   583,   582 },\
        {   642,   603,   602 },\
        {   662,   623,   622 },\
        {   682,   643,   642 },\
        {   702,   663,   662 },\
        {   722,   683,   682 },\
        {   742,   703,   702 },\
        {   762,   764,   765 },\
        {   784,   765,   764 },\
        {   804,   785,   784 },\
        {   824,   805,   804 },\
        {   844,   825,   824 },\
        {   864,   867,   866 },\
        {   886,   888,   889 },\
        {   908,   889,   888 },\
        {   928,   909,   908 },\
        {   948,   929,   928 },\
        {   968,   949,   948 },\
        {   988,   991,   990 },\
        {  1010,  1012,  1013 },\
        {  1032,  1013,  1012 },\
        {  1052,  1033,  1032 },\
        {  1072,  1053,  1052 },\
        {  1092,  1073,  1072 },\
        {  1112,  1115,  1114 },\
    };

    vector<Eigen::Vector3d> ref_triangle_normals =
    {
        {   0.078458,   0.006175,   0.996898 },\
        {  -0.078458,  -0.006175,   0.996898 },\
        {   0.233406,   0.018369,   0.972206 },\
        {  -0.233406,  -0.018369,   0.972206 },\
        {   0.382510,   0.030104,   0.923461 },\
        {  -0.382510,  -0.030104,   0.923461 },\
        {   0.522057,   0.041087,   0.851920 },\
        {  -0.522057,  -0.041087,   0.851920 },\
        {   0.648601,   0.051046,   0.759415 },\
        {  -0.648601,  -0.051046,   0.759415 },\
        {   0.759048,   0.059738,   0.648288 },\
        {  -0.759048,  -0.059738,   0.648288 },\
        {   0.850727,   0.066954,   0.521326 },\
        {  -0.850727,  -0.066954,   0.521326 },\
        {   0.921447,   0.072519,   0.381676 },\
        {  -0.921447,  -0.072519,   0.381676 },\
        {   0.969535,   0.076304,   0.232765 },\
        {  -0.969535,  -0.076304,   0.232765 },\
        {   0.993863,   0.078219,   0.078219 },\
        {  -0.993863,  -0.078219,   0.078219 },\
        {   0.993863,   0.078219,  -0.078219 },\
        {  -0.993863,  -0.078219,  -0.078219 },\
        {   0.969535,   0.076304,  -0.232765 },\
        {  -0.969535,  -0.076304,  -0.232765 },\
        {   0.921447,   0.072519,  -0.381676 },\
        {  -0.921447,  -0.072519,  -0.381676 },\
        {   0.850727,   0.066954,  -0.521326 },\
        {  -0.850727,  -0.066954,  -0.521326 },\
        {   0.759048,   0.059738,  -0.648288 },\
        {  -0.759048,  -0.059738,  -0.648288 },\
        {   0.648601,   0.051046,  -0.759415 },\
        {  -0.648601,  -0.051046,  -0.759415 },\
        {   0.522057,   0.041087,  -0.851920 },\
        {  -0.522057,  -0.041087,  -0.851920 },\
        {   0.382510,   0.030104,  -0.923461 },\
        {  -0.382510,  -0.030104,  -0.923461 },\
        {   0.233406,   0.018369,  -0.972206 },\
        {  -0.233406,  -0.018369,  -0.972206 },\
        {   1.000000,   0.000000,   0.000000 },\
        {   0.000000,   0.987688,   0.156434 },\
        {   0.000000,   0.987688,   0.156434 },\
        {   0.000000,   0.987688,   0.156434 },\
        {   0.000000,   0.987688,   0.156434 },\
        {  -1.000000,   0.000000,   0.000000 },\
        {   0.000000,   1.000000,   0.000000 },\
        {   0.156434,   0.000000,   0.987688 },\
        {   0.156434,   0.000000,   0.987688 },\
        {   0.156434,   0.000000,   0.987688 },\
        {   0.156434,   0.000000,   0.987688 },\
        {   0.000000,  -1.000000,   0.000000 },\
        {   0.000000,   0.000000,   1.000000 },\
        {   0.987688,   0.156434,   0.000000 },\
        {   0.987688,   0.156434,   0.000000 },\
        {   0.987688,   0.156434,   0.000000 },\
        {   0.987688,   0.156434,   0.000000 },\
        {   0.000000,   0.000000,  -1.000000 },\
    };

    auto output_tm = open3d::CreateMeshCoordinateFrame(0.1);

    // this procedure generates too many values to save here
    // using this stride in order to sample the total number of generated values
    int stride = 40;
    EXPECT_EQ(1134, output_tm->vertices_.size());
    for (size_t i = 0; i < ref_vertices.size(); i++)
    {
        EXPECT_NEAR(ref_vertices[i](0, 0), output_tm->vertices_[i * stride](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](1, 0), output_tm->vertices_[i * stride](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertices[i](2, 0), output_tm->vertices_[i * stride](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(1134, output_tm->vertex_normals_.size());
    for (size_t i = 0; i < ref_vertex_normals.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_normals[i](0, 0), output_tm->vertex_normals_[i * stride](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](1, 0), output_tm->vertex_normals_[i * stride](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_normals[i](2, 0), output_tm->vertex_normals_[i * stride](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(1134, output_tm->vertex_colors_.size());
    for (size_t i = 0; i < ref_vertex_colors.size(); i++)
    {
        EXPECT_NEAR(ref_vertex_colors[i](0, 0), output_tm->vertex_colors_[i * stride](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](1, 0), output_tm->vertex_colors_[i * stride](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_vertex_colors[i](2, 0), output_tm->vertex_colors_[i * stride](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2240, output_tm->triangles_.size());
    for (size_t i = 0; i < ref_triangles.size(); i++)
    {
        EXPECT_NEAR(ref_triangles[i](0, 0), output_tm->triangles_[i * stride](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](1, 0), output_tm->triangles_[i * stride](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangles[i](2, 0), output_tm->triangles_[i * stride](2, 0), UnitTest::THRESHOLD_1E_6);
    }

    EXPECT_EQ(2240, output_tm->triangle_normals_.size());
    for (size_t i = 0; i < ref_triangle_normals.size(); i++)
    {
        EXPECT_NEAR(ref_triangle_normals[i](0, 0), output_tm->triangle_normals_[i * stride](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](1, 0), output_tm->triangle_normals_[i * stride](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_triangle_normals[i](2, 0), output_tm->triangle_normals_[i * stride](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}
