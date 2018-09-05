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
#include "Core/Geometry/PointCloud.h"

#include <algorithm>
using namespace std;

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, Constructor)
{
    open3d::PointCloud pc;

    // inherited from Geometry2D
    EXPECT_EQ(open3d::Geometry::GeometryType::PointCloud, pc.GetGeometryType());
    EXPECT_EQ(3, pc.Dimension());

    // public member variables
    EXPECT_EQ(0, pc.points_.size());
    EXPECT_EQ(0, pc.normals_.size());
    EXPECT_EQ(0, pc.colors_.size());

    // public members
    EXPECT_TRUE(pc.IsEmpty());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), pc.GetMinBound());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), pc.GetMaxBound());
    EXPECT_FALSE(pc.HasPoints());
    EXPECT_FALSE(pc.HasNormals());
    EXPECT_FALSE(pc.HasColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_MemberData)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, Clear)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    pc.points_.resize(size);
    pc.normals_.resize(size);
    pc.colors_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);
    UnitTest::Rand(pc.normals_, vmin, vmax);
    UnitTest::Rand(pc.colors_, vmin, vmax);

    EXPECT_TRUE(pc.HasPoints());

    pc.Clear();

    // public members
    EXPECT_TRUE(pc.IsEmpty());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), pc.GetMinBound());
    EXPECT_EQ(Eigen::Vector3d(0.0, 0.0, 0.0), pc.GetMaxBound());
    EXPECT_FALSE(pc.HasPoints());
    EXPECT_FALSE(pc.HasNormals());
    EXPECT_FALSE(pc.HasColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, IsEmpty)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    EXPECT_TRUE(pc.IsEmpty());

    pc.points_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);

    EXPECT_FALSE(pc.IsEmpty());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, GetMinBound)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    pc.points_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);

    Eigen::Vector3d minBound = pc.GetMinBound();

    EXPECT_NEAR(20.0230, minBound(0, 0), UnitTest::THRESHOLD_1E_3);
    EXPECT_NEAR(3.23146, minBound(1, 0), UnitTest::THRESHOLD_1E_3);
    EXPECT_NEAR(3.57857, minBound(2, 0), UnitTest::THRESHOLD_1E_3);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, GetMaxBound)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    pc.points_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);

    Eigen::Vector3d maxBound = pc.GetMaxBound();

    EXPECT_NEAR(997.798999, maxBound(0, 0), UnitTest::THRESHOLD_1E_3);
    EXPECT_NEAR(998.924518, maxBound(1, 0), UnitTest::THRESHOLD_1E_3);
    EXPECT_NEAR(999.993571, maxBound(2, 0), UnitTest::THRESHOLD_1E_3);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, Transform)
{
    vector<Eigen::Vector3d> ref_points =
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

    vector<Eigen::Vector3d> ref_normals =
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
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    pc.normals_.resize(size);
    UnitTest::Rand(pc.normals_, vmin, vmax);

    Eigen::Matrix4d transformation;
    transformation << 0.10, 0.20, 0.30, 0.40,
                      0.50, 0.60, 0.70, 0.80,
                      0.90, 0.10, 0.11, 0.12,
                      0.13, 0.14, 0.15, 0.16;

    pc.Transform(transformation);

    for (size_t i = 0; i < pc.points_.size(); i++)
    {
        EXPECT_NEAR(ref_points[i](0, 0), pc.points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](1, 0), pc.points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](2, 0), pc.points_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_normals[i](0, 0), pc.normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_normals[i](1, 0), pc.normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_normals[i](2, 0), pc.normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, HasPoints)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    EXPECT_FALSE(pc.HasPoints());

    pc.points_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);

    EXPECT_TRUE(pc.HasPoints());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, HasNormals)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    EXPECT_FALSE(pc.HasNormals());

    pc.points_.resize(size);
    pc.normals_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);
    UnitTest::Rand(pc.normals_, vmin, vmax);

    EXPECT_TRUE(pc.HasNormals());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, HasColors)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    EXPECT_FALSE(pc.HasColors());

    pc.points_.resize(size);
    pc.colors_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);
    UnitTest::Rand(pc.colors_, vmin, vmax);

    EXPECT_TRUE(pc.HasColors());
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, NormalizeNormals)
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
        {   0.043296,   0.989703,   0.136434 } \
    };

    int size = 20;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    pc.normals_.resize(size);

    UnitTest::Rand(pc.normals_, vmin, vmax);

    pc.NormalizeNormals();

    for (size_t i = 0; i < pc.normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), pc.normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), pc.normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), pc.normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, PaintUniformColor)
{
    int size = 100;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    open3d::PointCloud pc;

    EXPECT_TRUE(pc.IsEmpty());

    pc.points_.resize(size);

    UnitTest::Rand(pc.points_, vmin, vmax);

    EXPECT_FALSE(pc.HasColors());

    pc.PaintUniformColor(Eigen::Vector3d(233, 171, 53));

    EXPECT_TRUE(pc.HasColors());

    for (size_t i = 0; i < pc.colors_.size(); i++)
    {
        EXPECT_DOUBLE_EQ(233, pc.colors_[i](0, 0));
        EXPECT_DOUBLE_EQ(171, pc.colors_[i](1, 0));
        EXPECT_DOUBLE_EQ( 53, pc.colors_[i](2, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, OperatorAppend)
{
    Eigen::Vector3d p0 = { 150, 230, 400 };
    Eigen::Vector3d p1 = { 250, 230, 400 };
    Eigen::Vector3d p2 = { 150, 130, 400 };
    Eigen::Vector3d p3 = { 150, 230, 300 };

    Eigen::Vector3d p4 = {  75, 115, 200 };
    Eigen::Vector3d p5 = { 125, 115, 200 };
    Eigen::Vector3d p6 = {  75,  65, 200 };
    Eigen::Vector3d p7 = {  75, 115, 150 };

    vector<Eigen::Vector3d> p;
    p.push_back(p0);
    p.push_back(p1);
    p.push_back(p2);
    p.push_back(p3);
    p.push_back(p4);
    p.push_back(p3);
    p.push_back(p4);
    p.push_back(p5);
    p.push_back(p6);
    p.push_back(p7);

    Eigen::Vector3d n0 = { 0.150, 0.230, 0.400 };
    Eigen::Vector3d n1 = { 0.250, 0.230, 0.400 };
    Eigen::Vector3d n2 = { 0.150, 0.130, 0.400 };
    Eigen::Vector3d n3 = { 0.150, 0.230, 0.300 };

    Eigen::Vector3d n4 = { 0.075, 0.115, 0.200 };
    Eigen::Vector3d n5 = { 0.125, 0.115, 0.200 };
    Eigen::Vector3d n6 = { 0.075, 0.065, 0.200 };
    Eigen::Vector3d n7 = { 0.075, 0.115, 0.150 };

    vector<Eigen::Vector3d> n;
    n.push_back(n0);
    n.push_back(n1);
    n.push_back(n2);
    n.push_back(n3);
    n.push_back(n4);
    n.push_back(n3);
    n.push_back(n4);
    n.push_back(n5);
    n.push_back(n6);
    n.push_back(n7);

    open3d::PointCloud pc0;
    open3d::PointCloud pc1;

    pc0.points_.push_back(p0);
    pc0.points_.push_back(p1);
    pc0.points_.push_back(p2);
    pc0.points_.push_back(p3);
    pc0.points_.push_back(p4);

    pc1.points_.push_back(p3);
    pc1.points_.push_back(p4);
    pc1.points_.push_back(p5);
    pc1.points_.push_back(p6);
    pc1.points_.push_back(p7);

    pc0.normals_.push_back(n0);
    pc0.normals_.push_back(n1);
    pc0.normals_.push_back(n2);
    pc0.normals_.push_back(n3);
    pc0.normals_.push_back(n4);

    pc1.normals_.push_back(n3);
    pc1.normals_.push_back(n4);
    pc1.normals_.push_back(n5);
    pc1.normals_.push_back(n6);
    pc1.normals_.push_back(n7);

    pc0.PaintUniformColor(Eigen::Vector3d(233, 171, 53));
    pc1.PaintUniformColor(Eigen::Vector3d( 53, 233, 171));

    pc0 += pc1;

    EXPECT_EQ(p.size(), pc0.points_.size());
    for (size_t i = 0; i < 10; i++)
        EXPECT_EQ(p[i], pc0.points_[i]);

    EXPECT_EQ(n.size(), pc0.normals_.size());
    for (size_t i = 0; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ(n[i](0, 0), pc0.normals_[i](0, 0));
        EXPECT_DOUBLE_EQ(n[i](1, 0), pc0.normals_[i](1, 0));
        EXPECT_DOUBLE_EQ(n[i](2, 0), pc0.normals_[i](2, 0));
    }

    EXPECT_EQ(p.size(), pc0.colors_.size());
    for (size_t i = 0; i < 5; i++)
    {
        EXPECT_DOUBLE_EQ(233, pc0.colors_[i](0, 0));
        EXPECT_DOUBLE_EQ(171, pc0.colors_[i](1, 0));
        EXPECT_DOUBLE_EQ( 53, pc0.colors_[i](2, 0));
    }

    for (size_t i = 5; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ( 53, pc0.colors_[i](0, 0));
        EXPECT_DOUBLE_EQ(233, pc0.colors_[i](1, 0));
        EXPECT_DOUBLE_EQ(171, pc0.colors_[i](2, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, OperatorADD)
{
    Eigen::Vector3d p0 = { 150, 230, 400 };
    Eigen::Vector3d p1 = { 250, 230, 400 };
    Eigen::Vector3d p2 = { 150, 130, 400 };
    Eigen::Vector3d p3 = { 150, 230, 300 };

    Eigen::Vector3d p4 = {  75, 115, 200 };
    Eigen::Vector3d p5 = { 125, 115, 200 };
    Eigen::Vector3d p6 = {  75,  65, 200 };
    Eigen::Vector3d p7 = {  75, 115, 150 };

    vector<Eigen::Vector3d> p;
    p.push_back(p0);
    p.push_back(p1);
    p.push_back(p2);
    p.push_back(p3);
    p.push_back(p4);
    p.push_back(p3);
    p.push_back(p4);
    p.push_back(p5);
    p.push_back(p6);
    p.push_back(p7);

    Eigen::Vector3d n0 = { 0.150, 0.230, 0.400 };
    Eigen::Vector3d n1 = { 0.250, 0.230, 0.400 };
    Eigen::Vector3d n2 = { 0.150, 0.130, 0.400 };
    Eigen::Vector3d n3 = { 0.150, 0.230, 0.300 };

    Eigen::Vector3d n4 = { 0.075, 0.115, 0.200 };
    Eigen::Vector3d n5 = { 0.125, 0.115, 0.200 };
    Eigen::Vector3d n6 = { 0.075, 0.065, 0.200 };
    Eigen::Vector3d n7 = { 0.075, 0.115, 0.150 };

    vector<Eigen::Vector3d> n;
    n.push_back(n0);
    n.push_back(n1);
    n.push_back(n2);
    n.push_back(n3);
    n.push_back(n4);
    n.push_back(n3);
    n.push_back(n4);
    n.push_back(n5);
    n.push_back(n6);
    n.push_back(n7);

    open3d::PointCloud pc0;
    open3d::PointCloud pc1;

    pc0.points_.push_back(p0);
    pc0.points_.push_back(p1);
    pc0.points_.push_back(p2);
    pc0.points_.push_back(p3);
    pc0.points_.push_back(p4);

    pc1.points_.push_back(p3);
    pc1.points_.push_back(p4);
    pc1.points_.push_back(p5);
    pc1.points_.push_back(p6);
    pc1.points_.push_back(p7);

    pc0.normals_.push_back(n0);
    pc0.normals_.push_back(n1);
    pc0.normals_.push_back(n2);
    pc0.normals_.push_back(n3);
    pc0.normals_.push_back(n4);

    pc1.normals_.push_back(n3);
    pc1.normals_.push_back(n4);
    pc1.normals_.push_back(n5);
    pc1.normals_.push_back(n6);
    pc1.normals_.push_back(n7);

    pc0.PaintUniformColor(Eigen::Vector3d(233, 171, 53));
    pc1.PaintUniformColor(Eigen::Vector3d( 53, 233, 171));

    open3d::PointCloud pc = pc0 + pc1;

    EXPECT_EQ(5, pc0.points_.size());
    EXPECT_EQ(5, pc0.normals_.size());
    EXPECT_EQ(5, pc0.colors_.size());

    EXPECT_EQ(p.size(), pc.points_.size());
    for (size_t i = 0; i < 10; i++)
        EXPECT_EQ(p[i], pc.points_[i]);

    EXPECT_EQ(n.size(), pc.normals_.size());
    for (size_t i = 0; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ(n[i](0, 0), pc.normals_[i](0, 0));
        EXPECT_DOUBLE_EQ(n[i](1, 0), pc.normals_[i](1, 0));
        EXPECT_DOUBLE_EQ(n[i](2, 0), pc.normals_[i](2, 0));
    }

    EXPECT_EQ(p.size(), pc.colors_.size());
    for (size_t i = 0; i < 5; i++)
    {
        EXPECT_DOUBLE_EQ(233, pc.colors_[i](0, 0));
        EXPECT_DOUBLE_EQ(171, pc.colors_[i](1, 0));
        EXPECT_DOUBLE_EQ( 53, pc.colors_[i](2, 0));
    }

    for (size_t i = 5; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ( 53, pc.colors_[i](0, 0));
        EXPECT_DOUBLE_EQ(233, pc.colors_[i](1, 0));
        EXPECT_DOUBLE_EQ(171, pc.colors_[i](2, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromFile)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromDepthImage)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromRGBDImage)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, SelectDownSample)
{
    vector<Eigen::Vector3d> ref =
    {
        { 347.115873, 184.622468, 609.105862 },\
        { 416.501281, 169.607086, 906.803934 },\
        { 434.513296,   3.231460, 344.942955 },\
        { 598.481300, 833.243294, 233.891705 },\
        { 593.211455, 666.556591, 288.777775 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 910.972031, 482.490657, 215.824959 },\
        { 471.483429, 592.539919, 944.318096 },\
        { 165.974166, 440.104528, 880.075236 },\
        { 204.328611, 889.955644, 125.468476 },\
        { 950.104032,  52.529262, 521.563380 },\
        { 530.807988, 757.293832, 304.295150 },\
        { 619.596484, 281.059412, 786.002098 },\
        { 134.902412, 520.210070,  78.232142 },\
        { 436.496997, 958.636962, 918.930388 },\
        { 593.211455, 666.556591, 288.777775 },\
        { 992.228461, 576.971113, 877.613778 },\
        { 229.136980, 700.619965, 316.867137 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 359.095369, 552.485022, 579.429994 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 437.637597, 931.835056, 930.809795 },\
        { 771.357698, 526.744979, 769.913836 },\
        { 675.475980, 482.950280, 481.935823 },\
        { 352.458347, 807.724520, 919.026474 } \
    };

    int size = 100;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    vector<size_t> indices(size / 4);
    UnitTest::Rand<size_t>(indices, 0, size);

    // remove duplicates
    std::vector<size_t>::iterator it;
    it = unique(indices.begin(), indices.end());
    indices.resize(distance(indices.begin(), it));

    auto output_pc = open3d::SelectDownSample(pc, indices);

    for (size_t i = 0; i < indices.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), output_pc->points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), output_pc->points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), output_pc->points_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, VoxelDownSample)
{
    vector<Eigen::Vector3d> ref_points =
    {
        { 352.458347, 807.724520, 919.026474 },\
        { 400.228622, 891.529452, 283.314746 },\
        { 771.357698, 526.744979, 769.913836 },\
        { 296.031618, 637.552268, 524.287190 },\
        {  86.055848, 192.213846, 663.226927 },\
        { 512.932394, 839.112235, 612.639833 },\
        { 493.582987, 972.775024, 292.516784 },\
        { 335.222756, 768.229595, 277.774711 },\
        {  69.755276, 949.327075, 525.995350 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 553.969956, 477.397052, 628.870925 },\
        { 798.440033, 911.647358, 197.551369 },\
        { 890.232603, 348.892935,  64.171321 },\
        { 141.602555, 606.968876,  16.300572 },\
        {  20.023049, 457.701737,  63.095838 },\
        { 840.187717, 394.382927, 783.099224 },\
        { 156.679089, 400.944394, 129.790447 },\
        { 916.195068, 635.711728, 717.296929 },\
        { 242.886771, 137.231577, 804.176754 },\
        { 108.808802, 998.924518, 218.256905 } \
    };

    vector<Eigen::Vector3d> ref_normals =
    {
        {   0.276810,   0.634363,   0.721776 },\
        {   0.393351,   0.876209,   0.278446 },\
        {   0.637242,   0.435160,   0.636049 },\
        {   0.337582,   0.727038,   0.597875 },\
        {   0.123668,   0.276225,   0.953103 },\
        {   0.442687,   0.724197,   0.528740 },\
        {   0.437042,   0.861341,   0.259008 },\
        {   0.379636,   0.870011,   0.314576 },\
        {   0.064140,   0.872907,   0.483653 },\
        {   0.319521,   0.449696,   0.834074 },\
        {   0.574357,   0.494966,   0.652014 },\
        {   0.650271,   0.742470,   0.160891 },\
        {   0.928961,   0.364071,   0.066963 },\
        {   0.227116,   0.973517,   0.026144 },\
        {   0.043296,   0.989703,   0.136434 },\
        {   0.691871,   0.324763,   0.644860 },\
        {   0.348477,   0.891758,   0.288673 },\
        {   0.690989,   0.479450,   0.540981 },\
        {   0.285349,   0.161223,   0.944766 },\
        {   0.105818,   0.971468,   0.212258 } \
    };

    vector<Eigen::Vector3d> ref_colors =
    {
        {  89.876879, 205.969753, 234.351751 },\
        { 102.058299, 227.340010,  72.245260 },\
        { 196.696213, 134.319970, 196.328028 },\
        {  75.488063, 162.575828, 133.693233 },\
        {  21.944241,  49.014531, 169.122866 },\
        { 130.797761, 213.973620, 156.223157 },\
        { 125.863662, 248.057631,  74.591780 },\
        {  85.481803, 195.898547,  70.832551 },\
        {  17.787595, 242.078404, 134.128814 },\
        {  93.020041, 130.917232, 242.818580 },\
        { 141.262339, 121.736248, 160.362086 },\
        { 203.602209, 232.470076,  50.375599 },\
        { 227.009314,  88.967698,  16.363687 },\
        {  36.108652, 154.777063,   4.156646 },\
        {   5.105877, 116.713943,  16.089439 },\
        { 214.247868, 100.567646, 199.690302 },\
        {  39.953168, 102.240821,  33.096564 },\
        { 233.629742, 162.106491, 182.910717 },\
        {  61.936127,  34.994052, 205.065072 },\
        {  27.746245, 254.725752,  55.655511 } \
    };

    int size = 20;
    open3d::PointCloud pc;

    pc.points_.resize(size);
    pc.normals_.resize(size);
    pc.colors_.resize(size);

    UnitTest::Rand(pc.points_, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1000.0, 1000.0, 1000.0));
    UnitTest::Rand(pc.normals_, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(10.0, 10.0, 10.0));
    UnitTest::Rand(pc.colors_, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(255.0, 255.0, 255.0));

    double voxel_size = 0.5;
    auto output_pc = open3d::VoxelDownSample(pc, voxel_size);

    for (size_t i = 0; i < output_pc->points_.size(); i++)
    {
        EXPECT_NEAR(ref_points[i](0, 0), output_pc->points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](1, 0), output_pc->points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](2, 0), output_pc->points_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_normals[i](0, 0), output_pc->normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_normals[i](1, 0), output_pc->normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_normals[i](2, 0), output_pc->normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_colors[i](0, 0), output_pc->colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_colors[i](1, 0), output_pc->colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_colors[i](2, 0), output_pc->colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, UniformDownSample)
{
    vector<Eigen::Vector3d> ref =
    {
        { 840.187717, 394.382927, 783.099224 },\
        { 364.784473, 513.400910, 952.229725 },\
        { 156.679089, 400.944394, 129.790447 },\
        { 493.582987, 972.775024, 292.516784 },\
        {  69.755276, 949.327075, 525.995350 },\
        { 238.279954, 970.634132, 902.208073 },\
        { 437.637597, 931.835056, 930.809795 },\
        { 829.201093, 330.337130, 228.968171 },\
        { 398.436667, 814.766896, 684.218525 },\
        { 619.596484, 281.059412, 786.002098 },\
        { 103.171188, 126.075339, 495.444067 },\
        { 584.488501, 244.412736, 152.389792 },\
        { 176.210656, 240.062372, 797.798052 },\
        {  69.906398, 204.655086, 461.420473 },\
        { 997.798999,  54.057578, 870.539865 },\
        { 359.095369, 552.485022, 579.429994 },\
        { 747.809296, 628.909931,  35.420907 },\
        { 666.880335, 497.258519, 163.968003 },\
        { 328.777043, 231.427952,  74.160970 },\
        { 471.483429, 592.539919, 944.318096 },\
        { 675.475980, 482.950280, 481.935823 },\
        { 347.115873, 184.622468, 609.105862 },\
        { 532.440973,  87.643628, 260.496942 },\
        { 775.767243, 288.379412, 329.642079 },\
        { 764.871435, 699.075403, 121.143162 } \
    };

    int size = 100;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    size_t every_k_points = 4;
    auto output_pc = open3d::UniformDownSample(pc, every_k_points);

    for (size_t i = 0; i < output_pc->points_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), output_pc->points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), output_pc->points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), output_pc->points_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, CropPointCloud)
{
    int size = 100;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    Eigen::Vector3d minBound(200.0, 200.0, 200.0);
    Eigen::Vector3d maxBound(800.0, 800.0, 800.0);
    auto output_pc = open3d::CropPointCloud(pc, minBound, maxBound);

    for (size_t i = 0; i < output_pc->points_.size(); i++)
    {
        EXPECT_LE(minBound(0, 0), output_pc->points_[i](0, 0));
        EXPECT_LE(minBound(1, 0), output_pc->points_[i](1, 0));
        EXPECT_LE(minBound(2, 0), output_pc->points_[i](2, 0));

        EXPECT_GE(maxBound(0, 0), output_pc->points_[i](0, 0));
        EXPECT_GE(maxBound(1, 0), output_pc->points_[i](1, 0));
        EXPECT_GE(maxBound(2, 0), output_pc->points_[i](2, 0));
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, EstimateNormals)
{
    vector<Eigen::Vector3d> ref =
    {
        {   0.301081,   0.852348,   0.427614 },\
        {   0.551697,   0.828710,  -0.094182 },\
        {   0.080430,  -0.973307,   0.214953 },\
        {   0.222569,   0.630019,   0.744002 },\
        {   0.034279,   0.427528,   0.903352 },\
        {   0.711501,   0.596113,   0.372042 },\
        {   0.518930,   0.554754,   0.650353 },\
        {   0.491667,   0.570274,  -0.658067 },\
        {   0.324264,   0.745016,   0.582927 },\
        {   0.120885,  -0.989778,   0.075671 },\
        {   0.371510,   0.766621,   0.523710 },\
        {   0.874810,  -0.156827,  -0.458381 },\
        {   0.240195,   0.936743,  -0.254594 },\
        {   0.518743,   0.540549,   0.662354 },\
        {   0.240195,   0.936743,  -0.254594 },\
        {   0.081304,  -0.500530,  -0.861893 },\
        {   0.755134,  -0.525094,  -0.392491 },\
        {   0.721656,   0.540276,  -0.432799 },\
        {   0.157527,  -0.859267,  -0.486668 },\
        {   0.446840,   0.725678,   0.523188 },\
        {   0.019810,  -0.591008,  -0.806422 },\
        {   0.027069,   0.855357,   0.517331 },\
        {   0.478109,   0.869547,  -0.123690 },\
        {   0.101961,  -0.787444,  -0.607895 },\
        {   0.074046,   0.569346,   0.818757 },\
        {   0.185248,   0.972674,   0.139958 },\
        {   0.185248,   0.972674,   0.139958 },\
        {   0.581798,   0.167516,  -0.795895 },\
        {   0.068192,  -0.845542,  -0.529536 },\
        {   0.624762,   0.491487,   0.606724 },\
        {   0.670548,   0.657891,   0.342847 },\
        {   0.589036,   0.016898,   0.807930 },\
        {   0.081415,   0.636514,   0.766956 },\
        {   0.157527,  -0.859267,  -0.486668 },\
        {   0.560342,   0.823543,  -0.088276 },\
        {   0.613427,   0.728417,   0.305150 },\
        {   0.185248,   0.972674,   0.139958 },\
        {   0.269101,   0.798026,   0.539203 },\
        {   0.606956,   0.785781,  -0.118968 },\
        {   0.114832,   0.865751,  -0.487124 } \
    };

    int size = 40;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    bool result = open3d::EstimateNormals(pc);

    for (size_t i = 0; i < pc.normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), pc.normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), pc.normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), pc.normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, OrientNormalsToAlignWithDirection)
{
    vector<Eigen::Vector3d> ref =
    {
        {   0.301081,   0.852348,   0.427614 },\
        {   0.551697,   0.828710,  -0.094182 },\
        {   0.080430,  -0.973307,   0.214953 },\
        {   0.222569,   0.630019,   0.744002 },\
        {   0.034279,   0.427528,   0.903352 },\
        {   0.711501,   0.596113,   0.372042 },\
        {   0.518930,   0.554754,   0.650353 },\
        {  -0.491667,  -0.570274,   0.658067 },\
        {   0.324264,   0.745016,   0.582927 },\
        {  -0.120885,   0.989778,  -0.075671 },\
        {   0.371510,   0.766621,   0.523710 },\
        {  -0.874810,   0.156827,   0.458381 },\
        {  -0.240195,  -0.936743,   0.254594 },\
        {   0.518743,   0.540549,   0.662354 },\
        {  -0.240195,  -0.936743,   0.254594 },\
        {  -0.081304,   0.500530,   0.861893 },\
        {  -0.755134,   0.525094,   0.392491 },\
        {  -0.721656,  -0.540276,   0.432799 },\
        {  -0.157527,   0.859267,   0.486668 },\
        {   0.446840,   0.725678,   0.523188 },\
        {  -0.019810,   0.591008,   0.806422 },\
        {   0.027069,   0.855357,   0.517331 },\
        {   0.478109,   0.869547,  -0.123690 },\
        {  -0.101961,   0.787444,   0.607895 },\
        {   0.074046,   0.569346,   0.818757 },\
        {   0.185248,   0.972674,   0.139958 },\
        {   0.185248,   0.972674,   0.139958 },\
        {  -0.581798,  -0.167516,   0.795895 },\
        {  -0.068192,   0.845542,   0.529536 },\
        {   0.624762,   0.491487,   0.606724 },\
        {   0.670548,   0.657891,   0.342847 },\
        {   0.589036,   0.016898,   0.807930 },\
        {   0.081415,   0.636514,   0.766956 },\
        {  -0.157527,   0.859267,   0.486668 },\
        {   0.560342,   0.823543,  -0.088276 },\
        {   0.613427,   0.728417,   0.305150 },\
        {   0.185248,   0.972674,   0.139958 },\
        {   0.269101,   0.798026,   0.539203 },\
        {   0.606956,   0.785781,  -0.118968 },\
        {  -0.114832,  -0.865751,   0.487124 } \
    };

    int size = 40;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    bool result = open3d::EstimateNormals(pc);
    result = open3d::OrientNormalsToAlignWithDirection(pc, Eigen::Vector3d(1.5, 0.5, 3.3));

    for (size_t i = 0; i < pc.normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), pc.normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), pc.normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), pc.normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, OrientNormalsTowardsCameraLocation)
{
    vector<Eigen::Vector3d> ref =
    {
        {  -0.301081,  -0.852348,  -0.427614 },\
        {  -0.551697,  -0.828710,   0.094182 },\
        {   0.080430,  -0.973307,   0.214953 },\
        {  -0.222569,  -0.630019,  -0.744002 },\
        {  -0.034279,  -0.427528,  -0.903352 },\
        {  -0.711501,  -0.596113,  -0.372042 },\
        {  -0.518930,  -0.554754,  -0.650353 },\
        {   0.491667,   0.570274,  -0.658067 },\
        {  -0.324264,  -0.745016,  -0.582927 },\
        {   0.120885,  -0.989778,   0.075671 },\
        {  -0.371510,  -0.766621,  -0.523710 },\
        {   0.874810,  -0.156827,  -0.458381 },\
        {  -0.240195,  -0.936743,   0.254594 },\
        {  -0.518743,  -0.540549,  -0.662354 },\
        {  -0.240195,  -0.936743,   0.254594 },\
        {   0.081304,  -0.500530,  -0.861893 },\
        {   0.755134,  -0.525094,  -0.392491 },\
        {   0.721656,   0.540276,  -0.432799 },\
        {   0.157527,  -0.859267,  -0.486668 },\
        {  -0.446840,  -0.725678,  -0.523188 },\
        {   0.019810,  -0.591008,  -0.806422 },\
        {  -0.027069,  -0.855357,  -0.517331 },\
        {  -0.478109,  -0.869547,   0.123690 },\
        {   0.101961,  -0.787444,  -0.607895 },\
        {  -0.074046,  -0.569346,  -0.818757 },\
        {  -0.185248,  -0.972674,  -0.139958 },\
        {  -0.185248,  -0.972674,  -0.139958 },\
        {   0.581798,   0.167516,  -0.795895 },\
        {   0.068192,  -0.845542,  -0.529536 },\
        {  -0.624762,  -0.491487,  -0.606724 },\
        {  -0.670548,  -0.657891,  -0.342847 },\
        {  -0.589036,  -0.016898,  -0.807930 },\
        {  -0.081415,  -0.636514,  -0.766956 },\
        {   0.157527,  -0.859267,  -0.486668 },\
        {  -0.560342,  -0.823543,   0.088276 },\
        {  -0.613427,  -0.728417,  -0.305150 },\
        {  -0.185248,  -0.972674,  -0.139958 },\
        {  -0.269101,  -0.798026,  -0.539203 },\
        {  -0.606956,  -0.785781,   0.118968 },\
        {   0.114832,   0.865751,  -0.487124 } \
    };

    int size = 40;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    bool result = open3d::EstimateNormals(pc);
    result = open3d::OrientNormalsTowardsCameraLocation(pc, Eigen::Vector3d(1.5, 0.5, 3.3));

    for (size_t i = 0; i < pc.normals_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), pc.normals_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), pc.normals_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), pc.normals_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudToPointCloudDistance)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudMeanAndCovariance)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudMahalanobisDistance)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_ComputePointCloudNearestNeighborDistance)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromFloatDepthImage)
{
    UnitTest::NotImplemented();
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, DISABLED_CreatePointCloudFromRGBDImageT)
{
    UnitTest::NotImplemented();
}
