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
#include "Core/Geometry/Image.h"
#include <Core/Geometry/RGBDImage.h>
#include "Core/Camera/PinholeCameraIntrinsic.h"

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
TEST(PointCloud, ComputePointCloudToPointCloudDistance)
{
    vector<double> ref =
        { 155.013456,  126.672493,  114.606722,  190.747153,  133.079840,\
          121.137276,  106.805907,  226.190750,  131.745147,  172.069584,\
          247.822223,  119.390962,   21.209580,   68.624498,  136.386737,\
          149.981320,  206.445708,  191.876431,  140.127314,  131.657386,\
          183.471289,  221.094822,  178.447628,  126.081556,   29.338770,\
          111.453558,  102.236849,  304.969947,   40.823263,  227.787078,\
          169.129676,  197.146871,  167.494524,  174.795150,  142.910946,\
          263.053174,  122.803815,  238.740548,  116.243401,  180.230879,\
           91.863637,   96.241462,   24.547707,  174.705689,   65.612463,\
          148.994593,  158.758879,  345.655903,  251.182091,  182.235820 };

    int size = 100;

    open3d::PointCloud pc0;
    open3d::PointCloud pc1;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    vector<Eigen::Vector3d> points(size);
    UnitTest::Rand(points, vmin, vmax);

    for (int i = 0; i < (size / 2); i++)
    {
        pc0.points_.push_back(points[         0 + i]);
        pc1.points_.push_back(points[(size / 2) + i]);
    }

    vector<double> distance = open3d::ComputePointCloudToPointCloudDistance(pc0, pc1);

    for (size_t i = 0; i < distance.size(); i++)
    {
        EXPECT_NEAR(ref[i], distance[i], UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, ComputePointCloudMeanAndCovariance)
{
    int size = 40;
    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    auto output = open3d::ComputePointCloudMeanAndCovariance(pc);

    Eigen::Vector3d mean = get<0>(output);
    Eigen::Matrix3d covariance = get<1>(output);

    EXPECT_NEAR(516.440978, mean(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(568.728705, mean(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(528.163148, mean(2, 0), UnitTest::THRESHOLD_1E_6);

    EXPECT_NEAR( 86744.369850, covariance(0, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR( -9541.211186, covariance(1, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  1574.386634, covariance(2, 0), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR( -9541.211186, covariance(0, 1), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR( 64572.811656, covariance(1, 1), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(-12863.917238, covariance(2, 1), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(  1574.386634, covariance(0, 2), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR(-12863.917238, covariance(1, 2), UnitTest::THRESHOLD_1E_6);
    EXPECT_NEAR( 86004.409342, covariance(2, 2), UnitTest::THRESHOLD_1E_6);
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, ComputePointCloudMahalanobisDistance)
{
    vector<double> ref =
    {   1.436659,    1.874154,    1.233698,    0.433908,    1.626155,\
        1.557505,    2.025676,    1.974729,    1.844851,    2.339683,\
        1.181081,    0.962552,    1.622606,    1.164056,    1.444526,\
        1.928016,    2.291645,    2.094157,    2.203939,    2.285143,\
        2.469006,    1.520594,    1.002449,    1.632936,    2.151522,\
        1.274891,    0.897636,    1.903137,    1.708510,    1.529560,\
        1.598700,    1.728185,    1.339974,    1.775422,    2.322568,\
        1.355921,    1.232913,    1.230436,    1.565730,    1.805811,\
        2.128716,    2.336201,    1.031655,    1.396892,    1.665647,\
        1.774400,    1.936523,    2.331766,    1.916676,    1.703192,\
        1.540894,    1.915370,    2.053272,    1.280547,    2.727001,\
        1.987940,    2.557249,    2.823211,    1.817930,    1.897643,\
        0.713170,    1.359395,    0.952844,    1.989975,    1.769820,\
        1.925630,    2.234202,    2.222686,    1.274623,    2.114525,\
        1.138660,    1.330170,    1.955832,    1.211106,    1.627522,\
        1.353384,    1.504520,    1.304239,    2.049181,    1.317779,\
        0.576005,    1.367861,    1.872072,    0.877623,    1.435333,\
        0.920848,    1.809583,    1.055132,    1.873652,    1.941344,\
        1.684577,    0.832126,    1.472177,    2.413383,    1.789212,\
        2.196511,    1.612605,    1.086377,    2.320772,    1.565733 };

    int size = 100;

    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    vector<double> distance = open3d::ComputePointCloudMahalanobisDistance(pc);

    for (size_t i = 0; i < distance.size(); i++)
    {
        EXPECT_NEAR(ref[i], distance[i], UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, ComputePointCloudNearestNeighborDistance)
{
    vector<double> ref =
        { 118.596911,  126.672493,  114.606722,  161.530864,  133.079840,\
           86.236949,  106.805907,  122.721492,  131.745147,  172.069584,\
          137.205963,  119.390962,   21.209580,   68.624498,  124.098984,\
          149.981320,  206.445708,  168.898664,  140.127314,  131.657386,\
          183.471289,  174.325569,  146.496642,  126.081556,   29.338770,\
          106.015339,   81.603600,  216.543725,   40.823263,  118.596911,\
           86.236949,  149.088247,  137.205963,  173.233732,  142.910946,\
          243.349506,  111.967224,  164.785428,  116.243401,  180.230879,\
           91.863637,   96.241462,   24.547707,  164.785428,   65.612463,\
          148.994593,  151.555829,  236.691949,  122.721492,  182.235820,\
          103.653549,  106.805907,   91.863637,   68.624498,  173.031983,\
          154.862657,  285.675197,  173.031983,   65.612463,  118.049042,\
          119.390962,   99.752314,  102.804606,  186.903796,  112.083072,\
           91.232205,  136.636030,   91.232205,  129.334849,  126.672493,\
           36.610135,  131.732577,  174.705689,   36.610135,   21.209580,\
           99.752314,  133.079840,  180.230879,  154.422709,  123.711935,\
          190.747153,  114.606722,  183.672581,  198.328110,  191.396844,\
           82.694609,  148.994593,   97.743496,  147.833963,  116.523372,\
          116.243401,   82.694609,  121.453127,  154.862657,   40.823263,\
           29.338770,  112.083072,   24.547707,  161.073133,  178.447628 };

    int size = 100;

    open3d::PointCloud pc;

    Eigen::Vector3d vmin(0.0, 0.0, 0.0);
    Eigen::Vector3d vmax(1000.0, 1000.0, 1000.0);

    pc.points_.resize(size);
    UnitTest::Rand(pc.points_, vmin, vmax);

    vector<double> distance = open3d::ComputePointCloudNearestNeighborDistance(pc);

    for (size_t i = 0; i < distance.size(); i++)
    {
        EXPECT_NEAR(ref[i], distance[i], UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
//
// ----------------------------------------------------------------------------
TEST(PointCloud, CreatePointCloudFromDepthImage)
{
    vector<Eigen::Vector3d> ref =
    {
        { -15.709662, -11.776101,  25.813999 },\
        { -31.647980, -23.798088,  52.167000 },\
        {  -7.881257,  -5.945074,  13.032000 },\
        { -30.145872, -22.811805,  50.005001 },\
        { -21.734044, -16.498585,  36.166000 },\
        { -25.000724, -18.662512,  41.081001 },\
        { -20.246287, -15.160878,  33.373001 },\
        { -36.219190, -27.207171,  59.889999 },\
        { -28.185984, -21.239675,  46.754002 },\
        { -23.713580, -17.926114,  39.459999 },\
        {  -9.505886,  -7.066190,  15.620000 },\
        { -31.858493, -23.756333,  52.514000 },\
        { -15.815128, -11.830214,  26.150999 },\
        {  -4.186843,  -3.141786,   6.945000 },\
        {  -8.614051,  -6.484428,  14.334000 },\
        { -33.263298, -24.622128,  54.658001 },\
        { -11.742641,  -8.719418,  19.356001 },\
        { -20.688904, -15.410790,  34.209999 },\
        { -38.349551, -28.656141,  63.612999 },\
        { -30.197857, -22.636429,  50.250000 },\
        { -30.617229, -22.567629,  50.310001 },\
        { -35.316494, -26.113137,  58.214001 },\
        { -13.822439, -10.252549,  22.856001 },\
        { -36.237141, -26.963181,  60.109001 },\
        { -37.240419, -27.797524,  61.969002 } \
    };

    open3d::Image image;

    // test image dimensions
    const int local_width = 5;
    const int local_height = 5;
    const int local_num_of_channels = 1;
    const int local_bytes_per_channel = 2;

    image.PrepareImage(local_width,
                       local_height,
                       local_num_of_channels,
                       local_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 0, 255);

    open3d::PinholeCameraIntrinsic intrinsic =
        open3d::PinholeCameraIntrinsic(
            open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto output_pc = open3d::CreatePointCloudFromDepthImage(image, intrinsic);

    for (size_t i = 0; i < output_pc->points_.size(); i++)
    {
        EXPECT_NEAR(ref[i](0, 0), output_pc->points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](1, 0), output_pc->points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref[i](2, 0), output_pc->points_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
// Test CreatePointCloudFromRGBDImage for the following configurations:
// index | color_num_of_channels | color_bytes_per_channel
//     1 |          3            |            1
//     0 |          1            |            4
// ----------------------------------------------------------------------------
void TEST_CreatePointCloudFromRGBDImage(
    const int& color_num_of_channels,
    const int& color_bytes_per_channel,
    const vector<Eigen::Vector3d>& ref_points,
    const vector<Eigen::Vector3d>& ref_colors)
{
    open3d::Image image;
    open3d::Image color;

    const int size = 5;

    // test image dimensions
    const int image_width = size;
    const int image_height = size;
    const int image_num_of_channels = 1;
    const int image_bytes_per_channel = 1;

    const int color_width = size;
    const int color_height = size;

    image.PrepareImage(image_width,
                       image_height,
                       image_num_of_channels,
                       image_bytes_per_channel);

    color.PrepareImage(color_width,
                       color_height,
                       color_num_of_channels,
                       color_bytes_per_channel);

    UnitTest::Rand<uint8_t>(image.data_, 100, 150);
    UnitTest::Rand<uint8_t>(color.data_, 130, 200);

    auto depth = open3d::ConvertDepthToFloatImage(image);

    open3d::RGBDImage rgbd_image(color, *depth);

    open3d::PinholeCameraIntrinsic intrinsic =
        open3d::PinholeCameraIntrinsic(
            open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    auto output_pc = open3d::CreatePointCloudFromRGBDImage(rgbd_image, intrinsic);

    for (size_t i = 0; i < output_pc->points_.size(); i++)
    {
        EXPECT_NEAR(ref_points[i](0, 0), output_pc->points_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](1, 0), output_pc->points_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_points[i](2, 0), output_pc->points_[i](2, 0), UnitTest::THRESHOLD_1E_6);

        EXPECT_NEAR(ref_colors[i](0, 0), output_pc->colors_[i](0, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_colors[i](1, 0), output_pc->colors_[i](1, 0), UnitTest::THRESHOLD_1E_6);
        EXPECT_NEAR(ref_colors[i](2, 0), output_pc->colors_[i](2, 0), UnitTest::THRESHOLD_1E_6);
    }
}

// ----------------------------------------------------------------------------
// Test CreatePointCloudFromRGBDImage for the following configuration:
// color_num_of_channels = 3
// color_bytes_per_channel = 1
// ----------------------------------------------------------------------------
TEST(PointCloud, CreatePointCloudFromRGBDImage_3_1)
{
    vector<Eigen::Vector3d> ref_points =
    {
        {  -0.000339,  -0.000254,   0.000557 },\
        {  -0.000283,  -0.000213,   0.000467 },\
        {  -0.000330,  -0.000249,   0.000545 },\
        {  -0.000329,  -0.000249,   0.000545 },\
        {  -0.000342,  -0.000259,   0.000569 },\
        {  -0.000260,  -0.000194,   0.000427 },\
        {  -0.000276,  -0.000207,   0.000455 },\
        {  -0.000327,  -0.000246,   0.000541 },\
        {  -0.000267,  -0.000201,   0.000443 },\
        {  -0.000299,  -0.000226,   0.000498 },\
        {  -0.000294,  -0.000218,   0.000482 },\
        {  -0.000312,  -0.000232,   0.000514 },\
        {  -0.000280,  -0.000209,   0.000463 },\
        {  -0.000296,  -0.000222,   0.000490 },\
        {  -0.000346,  -0.000261,   0.000576 },\
        {  -0.000346,  -0.000256,   0.000569 },\
        {  -0.000312,  -0.000231,   0.000514 },\
        {  -0.000320,  -0.000238,   0.000529 },\
        {  -0.000253,  -0.000189,   0.000420 },\
        {  -0.000306,  -0.000230,   0.000510 },\
        {  -0.000239,  -0.000176,   0.000392 },\
        {  -0.000266,  -0.000197,   0.000439 },\
        {  -0.000251,  -0.000186,   0.000416 },\
        {  -0.000331,  -0.000246,   0.000549 },\
        {  -0.000252,  -0.000188,   0.000420 } \
    };

    vector<Eigen::Vector3d> ref_colors =
    {
        {   0.737255,   0.615686,   0.721569 },\
        {   0.725490,   0.756863,   0.560784 },\
        {   0.600000,   0.717647,   0.584314 },\
        {   0.658824,   0.639216,   0.682353 },\
        {   0.607843,   0.647059,   0.768627 },\
        {   0.760784,   0.682353,   0.705882 },\
        {   0.545098,   0.674510,   0.513725 },\
        {   0.576471,   0.545098,   0.729412 },\
        {   0.549020,   0.619608,   0.545098 },\
        {   0.537255,   0.780392,   0.568627 },\
        {   0.647059,   0.737255,   0.674510 },\
        {   0.588235,   0.682353,   0.650980 },\
        {   0.643137,   0.776471,   0.588235 },\
        {   0.717647,   0.650980,   0.717647 },\
        {   0.619608,   0.752941,   0.584314 },\
        {   0.603922,   0.729412,   0.760784 },\
        {   0.525490,   0.768627,   0.650980 },\
        {   0.533333,   0.560784,   0.690196 },\
        {   0.752941,   0.603922,   0.525490 },\
        {   0.513725,   0.635294,   0.525490 },\
        {   0.572549,   0.772549,   0.756863 },\
        {   0.741176,   0.580392,   0.654902 },\
        {   0.611765,   0.717647,   0.647059 },\
        {   0.690196,   0.654902,   0.517647 },\
        {   0.627451,   0.764706,   0.764706 } \
    };
    const int color_num_of_channels = 3;
    const int color_bytes_per_channel = 1;

    TEST_CreatePointCloudFromRGBDImage(
        color_num_of_channels,
        color_bytes_per_channel,
        ref_points,
        ref_colors);
}

// ----------------------------------------------------------------------------
// Test CreatePointCloudFromRGBDImage for the following configuration:
// color_num_of_channels = 1
// color_bytes_per_channel = 4
// ----------------------------------------------------------------------------
TEST(PointCloud, CreatePointCloudFromRGBDImage_1_4)
{
    vector<Eigen::Vector3d> ref_points =
    {
        {  -0.000339,  -0.000254,   0.000557 },\
        {  -0.000283,  -0.000213,   0.000467 },\
        {  -0.000330,  -0.000249,   0.000545 },\
        {  -0.000329,  -0.000249,   0.000545 },\
        {  -0.000342,  -0.000259,   0.000569 },\
        {  -0.000260,  -0.000194,   0.000427 },\
        {  -0.000276,  -0.000207,   0.000455 },\
        {  -0.000327,  -0.000246,   0.000541 },\
        {  -0.000267,  -0.000201,   0.000443 },\
        {  -0.000299,  -0.000226,   0.000498 },\
        {  -0.000294,  -0.000218,   0.000482 },\
        {  -0.000312,  -0.000232,   0.000514 },\
        {  -0.000280,  -0.000209,   0.000463 },\
        {  -0.000296,  -0.000222,   0.000490 },\
        {  -0.000346,  -0.000261,   0.000576 },\
        {  -0.000346,  -0.000256,   0.000569 },\
        {  -0.000312,  -0.000231,   0.000514 },\
        {  -0.000320,  -0.000238,   0.000529 },\
        {  -0.000253,  -0.000189,   0.000420 },\
        {  -0.000306,  -0.000230,   0.000510 },\
        {  -0.000239,  -0.000176,   0.000392 },\
        {  -0.000266,  -0.000197,   0.000439 },\
        {  -0.000251,  -0.000186,   0.000416 },\
        {  -0.000331,  -0.000246,   0.000549 },\
        {  -0.000252,  -0.000188,   0.000420 } \
    };

    vector<Eigen::Vector3d> ref_colors =
    {
        {  -0.000352,  -0.000352,  -0.000352 },\
        {  -0.000018,  -0.000018,  -0.000018 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        { -98.323448, -98.323448, -98.323448 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.001065,  -0.001065,  -0.001065 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.020211,  -0.020211,  -0.020211 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.000018,  -0.000018,  -0.000018 },\
        {  -4.959918,  -4.959918,  -4.959918 },\
        { -93.301918, -93.301918, -93.301918 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.094615,  -0.094615,  -0.094615 },\
        {  -0.000019,  -0.000019,  -0.000019 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        {  -1.254324,  -1.254324,  -1.254324 },\
        {  -4.581266,  -4.581266,  -4.581266 },\
        {  -0.000000,  -0.000000,  -0.000000 },\
        { -80.372437, -80.372437, -80.372437 },\
        { -22.216608, -22.216608, -22.216608 } \
    };

    const int color_num_of_channels = 1;
    const int color_bytes_per_channel = 4;

    TEST_CreatePointCloudFromRGBDImage(
        color_num_of_channels,
        color_bytes_per_channel,
        ref_points,
        ref_colors);
}
