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

#include "open3d/geometry/PointCloud.h"

#include <algorithm>

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/RGBDImage.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/PinholeCameraTrajectoryIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(PointCloud, ConstructorDefault) {
    geometry::PointCloud pcd;

    EXPECT_EQ(geometry::Geometry::GeometryType::PointCloud,
              pcd.GetGeometryType());
    EXPECT_EQ(pcd.Dimension(), 3);

    EXPECT_EQ(pcd.points_.size(), 0);
    EXPECT_EQ(pcd.normals_.size(), 0);
    EXPECT_EQ(pcd.colors_.size(), 0);
}

TEST(PointCloud, ConstructorFromPoints) {
    std::vector<Eigen::Vector3d> points = {{0, 1, 2}, {3, 4, 5}};
    geometry::PointCloud pcd(points);

    EXPECT_EQ(pcd.points_.size(), 2);
    EXPECT_EQ(pcd.normals_.size(), 0);
    EXPECT_EQ(pcd.colors_.size(), 0);

    ExpectEQ(pcd.points_, points);
}

TEST(PointCloud, Clear_IsEmpty) {
    std::vector<Eigen::Vector3d> points = {{0, 1, 2}, {3, 4, 5}};
    std::vector<Eigen::Vector3d> normals = {{0, 1, 2}, {3, 4, 5}};
    std::vector<Eigen::Vector3d> colors = {{0.0, 0.1, 0.2}, {0.3, 0.4, 0.5}};

    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    pcd.colors_ = colors;

    EXPECT_FALSE(pcd.IsEmpty());
    EXPECT_EQ(pcd.points_.size(), 2);
    EXPECT_EQ(pcd.normals_.size(), 2);
    EXPECT_EQ(pcd.colors_.size(), 2);

    pcd.Clear();
    EXPECT_TRUE(pcd.IsEmpty());
    EXPECT_EQ(pcd.points_.size(), 0);
    EXPECT_EQ(pcd.normals_.size(), 0);
    EXPECT_EQ(pcd.colors_.size(), 0);
}

TEST(PointCloud, GetMinBound) {
    geometry::PointCloud pcd({{1, 10, 20}, {30, 2, 40}, {50, 60, 3}});
    ExpectEQ(pcd.GetMinBound(), Eigen::Vector3d(1, 2, 3));

    geometry::PointCloud pc_empty;
    ExpectEQ(pc_empty.GetMinBound(), Eigen::Vector3d(0, 0, 0));
}

TEST(PointCloud, GetMaxBound) {
    geometry::PointCloud pcd({{1, 10, 20}, {30, 2, 40}, {50, 60, 3}});
    ExpectEQ(pcd.GetMaxBound(), Eigen::Vector3d(50, 60, 40));

    geometry::PointCloud pc_empty;
    ExpectEQ(pc_empty.GetMaxBound(), Eigen::Vector3d(0, 0, 0));
}

TEST(PointCloud, GetCenter) {
    geometry::PointCloud pcd({{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}});
    ExpectEQ(pcd.GetCenter(), Eigen::Vector3d(4.5, 5.5, 6.5));

    geometry::PointCloud pc_empty;
    ExpectEQ(pc_empty.GetCenter(), Eigen::Vector3d(0, 0, 0));
}

TEST(PointCloud, GetAxisAlignedBoundingBox) {
    geometry::PointCloud pcd;
    geometry::AxisAlignedBoundingBox aabb;

    pcd = geometry::PointCloud();
    aabb = pcd.GetAxisAlignedBoundingBox();
    EXPECT_EQ(aabb.min_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.max_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.color_, Eigen::Vector3d(0, 0, 0));

    pcd = geometry::PointCloud({{0, 0, 0}});
    aabb = pcd.GetAxisAlignedBoundingBox();
    EXPECT_EQ(aabb.min_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.max_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.color_, Eigen::Vector3d(0, 0, 0));

    pcd = geometry::PointCloud({{0, 2, 0}, {1, 1, 2}, {1, 0, 3}});
    aabb = pcd.GetAxisAlignedBoundingBox();
    EXPECT_EQ(aabb.min_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.max_bound_, Eigen::Vector3d(1, 2, 3));
    EXPECT_EQ(aabb.color_, Eigen::Vector3d(0, 0, 0));

    pcd = geometry::PointCloud({{0, 0, 0}, {1, 2, 3}});
    aabb = pcd.GetAxisAlignedBoundingBox();
    EXPECT_EQ(aabb.min_bound_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(aabb.max_bound_, Eigen::Vector3d(1, 2, 3));
    EXPECT_EQ(aabb.color_, Eigen::Vector3d(0, 0, 0));
}

TEST(PointCloud, GetOrientedBoundingBox) {
    geometry::PointCloud pcd;
    geometry::OrientedBoundingBox obb;

    // Emtpy (GetOrientedBoundingBox requires >=4 points)
    pcd = geometry::PointCloud();
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());

    // Point
    pcd = geometry::PointCloud({{0, 0, 0}});
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());
    pcd = geometry::PointCloud({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());

    // Line
    pcd = geometry::PointCloud({{0, 0, 0}, {1, 1, 1}});
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());
    pcd = geometry::PointCloud({{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());

    // Plane
    pcd = geometry::PointCloud({{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}});
    EXPECT_ANY_THROW(pcd.GetOrientedBoundingBox());

    // Valid 4 points
    pcd = geometry::PointCloud({{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}});
    pcd.GetOrientedBoundingBox();

    // 8 points with known ground truth
    pcd = geometry::PointCloud({{0, 0, 0},
                                {0, 0, 1},
                                {0, 2, 0},
                                {0, 2, 1},
                                {3, 0, 0},
                                {3, 0, 1},
                                {3, 2, 0},
                                {3, 2, 1}});
    obb = pcd.GetOrientedBoundingBox();
    EXPECT_EQ(obb.center_, Eigen::Vector3d(1.5, 1, 0.5));
    EXPECT_EQ(obb.extent_, Eigen::Vector3d(3, 2, 1));
    EXPECT_EQ(obb.color_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(obb.R_, Eigen::Matrix3d::Identity());
    ExpectEQ(Sort(obb.GetBoxPoints()),
             Sort(std::vector<Eigen::Vector3d>({{0, 0, 0},
                                                {0, 0, 1},
                                                {0, 2, 0},
                                                {0, 2, 1},
                                                {3, 0, 0},
                                                {3, 0, 1},
                                                {3, 2, 0},
                                                {3, 2, 1}})));
}

TEST(PointCloud, Transform) {
    std::vector<Eigen::Vector3d> points = {
            {0, 0, 0},
            {1, 2, 4},
    };
    std::vector<Eigen::Vector3d> normals = {
            {4, 2, 1},
            {0, 0, 0},
    };
    Eigen::Matrix4d transformation;
    transformation << 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            5.5, 6.0, 6.5, 7.0, 7.5;

    std::vector<Eigen::Vector3d> points_transformed = {
            {0.20000, 0.46666, 0.73333},
            {0.11926, 0.41284, 0.70642},
    };
    std::vector<Eigen::Vector3d> normals_transformed = {
            {2, 16, 30},
            {0, 0, 0},
    };

    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    pcd.Transform(transformation);
    ExpectEQ(pcd.points_, points_transformed, 1e-4);
    ExpectEQ(pcd.normals_, normals_transformed, 1e-4);
}

TEST(PointCloud, Translate) {
    std::vector<Eigen::Vector3d> points = {
            {0, 1, 2},
            {6, 7, 8},
    };
    Eigen::Vector3d translation(10, 20, 30);
    std::vector<Eigen::Vector3d> points_translated = {
            {10, 21, 32},
            {16, 27, 38},
    };
    std::vector<Eigen::Vector3d> points_translated_non_relative = {
            {7, 17, 27},
            {13, 23, 33},
    };

    // Relative: direct translation.
    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.Translate(translation);
    ExpectEQ(pcd.points_, points_translated);

    // Non-relative: new center is the translation.
    pcd.points_ = points;
    pcd.Translate(translation, /*relative=*/false);
    ExpectEQ(pcd.points_, points_translated_non_relative);
    ExpectEQ(pcd.GetCenter(), translation);
}

TEST(PointCloud, Scale) {
    std::vector<Eigen::Vector3d> points = {{0, 1, 2}, {6, 7, 8}};
    double scale = 10;
    geometry::PointCloud pcd;

    pcd.points_ = points;
    pcd.Scale(scale, Eigen::Vector3d(0, 0, 0));
    ExpectEQ(pcd.points_,
             std::vector<Eigen::Vector3d>({{0, 10, 20}, {60, 70, 80}}));

    pcd.points_ = points;
    pcd.Scale(scale, Eigen::Vector3d(1, 1, 1));
    ExpectEQ(pcd.points_,
             std::vector<Eigen::Vector3d>({{-9, 1, 11}, {51, 61, 71}}));
}

TEST(PointCloud, Rotate) {
    std::vector<Eigen::Vector3d> points = {{0, 1, 2}, {3, 4, 5}};
    std::vector<Eigen::Vector3d> normals = {{5, 4, 3}, {2, 1, 0}};
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(0.25 * M_PI, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.33 * M_PI, Eigen::Vector3d::UnitZ());

    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    Eigen::Vector3d center = pcd.GetCenter();
    pcd.Rotate(R, center);

    ExpectEQ(pcd.points_,
             std::vector<Eigen::Vector3d>(
                     {{0, 1.42016, 1.67409}, {3, 3.57984, 5.32591}}));
    ExpectEQ(pcd.normals_,
             std::vector<Eigen::Vector3d>(
                     {{3, 3.84816, 5.11778}, {0, 1.688476, 1.465963}}));
    ExpectEQ(pcd.GetCenter(),
             center);  // Rotate relative to the original center
}

TEST(PointCloud, OperatorPlusEqual) {
    std::vector<Eigen::Vector3d> points_a = {{0, 1, 2}, {3, 4, 5}};
    std::vector<Eigen::Vector3d> normals_a = {{0, 1, 2}, {3, 4, 5}};
    std::vector<Eigen::Vector3d> colors_a = {{0, 1, 2}, {3, 4, 5}};

    std::vector<Eigen::Vector3d> points_b = {{6, 7, 8}, {9, 10, 11}};
    std::vector<Eigen::Vector3d> normals_b = {{6, 7, 8}, {9, 10, 11}};
    std::vector<Eigen::Vector3d> colors_b = {{6, 7, 8}, {9, 10, 11}};

    std::vector<Eigen::Vector3d> empty(0);

    std::vector<Eigen::Vector3d> points_a_b = {
            {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};
    std::vector<Eigen::Vector3d> normals_a_b = {
            {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};
    std::vector<Eigen::Vector3d> colors_a_b = {
            {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}};

    geometry::PointCloud pc_a_full;
    geometry::PointCloud pc_b_full;
    pc_a_full.points_ = points_a;
    pc_a_full.normals_ = normals_a;
    pc_a_full.colors_ = colors_a;
    pc_b_full.points_ = points_b;
    pc_b_full.normals_ = normals_b;
    pc_b_full.colors_ = colors_b;

    geometry::PointCloud pc_a = pc_a_full;
    geometry::PointCloud pc_b = pc_b_full;
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, points_a_b);
    ExpectEQ(pc_a.normals_, normals_a_b);
    ExpectEQ(pc_a.colors_, colors_a_b);

    pc_a = pc_a_full;
    pc_b = pc_b_full;
    pc_a.normals_.clear();
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, points_a_b);
    ExpectEQ(pc_a.normals_, empty);
    ExpectEQ(pc_a.colors_, colors_a_b);

    pc_a = pc_a_full;
    pc_b = pc_b_full;
    pc_b.normals_.clear();
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, points_a_b);
    ExpectEQ(pc_a.normals_, empty);
    ExpectEQ(pc_a.colors_, colors_a_b);

    pc_a = pc_a_full;
    pc_b = pc_b_full;
    pc_a.colors_.clear();
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, points_a_b);
    ExpectEQ(pc_a.normals_, normals_a_b);
    ExpectEQ(pc_a.colors_, empty);

    pc_a = pc_a_full;
    pc_b = pc_b_full;
    pc_b.colors_.clear();
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, points_a_b);
    ExpectEQ(pc_a.normals_, normals_a_b);
    ExpectEQ(pc_a.colors_, empty);

    pc_a.Clear();
    pc_a += pc_b_full;
    ExpectEQ(pc_a.points_, pc_b_full.points_);
    ExpectEQ(pc_a.normals_, pc_b_full.normals_);
    ExpectEQ(pc_a.colors_, pc_b_full.colors_);

    pc_a = pc_a_full;
    pc_b.Clear();
    pc_a += pc_b;
    ExpectEQ(pc_a.points_, pc_a_full.points_);
    ExpectEQ(pc_a.normals_, pc_a_full.normals_);
    ExpectEQ(pc_a.colors_, pc_a_full.colors_);
}

TEST(PointCloud, OperatorPlus) {
    std::vector<Eigen::Vector3d> points_a = {{0, 1, 2}};
    std::vector<Eigen::Vector3d> normals_a = {{0, 1, 2}};
    std::vector<Eigen::Vector3d> colors_a = {{0, 1, 2}};
    std::vector<Eigen::Vector3d> points_b = {{3, 4, 5}};
    std::vector<Eigen::Vector3d> normals_b = {{3, 4, 5}};
    std::vector<Eigen::Vector3d> colors_b = {{3, 4, 5}};

    geometry::PointCloud pc_a;
    geometry::PointCloud pc_b;
    pc_a.points_ = points_a;
    pc_a.normals_ = normals_a;
    pc_a.colors_ = colors_a;
    pc_b.points_ = points_b;
    pc_b.normals_ = normals_b;
    pc_b.colors_ = colors_b;

    geometry::PointCloud pc_c = pc_a + pc_b;
    ExpectEQ(pc_c.points_,
             std::vector<Eigen::Vector3d>({{0, 1, 2}, {3, 4, 5}}));
    ExpectEQ(pc_c.normals_,
             std::vector<Eigen::Vector3d>({{0, 1, 2}, {3, 4, 5}}));
    ExpectEQ(pc_c.colors_,
             std::vector<Eigen::Vector3d>({{0, 1, 2}, {3, 4, 5}}));
}

TEST(PointCloud, HasPoints) {
    geometry::PointCloud pcd;
    EXPECT_FALSE(pcd.HasPoints());
    pcd.points_.resize(5);
    EXPECT_TRUE(pcd.HasPoints());
}

TEST(PointCloud, HasNormals) {
    geometry::PointCloud pcd;
    EXPECT_FALSE(pcd.HasNormals());

    // False if #points == 0
    pcd.points_.resize(0);
    pcd.normals_.resize(5);
    EXPECT_FALSE(pcd.HasNormals());

    // False if not consistant
    pcd.points_.resize(4);
    pcd.normals_.resize(5);
    EXPECT_FALSE(pcd.HasNormals());

    // True if non-zero and consistant
    pcd.points_.resize(5);
    pcd.normals_.resize(5);
    EXPECT_TRUE(pcd.HasNormals());
}

TEST(PointCloud, HasColors) {
    geometry::PointCloud pcd;
    EXPECT_FALSE(pcd.HasNormals());

    // False if #points == 0
    pcd.points_.resize(0);
    pcd.colors_.resize(5);
    EXPECT_FALSE(pcd.HasColors());

    // False if not consistant
    pcd.points_.resize(4);
    pcd.colors_.resize(5);
    EXPECT_FALSE(pcd.HasColors());

    // True if non-zero and consistant
    pcd.points_.resize(5);
    pcd.colors_.resize(5);
    EXPECT_TRUE(pcd.HasColors());
}

TEST(PointCloud, NormalizeNormals) {
    geometry::PointCloud pcd;
    pcd.normals_ = {{2, 2, 2}, {1, 1, 1}, {-1, -1, -1}, {0, 0, 1},
                    {0, 1, 0}, {1, 0, 0}, {0, 0, 0}};
    pcd.NormalizeNormals();
    ExpectEQ(pcd.normals_, std::vector<Eigen::Vector3d>({
                                   {0.57735, 0.57735, 0.57735},
                                   {0.57735, 0.57735, 0.57735},
                                   {-0.57735, -0.57735, -0.57735},
                                   {0, 0, 1},
                                   {0, 1, 0},
                                   {1, 0, 0},
                                   {0, 0, 0},
                           }));
}

TEST(PointCloud, PaintUniformColor) {
    geometry::PointCloud pcd;
    pcd.points_.resize(2);
    EXPECT_EQ(pcd.points_.size(), 2);
    EXPECT_EQ(pcd.colors_.size(), 0);

    Eigen::Vector3d color(0.125, 0.25, 0.5);
    pcd.PaintUniformColor(color);
    EXPECT_EQ(pcd.colors_.size(), 2);

    EXPECT_EQ(pcd.colors_, std::vector<Eigen::Vector3d>({color, color}));
}

TEST(PointCloud, SelectByIndex) {
    std::vector<Eigen::Vector3d> points({
            {0, 0, 0},
            {1, 1, 1},
            {2, 2, 2},
            {3, 3, 3},
    });
    std::vector<Eigen::Vector3d> colors({
            {0.0, 0.0, 0.0},
            {0.1, 0.1, 0.1},
            {0.2, 0.2, 0.2},
            {0.3, 0.3, 0.3},
    });
    std::vector<Eigen::Vector3d> normals({
            {10, 10, 10},
            {11, 11, 11},
            {12, 12, 12},
            {13, 13, 13},
    });

    std::vector<size_t> indices{0, 1, 3};

    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.colors_ = colors;
    pcd.normals_ = normals;

    std::shared_ptr<geometry::PointCloud> pc0 = pcd.SelectByIndex(indices);
    ExpectEQ(pc0->points_, std::vector<Eigen::Vector3d>({
                                   {0, 0, 0},
                                   {1, 1, 1},
                                   {3, 3, 3},
                           }));
    ExpectEQ(pc0->colors_, std::vector<Eigen::Vector3d>({
                                   {0.0, 0.0, 0.0},
                                   {0.1, 0.1, 0.1},
                                   {0.3, 0.3, 0.3},
                           }));
    ExpectEQ(pc0->normals_, std::vector<Eigen::Vector3d>({
                                    {10, 10, 10},
                                    {11, 11, 11},
                                    {13, 13, 13},
                            }));

    std::shared_ptr<geometry::PointCloud> pc1 =
            pcd.SelectByIndex(indices, /*invert=*/true);
    ExpectEQ(pc1->points_, std::vector<Eigen::Vector3d>({
                                   {2, 2, 2},
                           }));
    ExpectEQ(pc1->colors_, std::vector<Eigen::Vector3d>({
                                   {0.2, 0.2, 0.2},
                           }));
    ExpectEQ(pc1->normals_, std::vector<Eigen::Vector3d>({
                                    {12, 12, 12},
                            }));
}

TEST(PointCloud, VoxelDownSample) {
    // voxel_size: 1
    // points_min_bound: (0.5, 0.5, 0.5)
    // voxel_min_bound: (0, 0, 0)
    // points coordinates: range from [0.5, 2.5)
    // voxel_{i,j,k}: 0 <= i, j, k <= 2; 27 possible voxels
    std::vector<Eigen::Vector3d> points{
            // voxel_{0, 0, 0}
            {0.5, 0.7, 0.6},
            {0.6, 0.5, 0.7},
            {0.7, 0.6, 0.5},
            {0.8, 0.8, 0.8},
            // voxel_{0, 1, 2}
            {0.5, 1.6, 2.4},
            {0.6, 1.5, 2.3},
    };
    std::vector<Eigen::Vector3d> normals{
            // voxel_{0, 0, 0}
            {0, 0, 1},
            {0, 2, 3},
            {0, 4, 5},
            {0, 6, 7},
            // voxel_{0, 1, 2}
            {1, 0, 1},
            {1, 2, 3},
    };
    std::vector<Eigen::Vector3d> colors{
            // voxel_{0, 0, 0}
            {0.0, 0.0, 0.1},
            {0.0, 0.2, 0.3},
            {0.0, 0.4, 0.5},
            {0.0, 0.6, 0.7},
            // voxel_{0, 1, 2}
            {0.1, 0.0, 0.1},
            {0.1, 0.2, 0.3},
    };
    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    pcd.colors_ = colors;

    // Ground-truth reference
    std::vector<Eigen::Vector3d> points_down{
            {0.65, 0.65, 0.65},
            {0.55, 1.55, 2.35},
    };
    std::vector<Eigen::Vector3d> normals_down{
            {0, 3, 4},
            {1, 1, 2},
    };
    std::vector<Eigen::Vector3d> colors_down{
            {0.0, 0.3, 0.4},
            {0.1, 0.1, 0.2},
    };

    std::shared_ptr<geometry::PointCloud> pc_down = pcd.VoxelDownSample(1.0);
    std::vector<size_t> sort_indices =
            GetIndicesAToB(pc_down->points_, points_down);

    ExpectEQ(ApplyIndices(pc_down->points_, sort_indices), points_down);
    ExpectEQ(ApplyIndices(pc_down->normals_, sort_indices), normals_down);
    ExpectEQ(ApplyIndices(pc_down->colors_, sort_indices), colors_down);
}

TEST(PointCloud, UniformDownSample) {
    std::vector<Eigen::Vector3d> points({
            {0, 0, 0},
            {1, 0, 0},
            {2, 0, 0},
            {3, 0, 0},
            {4, 0, 0},
            {5, 0, 0},
            {6, 0, 0},
            {7, 0, 0},
    });
    std::vector<Eigen::Vector3d> normals({
            {0, 0, 0},
            {0, 1, 0},
            {0, 2, 0},
            {0, 3, 0},
            {0, 4, 0},
            {0, 5, 0},
            {0, 6, 0},
            {0, 7, 0},
    });
    std::vector<Eigen::Vector3d> colors({
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 0.1},
            {0.0, 0.0, 0.2},
            {0.0, 0.0, 0.3},
            {0.0, 0.0, 0.4},
            {0.0, 0.0, 0.5},
            {0.0, 0.0, 0.6},
            {0.0, 0.0, 0.7},
    });
    std::vector<Eigen::Vector3d> colors_gradients({
            {0.0, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0333333, 0.0, 0.0},
            {0.0, 0.0, 0.0},
    });

    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    pcd.colors_ = colors;
    pcd.color_gradients_ = colors_gradients;

    std::shared_ptr<geometry::PointCloud> pc_down = pcd.UniformDownSample(3);
    ExpectEQ(pc_down->points_, std::vector<Eigen::Vector3d>({
                                       {0, 0, 0},
                                       {3, 0, 0},
                                       {6, 0, 0},

                               }));
    ExpectEQ(pc_down->normals_, std::vector<Eigen::Vector3d>({
                                        {0, 0, 0},
                                        {0, 3, 0},
                                        {0, 6, 0},
                                }));
    ExpectEQ(pc_down->colors_, std::vector<Eigen::Vector3d>({
                                       {0.0, 0.0, 0.0},
                                       {0.0, 0.0, 0.3},
                                       {0.0, 0.0, 0.6},
                               }));
    ExpectEQ(pc_down->color_gradients_, std::vector<Eigen::Vector3d>({
                                                {0.0, 0.0, 0.0},
                                                {0.0333333, 0.0, 0.0},
                                                {0.0333333, 0.0, 0.0},
                                        }));
}

TEST(PointCloud, Crop_AxisAlignedBoundingBox) {
    geometry::AxisAlignedBoundingBox aabb({0, 0, 0}, {2, 2, 2});
    geometry::PointCloud pcd({{0, 0, 0},
                              {2, 2, 2},
                              {1, 1, 1},
                              {1, 1, 2},
                              {3, 1, 1},
                              {-1, 1, 1}});
    pcd.normals_ = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                    {3, 0, 0}, {4, 0, 0}, {5, 0, 0}};
    pcd.colors_ = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.2, 0.0, 0.0},
                   {0.3, 0.0, 0.0}, {0.4, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    std::shared_ptr<geometry::PointCloud> pc_crop = pcd.Crop(aabb);
    ExpectEQ(pc_crop->points_, std::vector<Eigen::Vector3d>({
                                       {0, 0, 0},
                                       {2, 2, 2},
                                       {1, 1, 1},
                                       {1, 1, 2},
                               }));
    ExpectEQ(pc_crop->normals_, std::vector<Eigen::Vector3d>({
                                        {0, 0, 0},
                                        {1, 0, 0},
                                        {2, 0, 0},
                                        {3, 0, 0},
                                }));
    ExpectEQ(pc_crop->colors_, std::vector<Eigen::Vector3d>({
                                       {0.0, 0.0, 0.0},
                                       {0.1, 0.0, 0.0},
                                       {0.2, 0.0, 0.0},
                                       {0.3, 0.0, 0.0},
                               }));
}

TEST(PointCloud, Crop_OrientedBoundingBox) {
    geometry::OrientedBoundingBox obb(Eigen::Vector3d{1, 1, 1},
                                      Eigen::Matrix3d::Identity(),
                                      Eigen::Vector3d{2, 2, 2});
    geometry::PointCloud pcd({
            {0, 0, 0},
            {2, 2, 2},
            {1, 1, 1},
            {1, 1, 2},
            {3, 1, 1},
            {-1, 1, 1},
    });
    pcd.normals_ = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0},
                    {3, 0, 0}, {4, 0, 0}, {5, 0, 0}};
    pcd.colors_ = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {0.2, 0.0, 0.0},
                   {0.3, 0.0, 0.0}, {0.4, 0.0, 0.0}, {0.5, 0.0, 0.0}};

    std::shared_ptr<geometry::PointCloud> pc_crop = pcd.Crop(obb);
    ExpectEQ(pc_crop->points_, std::vector<Eigen::Vector3d>({
                                       {0, 0, 0},
                                       {2, 2, 2},
                                       {1, 1, 1},
                                       {1, 1, 2},
                               }));
    ExpectEQ(pc_crop->normals_, std::vector<Eigen::Vector3d>({
                                        {0, 0, 0},
                                        {1, 0, 0},
                                        {2, 0, 0},
                                        {3, 0, 0},
                                }));
    ExpectEQ(pc_crop->colors_, std::vector<Eigen::Vector3d>({
                                       {0.0, 0.0, 0.0},
                                       {0.1, 0.0, 0.0},
                                       {0.2, 0.0, 0.0},
                                       {0.3, 0.0, 0.0},
                               }));
}

TEST(PointCloud, EstimateNormals) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1},
    });
    pcd.EstimateNormals(geometry::KDTreeSearchParamKNN(/*knn=*/4));
    pcd.NormalizeNormals();
    double v = 1.0 / std::sqrt(3.0);
    ExpectEQ(pcd.normals_, std::vector<Eigen::Vector3d>({{v, v, v},
                                                         {-v, -v, v},
                                                         {v, -v, v},
                                                         {-v, v, v},
                                                         {-v, v, v},
                                                         {v, -v, v},
                                                         {-v, -v, v},
                                                         {v, v, v}}));
}

TEST(PointCloud, EstimateColorGradients) {
    std::vector<Eigen::Vector3d> points({
            {0, 0, 0},
            {1, 0, 0},
            {2, 0, 0},
            {3, 0, 0},
            {4, 0, 0},
            {5, 0, 0},
            {6, 0, 0},
            {7, 0, 0},
    });
    std::vector<Eigen::Vector3d> normals({
            {0, 0, 0},
            {0, 1, 0},
            {0, 2, 0},
            {0, 3, 0},
            {0, 4, 0},
            {0, 5, 0},
            {0, 6, 0},
            {0, 7, 0},
    });
    std::vector<Eigen::Vector3d> colors({
            {0.0, 0.5, 0.0},
            {0.0, 0.5, 0.1},
            {0.0, 0.5, 0.2},
            {0.0, 0.5, 0.3},
            {0.0, 0.5, 0.4},
            {0.0, 0.5, 0.5},
            {0.0, 0.5, 0.6},
            {0.0, 0.5, 0.7},
    });
    geometry::PointCloud pcd;
    pcd.points_ = points;
    pcd.normals_ = normals;
    pcd.colors_ = colors;

    pcd.EstimateColorGradients(geometry::KDTreeSearchParamHybrid(3.0, 5));

    double c = 0.0333333;

    ExpectEQ(pcd.color_gradients_,
             std::vector<Eigen::Vector3d>({{0.0, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {c, 0.0, 0.0},
                                           {0.0, 0.0, 0.0}}),
             1e-5);
}

TEST(PointCloud, OrientNormalsToAlignWithDirection) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1},
    });
    pcd.EstimateNormals(geometry::KDTreeSearchParamKNN(/*knn=*/4));
    pcd.NormalizeNormals();
    double v = 1.0 / std::sqrt(3.0);
    pcd.OrientNormalsToAlignWithDirection(Eigen::Vector3d{0, 0, -1});
    ExpectEQ(pcd.normals_, std::vector<Eigen::Vector3d>({{-v, -v, -v},
                                                         {v, v, -v},
                                                         {-v, v, -v},
                                                         {v, -v, -v},
                                                         {v, -v, -v},
                                                         {-v, v, -v},
                                                         {v, v, -v},
                                                         {-v, -v, -v}}));

    // normal.norm() == 0 case
    pcd.points_ = std::vector<Eigen::Vector3d>{{10, 10, 10}};
    pcd.normals_ = std::vector<Eigen::Vector3d>{{0, 0, 0}};
    pcd.OrientNormalsToAlignWithDirection(Eigen::Vector3d{0, 0, -1});
    pcd.normals_ = std::vector<Eigen::Vector3d>{{0, 0, -1}};
}

TEST(PointCloud, OrientNormalsTowardsCameraLocation) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 1, 0},
            {1, 0, 0},
            {1, 1, 0},
    });
    pcd.EstimateNormals(geometry::KDTreeSearchParamKNN(/*knn=*/4));
    pcd.NormalizeNormals();
    std::vector<Eigen::Vector3d> ref_normals(
            {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}});
    std::vector<Eigen::Vector3d> ref_normals_rev(
            {{0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}});

    // Initial
    ExpectEQ(pcd.normals_, ref_normals);
    // Camera outside
    pcd.OrientNormalsTowardsCameraLocation(Eigen::Vector3d{2, 3, 4});
    ExpectEQ(pcd.normals_, ref_normals);
    // Camera inside
    pcd.OrientNormalsTowardsCameraLocation(Eigen::Vector3d{-2, -3, -4});
    ExpectEQ(pcd.normals_, ref_normals_rev);
}

TEST(PointCloud, OrientNormalsConsistentTangentPlane) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1},
            {0.5, 0.5, -0.25},
            {0.5, 0.5, 1.25},
            {0.5, -0.25, 0.5},
            {0.5, 1.25, 0.5},
            {-0.25, 0.5, 0.5},
            {1.25, 0.5, 0.5},
    });

    // Hard-coded test
    pcd.EstimateNormals(geometry::KDTreeSearchParamKNN(/*knn=*/4));
    double a = 0.57735;
    double b = 0.0927618;
    double c = 0.991358;
    ExpectEQ(pcd.normals_, std::vector<Eigen::Vector3d>({{a, a, a},
                                                         {-a, -a, a},
                                                         {a, -a, a},
                                                         {-a, a, a},
                                                         {-a, a, a},
                                                         {a, -a, a},
                                                         {-a, -a, a},
                                                         {a, a, a},
                                                         {-b, -b, -c},
                                                         {b, b, -c},
                                                         {b, c, b},
                                                         {-b, c, -b},
                                                         {c, b, b},
                                                         {c, -b, -b}}));

    pcd.OrientNormalsConsistentTangentPlane(/*k=*/4);
    ExpectEQ(pcd.normals_, std::vector<Eigen::Vector3d>({{-a, -a, -a},
                                                         {-a, -a, a},
                                                         {-a, a, -a},
                                                         {-a, a, a},
                                                         {a, -a, -a},
                                                         {a, -a, a},
                                                         {a, a, -a},
                                                         {a, a, a},
                                                         {-b, -b, -c},
                                                         {-b, -b, c},
                                                         {-b, -c, -b},
                                                         {-b, c, -b},
                                                         {-c, -b, -b},
                                                         {c, -b, -b}}));
}

TEST(PointCloud, ComputePointCloudToPointCloudDistance) {
    geometry::PointCloud pc0({{0, 0, 0}, {1, 2, 0}, {2, 2, 0}});
    geometry::PointCloud pc1({{-1, 0, 0}, {-2, 0, 0}, {-1, 2, 0}});

    pc0.ComputePointCloudDistance(pc1);
    ExpectEQ(pc0.ComputePointCloudDistance(pc1),
             std::vector<double>({1, 2, 3}));
}

TEST(PointCloud, ComputeMeanAndCovariance) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1},
    });

    Eigen::Vector3d ref_mean(0.5, 0.5, 0.5);
    Eigen::Matrix3d ref_covariance;
    ref_covariance << 0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25;

    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    std::tie(mean, covariance) = pcd.ComputeMeanAndCovariance();
    ExpectEQ(mean, ref_mean);
    ExpectEQ(covariance, ref_covariance);

    pcd.points_ = {};
    ref_mean = Eigen::Vector3d::Zero();
    ref_covariance = Eigen::Matrix3d::Identity();
    std::tie(mean, covariance) = pcd.ComputeMeanAndCovariance();
    ExpectEQ(mean, ref_mean);
    ExpectEQ(covariance, ref_covariance);

    pcd.points_ = {{1, 1, 1}};
    ref_mean = Eigen::Vector3d({1, 1, 1});
    ref_covariance = Eigen::Matrix3d::Zero();
    std::tie(mean, covariance) = pcd.ComputeMeanAndCovariance();
    ExpectEQ(mean, ref_mean);
    ExpectEQ(covariance, ref_covariance);
}

TEST(PointCloud, ComputeMahalanobisDistance) {
    geometry::PointCloud pcd({
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 0, 2},
            {1, 1, 1},
    });
    std::vector<double> distance = pcd.ComputeMahalanobisDistance();
    ExpectEQ(distance,
             std::vector<double>({1.77951, 0.81650, 2.00000, 1.77951, 2.00000}),
             1e-4);

    // Empty
    pcd.points_ = {};
    distance = pcd.ComputeMahalanobisDistance();
    ExpectEQ(distance, std::vector<double>({}));

    // Nan if the covariance is not inversable
    pcd.points_ = {{1, 1, 1}};
    distance = pcd.ComputeMahalanobisDistance();
    EXPECT_EQ(distance.size(), 1);
    EXPECT_TRUE(std::isnan(distance[0]));

    pcd.points_ = {{0, 0, 0}, {1, 1, 1}};
    distance = pcd.ComputeMahalanobisDistance();
    EXPECT_EQ(distance.size(), 2);
    EXPECT_TRUE(std::isnan(distance[0]) && std::isnan(distance[1]));
}

TEST(PointCloud, ComputeNearestNeighborDistance) {
    geometry::PointCloud pcd;

    // Regular case
    pcd.points_ = {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 2, 0}};
    ExpectEQ(pcd.ComputeNearestNeighborDistance(),
             std::vector<double>({0, 0, 1, 2}));

    // < 2 points
    pcd.points_ = {};
    ExpectEQ(pcd.ComputeNearestNeighborDistance(), std::vector<double>({}));
    pcd.points_ = {{10, 10, 10}};
    ExpectEQ(pcd.ComputeNearestNeighborDistance(), std::vector<double>({0}));

    // 2 points
    pcd.points_ = {{0, 0, 0}, {1, 0, 0}};
    ExpectEQ(pcd.ComputeNearestNeighborDistance(), std::vector<double>({1, 1}));
}

TEST(PointCloud, ComputeConvexHull) {
    geometry::PointCloud pcd;
    std::shared_ptr<geometry::TriangleMesh> mesh;
    std::vector<size_t> pt_map;

    // Needs at least 4 points
    pcd.points_ = {};
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.points_ = {{0, 0, 0}};
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.points_ = {{0, 0, 0}, {0, 0, 1}};
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.points_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}};
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());

    // Hard-coded test
    pcd.points_ = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
    std::tie(mesh, pt_map) = pcd.ComputeConvexHull();
    EXPECT_EQ(pt_map, std::vector<size_t>({2, 3, 0, 1}));
    ExpectEQ(mesh->vertices_, ApplyIndices(pcd.points_, pt_map));

    // Hard-coded test
    pcd.points_ = {{0.5, 0.5, 0.5}, {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                   {1, 0, 0},       {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    std::tie(mesh, pt_map) = pcd.ComputeConvexHull();
    EXPECT_EQ(pt_map, std::vector<size_t>({7, 3, 1, 5, 6, 2, 8, 4}));
    ExpectEQ(mesh->vertices_, ApplyIndices(pcd.points_, pt_map));
    ExpectEQ(mesh->triangles_, std::vector<Eigen::Vector3i>({{0, 1, 2},
                                                             {0, 3, 2},
                                                             {4, 3, 2},
                                                             {4, 5, 2},
                                                             {4, 0, 3},
                                                             {4, 0, 6},
                                                             {7, 1, 2},
                                                             {7, 5, 2},
                                                             {7, 0, 1},
                                                             {7, 0, 6},
                                                             {7, 4, 5},
                                                             {7, 4, 6}}));
}

TEST(PointCloud, HiddenPointRemoval) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    EXPECT_EQ(pcd.points_.size(), 196133);
    ExpectEQ(pcd.GetMaxBound(), Eigen::Vector3d(3.96609, 2.427476, 2.55859));
    ExpectEQ(pcd.GetMinBound(), Eigen::Vector3d(0.558594, 0.832031, 0.566637));

    // Hard-coded test
    std::shared_ptr<geometry::TriangleMesh> mesh;
    std::vector<size_t> pt_map;
    std::tie(mesh, pt_map) =
            pcd.HiddenPointRemoval(Eigen::Vector3d{0, 0, 5}, 5 * 100);
    ExpectEQ(mesh->vertices_, ApplyIndices(pcd.points_, pt_map));
    EXPECT_EQ(mesh->vertices_.size(), 24581);
}

TEST(PointCloud, ClusterDBSCAN) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.ply", pcd);
    EXPECT_EQ(pcd.points_.size(), 196133);

    // Hard-coded test
    std::vector<int> cluster = pcd.ClusterDBSCAN(0.02, 10, false);
    EXPECT_EQ(cluster.size(), 196133);
    std::unordered_set<int> cluster_set(cluster.begin(), cluster.end());
    EXPECT_EQ(cluster_set.size(), 11);
    int cluster_sum = std::accumulate(cluster.begin(), cluster.end(), 0);
    EXPECT_EQ(cluster_sum, 398580);
}

TEST(PointCloud, SegmentPlane) {
    geometry::PointCloud pcd;
    io::ReadPointCloud(std::string(TEST_DATA_DIR) + "/fragment.pcd", pcd);
    EXPECT_EQ(pcd.points_.size(), 113662);

    // Hard-coded test
    Eigen::Vector4d plane_model;
    std::vector<size_t> inliers;
    std::tie(plane_model, inliers) = pcd.SegmentPlane(0.01, 3, 1000);

    // TODO: seed the ransac
    ExpectEQ(plane_model, Eigen::Vector4d(-0.06, -0.10, 0.99, -1.06), 0.1);
}

TEST(PointCloud, SegmentPlaneKnownPlane) {
    // Points sampled from the plane x + y + z + 1 = 0
    std::vector<Eigen::Vector3d> ref = {{1.0, 1.0, -3.0},
                                        {2.0, 2.0, -5.0},
                                        {-1.0, -1.0, 1.0},
                                        {-2.0, -2.0, 3.0},
                                        {10.0, 10.0, -21.0}};
    geometry::PointCloud pcd(ref);

    Eigen::Vector4d plane_model;
    std::vector<size_t> inliers;
    std::tie(plane_model, inliers) = pcd.SegmentPlane(0.01, 3, 10);
    ExpectEQ(pcd.SelectByIndex(inliers)->points_, ref);
}

TEST(PointCloud, CreateFromDepthImage) {
    const std::string trajectory_path =
            std::string(TEST_DATA_DIR) + "/RGBD/trajectory.log";
    const std::string im_depth_path =
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png";

    camera::PinholeCameraTrajectory trajectory;
    io::ReadPinholeCameraTrajectory(trajectory_path, trajectory);
    camera::PinholeCameraIntrinsic intrinsic =
            trajectory.parameters_[0].intrinsic_;
    Eigen::Matrix4d extrinsic = trajectory.parameters_[0].extrinsic_;
    std::shared_ptr<geometry::Image> im_depth =
            io::CreateImageFromFile(im_depth_path);

    std::shared_ptr<geometry::PointCloud> pcd =
            geometry::PointCloud::CreateFromDepthImage(*im_depth, intrinsic,
                                                       extrinsic);

    // Hard-coded test
    EXPECT_EQ(pcd->points_.size(), 267129);
    ExpectEQ(pcd->GetMinBound(), Eigen::Vector3d(-2.59579, 0.120689, 1.64421),
             1e-5);
    ExpectEQ(pcd->GetMaxBound(), Eigen::Vector3d(-1.08349, 1.68228, 4.18797),
             1e-5);
    EXPECT_EQ(pcd->colors_.size(), 0);
    // visualization::DrawGeometries({pcd}); // Uncomment for manual check
}

TEST(PointCloud, CreateFromRGBDImage) {
    const std::string trajectory_path =
            std::string(TEST_DATA_DIR) + "/RGBD/trajectory.log";
    const std::string im_depth_path =
            std::string(TEST_DATA_DIR) + "/RGBD/depth/00000.png";
    const std::string im_rgb_path =
            std::string(TEST_DATA_DIR) + "/RGBD/color/00000.jpg";

    camera::PinholeCameraTrajectory trajectory;
    io::ReadPinholeCameraTrajectory(trajectory_path, trajectory);
    camera::PinholeCameraIntrinsic intrinsic =
            trajectory.parameters_[0].intrinsic_;
    Eigen::Matrix4d extrinsic = trajectory.parameters_[0].extrinsic_;

    std::shared_ptr<geometry::Image> im_depth =
            io::CreateImageFromFile(im_depth_path);
    std::shared_ptr<geometry::Image> im_depth_float =
            im_depth->ConvertDepthToFloatImage();
    std::shared_ptr<geometry::Image> im_rgb =
            io::CreateImageFromFile(im_rgb_path);
    geometry::RGBDImage im_rgbd(*im_rgb, *im_depth_float);

    std::shared_ptr<geometry::PointCloud> pcd =
            geometry::PointCloud::CreateFromRGBDImage(im_rgbd, intrinsic,
                                                      extrinsic);

    // Hard-coded test
    EXPECT_EQ(pcd->points_.size(), 267129);
    ExpectEQ(pcd->GetMinBound(), Eigen::Vector3d(-2.59579, 0.120689, 1.64421),
             1e-5);
    ExpectEQ(pcd->GetMaxBound(), Eigen::Vector3d(-1.08349, 1.68228, 4.18797),
             1e-5);
    EXPECT_EQ(pcd->colors_.size(), pcd->points_.size());
    // visualization::DrawGeometries({pcd}); // Uncomment for manual check
}

}  // namespace tests
}  // namespace open3d
