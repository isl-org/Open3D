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

#include "open3d/t/geometry/PointCloud.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class PointCloudPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(PointCloud,
                         PointCloudPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class PointCloudPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        PointCloud,
        PointCloudPermuteDevicePairs,
        testing::ValuesIn(PointCloudPermuteDevicePairs::TestCases()));

TEST_P(PointCloudPermuteDevices, DefaultConstructor) {
    t::geometry::PointCloud pcd;

    // Inherited from Geometry3D.
    EXPECT_EQ(pcd.GetGeometryType(),
              t::geometry::Geometry::GeometryType::PointCloud);
    EXPECT_EQ(pcd.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(pcd.IsEmpty());
    EXPECT_FALSE(pcd.HasPointPositions());
    EXPECT_FALSE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());

    // Default device.
    EXPECT_EQ(pcd.GetDevice(), core::Device("CPU:0"));

    // ToString
    EXPECT_EQ(pcd.ToString(),
              "PointCloud on CPU:0 [0 points ()] Attributes: None.");
}

TEST_P(PointCloudPermuteDevices, ConstructFromPoints) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;
    core::Tensor points = core::Tensor::Ones({10, 3}, dtype, device);
    core::Tensor single_point = core::Tensor::Ones({3}, dtype, device);

    t::geometry::PointCloud pcd(points);
    EXPECT_TRUE(pcd.HasPointPositions());
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 10);
}

TEST_P(PointCloudPermuteDevices, ConstructFromPointDict) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    core::Tensor points = core::Tensor::Ones({10, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({10, 3}, dtype, device) * 0.5;
    core::Tensor normals = core::Tensor::Ones({10, 3}, dtype, device) * 0.25;
    std::unordered_map<std::string, core::Tensor> point_dict{
            {"positions", points},
            {"colors", colors},
            {"normals", normals},
    };

    t::geometry::PointCloud pcd(point_dict);
    EXPECT_TRUE(pcd.HasPointPositions());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_TRUE(pcd.HasPointNormals());

    EXPECT_TRUE(pcd.GetPointPositions().AllClose(
            core::Tensor::Ones({10, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({10, 3}, dtype, device) * 0.5));
    EXPECT_TRUE(pcd.GetPointNormals().AllClose(
            core::Tensor::Ones({10, 3}, dtype, device) * 0.25));
}

TEST_P(PointCloudPermuteDevices, GetMinBound_GetMaxBound_GetCenter) {
    core::Device device = GetParam();
    t::geometry::PointCloud pcd(device);

    core::Tensor points = core::Tensor(std::vector<float>{1, 2, 3, 4, 5, 6},
                                       {2, 3}, core::Float32, device);
    pcd.SetPointPositions(points);

    EXPECT_FALSE(pcd.IsEmpty());
    EXPECT_TRUE(pcd.HasPointPositions());
    EXPECT_EQ(pcd.GetMinBound().ToFlatVector<float>(),
              std::vector<float>({1, 2, 3}));
    EXPECT_EQ(pcd.GetMaxBound().ToFlatVector<float>(),
              std::vector<float>({4, 5, 6}));
    EXPECT_EQ(pcd.GetCenter().ToFlatVector<float>(),
              std::vector<float>({2.5, 3.5, 4.5}));
}

TEST_P(PointCloudPermuteDevicePairs, CopyDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    core::Dtype dtype = core::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, src_device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, src_device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, src_device) * 3;

    t::geometry::PointCloud pcd(src_device);

    pcd.SetPointPositions(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    // Copy is created on the dst_device.
    t::geometry::PointCloud pcd_copy = pcd.To(dst_device, /*copy=*/true);

    EXPECT_EQ(pcd_copy.GetDevice(), dst_device);
    EXPECT_EQ(pcd_copy.GetPointPositions().GetDtype(),
              pcd.GetPointPositions().GetDtype());
}

TEST_P(PointCloudPermuteDevices, Copy) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, device) * 3;

    t::geometry::PointCloud pcd(device);

    pcd.SetPointPositions(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    // Copy is on the same device as source.
    t::geometry::PointCloud pcd_copy = pcd.Clone();

    // Copy does not share the same memory with source (deep copy).
    EXPECT_FALSE(pcd_copy.GetPointPositions().IsSame(pcd.GetPointPositions()));
    EXPECT_FALSE(pcd_copy.GetPointColors().IsSame(pcd.GetPointColors()));
    EXPECT_FALSE(
            pcd_copy.GetPointAttr("labels").IsSame(pcd.GetPointAttr("labels")));

    // Copy has the same attributes and values as source.
    EXPECT_TRUE(pcd_copy.GetPointPositions().AllClose(pcd.GetPointPositions()));
    EXPECT_TRUE(pcd_copy.GetPointPositions().AllClose(pcd.GetPointPositions()));
    EXPECT_TRUE(pcd_copy.GetPointAttr("labels").AllClose(
            pcd.GetPointAttr("labels")));
    EXPECT_ANY_THROW(pcd_copy.GetPointNormals());
}

TEST_P(PointCloudPermuteDevices, Transform) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;
    t::geometry::PointCloud pcd(device);
    core::Tensor transformation(
            std::vector<float>{1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
            {4, 4}, dtype, device);

    pcd.SetPointPositions(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));
    pcd.SetPointNormals(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));
    pcd.Transform(transformation);
    EXPECT_EQ(pcd.GetPointPositions().ToFlatVector<float>(),
              std::vector<float>({3, 3, 2}));
    EXPECT_EQ(pcd.GetPointNormals().ToFlatVector<float>(),
              std::vector<float>({2, 2, 1}));
}

TEST_P(PointCloudPermuteDevices, Translate) {
    core::Device device = GetParam();
    t::geometry::PointCloud pcd(device);
    core::Tensor translation(std::vector<float>{10, 20, 30}, {3}, core::Float32,
                             device);

    // Relative.
    pcd.SetPointPositions(core::Tensor(std::vector<float>{0, 1, 2, 6, 7, 8},
                                       {2, 3}, core::Float32, device));
    pcd.Translate(translation, /*relative=*/true);
    EXPECT_EQ(pcd.GetPointPositions().ToFlatVector<float>(),
              std::vector<float>({10, 21, 32, 16, 27, 38}));

    // Non-relative.
    pcd.SetPointPositions(core::Tensor(std::vector<float>{0, 1, 2, 6, 7, 8},
                                       {2, 3}, core::Float32, device));
    pcd.Translate(translation, /*relative=*/false);
    EXPECT_EQ(pcd.GetPointPositions().ToFlatVector<float>(),
              std::vector<float>({7, 17, 27, 13, 23, 33}));
}

TEST_P(PointCloudPermuteDevices, Scale) {
    core::Device device = GetParam();
    t::geometry::PointCloud pcd(device);
    core::Tensor points =
            core::Tensor(std::vector<float>{0, 0, 0, 1, 1, 1, 2, 2, 2}, {3, 3},
                         core::Float32, device);
    pcd.SetPointPositions(points);
    core::Tensor center(std::vector<float>{1, 1, 1}, {3}, core::Float32,
                        device);
    float scale = 4;
    pcd.Scale(scale, center);
    EXPECT_EQ(points.ToFlatVector<float>(),
              std::vector<float>({-3, -3, -3, 1, 1, 1, 5, 5, 5}));
}

TEST_P(PointCloudPermuteDevices, Rotate) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;
    t::geometry::PointCloud pcd(device);
    core::Tensor rotation(std::vector<float>{1, 1, 0, 0, 1, 1, 0, 1, 0}, {3, 3},
                          dtype, device);
    core::Tensor center = core::Tensor::Ones({3}, dtype, device);

    pcd.SetPointPositions(
            core::Tensor(std::vector<float>{2, 2, 2}, {1, 3}, dtype, device));
    pcd.SetPointNormals(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));

    pcd.Rotate(rotation, center);
    EXPECT_EQ(pcd.GetPointPositions().ToFlatVector<float>(),
              std::vector<float>({3, 3, 2}));
    EXPECT_EQ(pcd.GetPointNormals().ToFlatVector<float>(),
              std::vector<float>({2, 2, 1}));
}

TEST_P(PointCloudPermuteDevices, EstimateNormals) {
    core::Device device = GetParam();

    core::Tensor points = core::Tensor::Init<double>({{0, 0, 0},
                                                      {0, 0, 1},
                                                      {0, 1, 0},
                                                      {0, 1, 1},
                                                      {1, 0, 0},
                                                      {1, 0, 1},
                                                      {1, 1, 0},
                                                      {1, 1, 1}},
                                                     device);
    t::geometry::PointCloud pcd(points);

    // Estimate normals using Hybrid Search.
    pcd.EstimateNormals(4, 2.0);

    core::Tensor normals =
            core::Tensor::Init<double>({{0.57735, 0.57735, 0.57735},
                                        {-0.57735, -0.57735, 0.57735},
                                        {0.57735, -0.57735, 0.57735},
                                        {-0.57735, 0.57735, 0.57735},
                                        {-0.57735, 0.57735, 0.57735},
                                        {0.57735, -0.57735, 0.57735},
                                        {-0.57735, -0.57735, 0.57735},
                                        {0.57735, 0.57735, 0.57735}},
                                       device);

    EXPECT_TRUE(pcd.GetPointNormals().AllClose(normals, 1e-4, 1e-4));
    pcd.RemovePointAttr("normals");

    // Estimate normals using KNN Search.
    pcd.EstimateNormals(4);
    EXPECT_TRUE(pcd.GetPointNormals().AllClose(normals, 1e-4, 1e-4));
}

TEST_P(PointCloudPermuteDevices, FromLegacy) {
    core::Device device = GetParam();
    geometry::PointCloud legacy_pcd;
    legacy_pcd.points_ = std::vector<Eigen::Vector3d>{Eigen::Vector3d(0, 0, 0),
                                                      Eigen::Vector3d(0, 0, 0)};
    legacy_pcd.colors_ = std::vector<Eigen::Vector3d>{Eigen::Vector3d(1, 1, 1),
                                                      Eigen::Vector3d(1, 1, 1)};

    // Float32: vector3d will be converted to float32.
    core::Dtype dtype = core::Float32;
    t::geometry::PointCloud pcd =
            t::geometry::PointCloud::FromLegacy(legacy_pcd, dtype, device);
    EXPECT_TRUE(pcd.HasPointPositions());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());
    EXPECT_TRUE(pcd.GetPointPositions().AllClose(
            core::Tensor::Zeros({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));

    // Float64 case.
    dtype = core::Float64;
    pcd = t::geometry::PointCloud::FromLegacy(legacy_pcd, dtype, device);
    EXPECT_TRUE(pcd.HasPointPositions());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());
    EXPECT_TRUE(pcd.GetPointPositions().AllClose(
            core::Tensor::Zeros({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));
}

TEST_P(PointCloudPermuteDevices, ToLegacy) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    t::geometry::PointCloud pcd({
            {"positions", core::Tensor::Ones({2, 3}, dtype, device)},
            {"colors", core::Tensor::Ones({2, 3}, dtype, device) * 2},
    });

    geometry::PointCloud legacy_pcd = pcd.ToLegacy();
    EXPECT_TRUE(legacy_pcd.HasPoints());
    EXPECT_TRUE(legacy_pcd.HasColors());
    EXPECT_FALSE(legacy_pcd.HasNormals());
    EXPECT_EQ(legacy_pcd.points_.size(), 2);
    EXPECT_EQ(legacy_pcd.colors_.size(), 2);
    EXPECT_EQ(legacy_pcd.normals_.size(), 0);
    ExpectEQ(legacy_pcd.points_,
             std::vector<Eigen::Vector3d>{Eigen::Vector3d(1, 1, 1),
                                          Eigen::Vector3d(1, 1, 1)});
    ExpectEQ(legacy_pcd.colors_,
             std::vector<Eigen::Vector3d>{Eigen::Vector3d(2, 2, 2),
                                          Eigen::Vector3d(2, 2, 2)});
}

TEST_P(PointCloudPermuteDevices, Getters) {
    using ::testing::AnyOf;
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    t::geometry::PointCloud pcd({
            {"positions", core::Tensor::Ones({2, 3}, dtype, device)},
            {"colors", core::Tensor::Ones({2, 3}, dtype, device) * 2},
            {"labels", core::Tensor::Ones({2, 3}, dtype, device) * 3},
    });

    EXPECT_TRUE(pcd.GetPointPositions().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 2));
    EXPECT_TRUE(pcd.GetPointAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 3));
    EXPECT_ANY_THROW(pcd.GetPointNormals());

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPointPositions(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPointColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPointAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::Tensor& tl = pcd.GetPointNormals(); (void)tl);

    // ToString
    std::string text = "PointCloud on " + device.ToString() +
                       " [2 points (Float32)] Attributes: ";
    EXPECT_THAT(pcd.ToString(),  // Compiler dependent output
                AnyOf(text + "colors (dtype = Float32, shape = {2, 3}), labels "
                             "(dtype = Float32, shape = {2, 3}).",
                      text + "labels (dtype = Float32, shape = (2, 3)), colors "
                             "(dtype = Float32, shape = {2, 3})."));
}

TEST_P(PointCloudPermuteDevices, Setters) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, device) * 3;

    t::geometry::PointCloud pcd(device);

    pcd.SetPointPositions(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    EXPECT_TRUE(pcd.GetPointPositions().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 2));
    EXPECT_TRUE(pcd.GetPointAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 3));
    EXPECT_ANY_THROW(pcd.GetPointNormals());

    // Mismatched device should throw an exception. This test is only
    // effective if device is a CUDA device.
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::Tensor cpu_points = core::Tensor::Ones({2, 3}, dtype, cpu_device);
        core::Tensor cpu_colors =
                core::Tensor::Ones({2, 3}, dtype, cpu_device) * 2;
        core::Tensor cpu_labels =
                core::Tensor::Ones({2, 3}, dtype, cpu_device) * 3;

        EXPECT_ANY_THROW(pcd.SetPointPositions(cpu_points));
        EXPECT_ANY_THROW(pcd.SetPointColors(cpu_colors));
        EXPECT_ANY_THROW(pcd.SetPointAttr("labels", cpu_labels));
    }
}

TEST_P(PointCloudPermuteDevices, Append) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, device);

    t::geometry::PointCloud pcd(device);

    pcd.SetPointPositions(points);
    pcd.SetPointColors(colors);

    t::geometry::PointCloud pcd2(device);

    pcd2 = pcd.Clone();
    pcd2.SetPointAttr("labels", labels);

    // Here pcd2 is being added to pcd, therefore it must have all the
    // attributes present in pcd and the resulting pointcloud will contain the
    // attributes of pcd only.
    t::geometry::PointCloud pcd3(device);
    pcd3 = pcd + pcd2;

    EXPECT_TRUE(pcd3.GetPointPositions().AllClose(
            core::Tensor::Ones({4, 3}, dtype, device)));
    EXPECT_TRUE(pcd3.GetPointColors().AllClose(
            core::Tensor::Ones({4, 3}, dtype, device)));

    EXPECT_ANY_THROW(pcd3.GetPointAttr("labels"));

    // pcd2 has an extra attribute "labels" which is missing in pcd, therefore
    // adding pcd to pcd2 will throw an error for missing attribute "labels"
    EXPECT_ANY_THROW(pcd2 + pcd);
}

TEST_P(PointCloudPermuteDevices, Has) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    t::geometry::PointCloud pcd(device);
    EXPECT_FALSE(pcd.HasPointPositions());
    EXPECT_FALSE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointAttr("labels"));

    pcd.SetPointPositions(core::Tensor::Ones({10, 3}, dtype, device));
    EXPECT_TRUE(pcd.HasPointPositions());

    // Different size.
    pcd.SetPointColors(core::Tensor::Ones({5, 3}, dtype, device));
    EXPECT_FALSE(pcd.HasPointColors());

    // Same size.
    pcd.SetPointColors(core::Tensor::Ones({10, 3}, dtype, device));
    EXPECT_TRUE(pcd.HasPointColors());
}

TEST_P(PointCloudPermuteDevices, RemovePointAttr) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Float32;

    t::geometry::PointCloud pcd({
            {"positions", core::Tensor::Ones({2, 3}, dtype, device)},
            {"colors", core::Tensor::Ones({2, 3}, dtype, device) * 2},
            {"labels", core::Tensor::Ones({2, 3}, dtype, device) * 3},
    });

    EXPECT_NO_THROW(pcd.GetPointAttr("labels"));
    pcd.RemovePointAttr("labels");
    EXPECT_ANY_THROW(pcd.GetPointAttr("labels"));

    // Not allowed to delete "positions" attribute.
    EXPECT_ANY_THROW(pcd.RemovePointAttr("positions"));
}

TEST_P(PointCloudPermuteDevices, CreateFromRGBDImage) {
    using ::testing::ElementsAre;
    using ::testing::UnorderedElementsAreArray;

    core::Device device = GetParam();
    float depth_scale = 1000.f, depth_max = 3.f;
    int stride = 1;
    core::Tensor im_depth =
            core::Tensor::Init<uint16_t>({{1000, 0}, {1000, 1000}}, device);
    core::Tensor im_color =
            core::Tensor::Init<float>({{{0.0, 0.0, 0.0}, {0.2, 0.2, 0.2}},
                                       {{0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}}},
                                      device);
    core::Tensor intrinsics = core::Tensor::Init<float>(
            {{10, 0, 1}, {0, 10, 1}, {0, 0, 1}}, device);
    core::Tensor extrinsics = core::Tensor::Eye(4, core::Float32, device);
    t::geometry::PointCloud pcd_ref(
            {{"positions",
              core::Tensor::Init<float>(
                      {{-0.1, -0.1, 1.0}, {0.0, -0.1, 1.0}, {0.0, 0.0, 1.0}},
                      device)},
             {"colors",
              core::Tensor::Init<float>(
                      {{0.0, 0.0, 0.0}, {0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}},
                      device)},
             {"normals",
              core::Tensor::Init<float>(
                      {{0.0, 0.0, 0.0}, {0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}},
                      device)}});

    // UnProject, no normals
    const bool with_normals = false;
    t::geometry::PointCloud pcd_out =
            t::geometry::PointCloud::CreateFromRGBDImage(
                    t::geometry::RGBDImage(im_color, im_depth), intrinsics,
                    extrinsics, depth_scale, depth_max, stride, with_normals);

    EXPECT_THAT(pcd_out.GetPointPositions().GetShape(), ElementsAre(3, 3));
    // Unordered check since output point cloud order is non-deterministic
    EXPECT_THAT(pcd_out.GetPointPositions().ToFlatVector<float>(),
                UnorderedElementsAreArray(
                        pcd_ref.GetPointPositions().ToFlatVector<float>()));
    EXPECT_TRUE(pcd_out.HasPointColors());
    EXPECT_THAT(pcd_out.GetPointColors().GetShape(), ElementsAre(3, 3));
    EXPECT_THAT(pcd_out.GetPointColors().ToFlatVector<float>(),
                UnorderedElementsAreArray(
                        pcd_ref.GetPointColors().ToFlatVector<float>()));
    EXPECT_FALSE(pcd_out.HasPointNormals());
}

TEST_P(PointCloudPermuteDevices, CreateFromRGBDOrDepthImageWithNormals) {
    core::Device device = GetParam();

    if (!t::geometry::Image::HAVE_IPPICV &&
        device.GetType() ==
                core::Device::DeviceType::CPU) {  // FilterBilateral on CPU
                                                  // needs IPPICV
        return;
    }

    core::Tensor extrinsics = core::Tensor::Eye(4, core::Float32, device);
    int stride = 1;
    float depth_scale = 10.f, depth_max = 2.5f;
    core::Tensor t_depth(
            std::vector<uint16_t>{1, 2,  3, 2, 1, 1, 3, 5, 3, 1, 1, 4, 7,
                                  4, 30, 1, 3, 5, 3, 1, 1, 2, 3, 2, 1},
            {5, 5, 1}, core::UInt16, device);
    core::Tensor t_color(
            std::vector<float>{0,   0,   0,   0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1,
                               0.1, 0.1, 0.0, 0.0, 0.0, 0,   0,   0,   0.2, 0.2,
                               0.2, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0,
                               0,   0,   0,   0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.3,
                               0.3, 0.3, 0.9, 0.9, 0.9, 0,   0,   0,   0.2, 0.2,
                               0.2, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0,
                               0,   0,   0,   0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1,
                               0.1, 0.1, 0.0, 0.0, 0.0},
            {5, 5, 3}, core::Float32, device);
    core::Tensor intrinsics(
            std::vector<double>{1.f, 0.f, 2.f, 0.f, 1.f, 2.f, 0.f, 0.f, 1.f},
            {3, 3}, core::Float64, device);
    core::Tensor t_vertex_ref, t_color_ref, t_normal_ref;
    // CUDA has slightly different output due to NPP vs IPP differences.
    if (device.IsCUDA()) {
        t_vertex_ref =
                core::Tensor::Init<float>({{-0.288695, -0.288695, 0.144348},
                                           {-0.233266, -0.466531, 0.233266},
                                           {0.0, -0.555697, 0.277849},
                                           {-0.333081, -0.166541, 0.166541},
                                           {-0.299973, -0.299973, 0.299973},
                                           {-0.355337, 0.0, 0.177669},
                                           {-0.33346, 0.0, 0.33346},
                                           {-0.333081, 0.166541, 0.166541},
                                           {-0.299973, 0.299973, 0.299973}},
                                          device);
        t_color_ref = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                 {0.1, 0.1, 0.1},
                                                 {0.2, 0.2, 0.2},
                                                 {0.0, 0.0, 0.0},
                                                 {0.2, 0.2, 0.2},
                                                 {0.0, 0.0, 0.0},
                                                 {0.3, 0.3, 0.3},
                                                 {0.0, 0.0, 0.0},
                                                 {0.2, 0.2, 0.2}},
                                                device);
        t_normal_ref =
                core::Tensor::Init<float>({{0.941573, 0.329163, 0.071364},
                                           {0.333815, 0.462634, -0.821302},
                                           {-0.318447, 0.404408, -0.857348},
                                           {0.984687, 0.138649, -0.105676},
                                           {0.24427, 0.134471, -0.960338},
                                           {0.980499, -0.140229, -0.137689},
                                           {0.226005, -0.132954, -0.965010},
                                           {0.941147, -0.325299, 0.0917771},
                                           {0.289271, -0.453489, -0.843012}},
                                          device);
    } else {
        t_vertex_ref =
                core::Tensor::Init<float>({{-0.292137, -0.292137, 0.146069},
                                           {-0.230743, -0.461487, 0.230743},
                                           {0.0, -0.569164, 0.284582},
                                           {-0.353444, -0.176722, 0.176722},
                                           {-0.277117, -0.277117, 0.277117},
                                           {-0.399304, 0.0, 0.199652},
                                           {-0.353444, 0.176722, 0.176722},
                                           {-0.277117, 0.277117, 0.277117}},
                                          device);
        t_color_ref = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                 {0.1, 0.1, 0.1},
                                                 {0.2, 0.2, 0.2},
                                                 {0.0, 0.0, 0.0},
                                                 {0.2, 0.2, 0.2},
                                                 {0.0, 0.0, 0.0},
                                                 {0.0, 0.0, 0.0},
                                                 {0.2, 0.2, 0.2}},
                                                device);
        t_normal_ref =
                core::Tensor::Init<float>({{0.886676, 0.419108, 0.195331},
                                           {0.351009, 0.310485, -0.883398},
                                           {-0.303793, 0.183558, -0.934888},
                                           {0.878084, 0.27837, -0.389203},
                                           {0.209566, 0.0993332, -0.972736},
                                           {0.687868, -0.266128, -0.675287},
                                           {0.854888, -0.495199, -0.154739},
                                           {0.238257, -0.29284, -0.926001}},
                                          device);
    }
    t::geometry::Image im_depth{t_depth}, im_color{t_color};

    // with normals: go through CreateVertexMap() and CreateNormalMap()
    const bool with_normals = true;
    // test without color
    t::geometry::PointCloud pcd_out =
            t::geometry::PointCloud::CreateFromDepthImage(
                    im_depth, intrinsics, extrinsics, depth_scale, depth_max,
                    stride, with_normals);

    EXPECT_TRUE(pcd_out.GetPointPositions().AllClose(t_vertex_ref));
    EXPECT_TRUE(pcd_out.HasPointNormals());
    EXPECT_TRUE(pcd_out.GetPointNormals().AllClose(t_normal_ref));
    EXPECT_FALSE(pcd_out.HasPointColors());

    // test with color
    pcd_out = t::geometry::PointCloud::CreateFromRGBDImage(
            t::geometry::RGBDImage(im_color, im_depth), intrinsics, extrinsics,
            depth_scale, depth_max, stride, with_normals);

    EXPECT_TRUE(pcd_out.GetPointPositions().AllClose(t_vertex_ref));
    EXPECT_TRUE(pcd_out.HasPointColors());
    EXPECT_TRUE(pcd_out.GetPointColors().AllClose(t_color_ref));
    EXPECT_TRUE(pcd_out.HasPointNormals());
    EXPECT_TRUE(pcd_out.GetPointNormals().AllClose(t_normal_ref));
}

TEST_P(PointCloudPermuteDevices, SelectByMask) {
    core::Device device = GetParam();

    const t::geometry::PointCloud pcd_small(
            core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                       {0.9, 0.2, 0.4},
                                       {0.3, 0.6, 0.8},
                                       {0.2, 0.4, 0.2}},
                                      device));
    const core::Tensor boolean_mask =
            core::Tensor::Init<bool>({true, false, false, true}, device);

    const auto pcd_select = pcd_small.SelectByMask(boolean_mask, false);
    EXPECT_TRUE(
            pcd_select.GetPointPositions().AllClose(core::Tensor::Init<float>(
                    {{0.1, 0.3, 0.9}, {0.2, 0.4, 0.2}}, device)));

    const auto pcd_select_invert = pcd_small.SelectByMask(boolean_mask, true);
    EXPECT_TRUE(pcd_select_invert.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.9, 0.2, 0.4}, {0.3, 0.6, 0.8}},
                                      device)));
}

TEST_P(PointCloudPermuteDevices, SelectByIndex) {
    core::Device device = GetParam();

    const t::geometry::PointCloud pcd_small(
            core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                       {0.9, 0.2, 0.4},
                                       {0.3, 0.6, 0.8},
                                       {0.2, 0.4, 0.2}},
                                      device));
    // Test indices without duplicated value.
    const core::Tensor indices = core::Tensor::Init<int64_t>({0, 3}, device);

    const auto pcd_select = pcd_small.SelectByIndex(indices, false);
    EXPECT_TRUE(
            pcd_select.GetPointPositions().AllClose(core::Tensor::Init<float>(
                    {{0.1, 0.3, 0.9}, {0.2, 0.4, 0.2}}, device)));

    const auto pcd_select_invert = pcd_small.SelectByIndex(indices, true);
    EXPECT_TRUE(pcd_select_invert.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.9, 0.2, 0.4}, {0.3, 0.6, 0.8}},
                                      device)));

    // Test indices with duplicated value.
    const core::Tensor duplicated_indices =
            core::Tensor::Init<int64_t>({0, 0, 3, 3}, device);

    const auto pcd_select_no_remove =
            pcd_small.SelectByIndex(duplicated_indices, false, false);
    EXPECT_TRUE(pcd_select_no_remove.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                       {0.1, 0.3, 0.9},
                                       {0.2, 0.4, 0.2},
                                       {0.2, 0.4, 0.2}},
                                      device)));

    const auto pcd_select_remove =
            pcd_small.SelectByIndex(duplicated_indices, false, true);
    EXPECT_TRUE(pcd_select_remove.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.1, 0.3, 0.9}, {0.2, 0.4, 0.2}},
                                      device)));

    const auto pcd_select_invert_no_remove =
            pcd_small.SelectByIndex(duplicated_indices, true, false);
    EXPECT_TRUE(pcd_select_invert_no_remove.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.9, 0.2, 0.4}, {0.3, 0.6, 0.8}},
                                      device)));

    const auto pcd_select_invert_remove =
            pcd_small.SelectByIndex(duplicated_indices, true, true);
    EXPECT_TRUE(pcd_select_invert_remove.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0.9, 0.2, 0.4}, {0.3, 0.6, 0.8}},
                                      device)));
}

TEST_P(PointCloudPermuteDevices, VoxelDownSample) {
    core::Device device = GetParam();

    // Value test
    t::geometry::PointCloud pcd_small(
            core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                       {0.9, 0.2, 0.4},
                                       {0.3, 0.6, 0.8},
                                       {0.2, 0.4, 0.2}},
                                      device));
    auto pcd_small_down = pcd_small.VoxelDownSample(1);
    EXPECT_TRUE(pcd_small_down.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0, 0, 0}}, device)));
}

TEST_P(PointCloudPermuteDevices, UniformDownSample) {
    core::Device device = GetParam();

    // Value test.
    t::geometry::PointCloud pcd_small(core::Tensor::Init<float>({{0, 0, 0},
                                                                 {1, 0, 0},
                                                                 {2, 0, 0},
                                                                 {3, 0, 0},
                                                                 {4, 0, 0},
                                                                 {5, 0, 0},
                                                                 {6, 0, 0},
                                                                 {7, 0, 0}},
                                                                device));
    auto pcd_small_down = pcd_small.UniformDownSample(3);
    EXPECT_TRUE(pcd_small_down.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{0, 0, 0}, {3, 0, 0}, {6, 0, 0}},
                                      device)));
}

TEST_P(PointCloudPermuteDevices, RandomDownSample) {
    core::Device device = GetParam();

    // Value test.
    t::geometry::PointCloud pcd_small(core::Tensor::Init<float>({{0, 0, 0},
                                                                 {1, 0, 0},
                                                                 {2, 0, 0},
                                                                 {3, 0, 0},
                                                                 {4, 0, 0},
                                                                 {5, 0, 0},
                                                                 {6, 0, 0},
                                                                 {7, 0, 0}},
                                                                device));
    auto pcd_small_down = pcd_small.RandomDownSample(0.5);
    EXPECT_TRUE(pcd_small_down.GetPointPositions().GetLength() == 4);
}

TEST_P(PointCloudPermuteDevices, RemoveRadiusOutliers) {
    core::Device device = GetParam();

    const t::geometry::PointCloud pcd_small(
            core::Tensor::Init<float>({{1.0, 1.0, 1.0},
                                       {1.1, 1.1, 1.1},
                                       {1.2, 1.2, 1.2},
                                       {1.3, 1.3, 1.3},
                                       {5.0, 5.0, 5.0},
                                       {5.1, 5.1, 5.1}},
                                      device));

    t::geometry::PointCloud output_pcd;
    core::Tensor selected_boolean_mask;
    std::tie(output_pcd, selected_boolean_mask) =
            pcd_small.RemoveRadiusOutliers(3, 0.5);

    EXPECT_TRUE(output_pcd.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{1.0, 1.0, 1.0},
                                       {1.1, 1.1, 1.1},
                                       {1.2, 1.2, 1.2},
                                       {1.3, 1.3, 1.3}},
                                      device)));
}

TEST_P(PointCloudPermuteDevices, ClusterDBSCAN) {
    core::Device device = GetParam();
    if (!device.IsCPU()) {
        GTEST_SKIP();
    }
    t::geometry::PointCloud pcd;
    data::PLYPointCloud pointcloud_ply;
    t::io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);
    EXPECT_EQ(pcd.GetPointPositions().GetLength(), 196133);

    // Hard-coded test
    core::Tensor cluster = pcd.ClusterDBSCAN(0.02, 10, false);

    EXPECT_EQ(cluster.GetDtype(), core::Int32);
    EXPECT_EQ(cluster.GetLength(), 196133);
    std::unordered_set<int> cluster_set(
            cluster.GetDataPtr<int>(),
            cluster.GetDataPtr<int>() + cluster.GetLength());
    EXPECT_EQ(cluster_set.size(), 11);
    int cluster_sum = cluster.Sum({0}).Item<int>();
    EXPECT_EQ(cluster_sum, 398580);
}

TEST_P(PointCloudPermuteDevices, ComputeConvexHull) {
    core::Device device = GetParam();
    if (!device.IsCPU()) {
        GTEST_SKIP();
    }
    t::geometry::PointCloud pcd;
    t::geometry::TriangleMesh mesh;

    // Needs at least 4 points
    pcd.SetPointPositions(core::Tensor({0, 3}, core::Float32));
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.SetPointPositions(core::Tensor::Init<float>({{0, 0, 0}}));
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.SetPointPositions(core::Tensor::Init<float>({{0, 0, 0}, {0, 0, 1}}));
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    pcd.SetPointPositions(
            core::Tensor::Init<float>({{0, 0, 0}, {0, 0, 1}, {0, 1, 0}}));
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());

    // Degenerate input
    pcd.SetPointPositions(core::Tensor::Init<float>(
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    EXPECT_ANY_THROW(pcd.ComputeConvexHull());
    // Allow adding random noise to fix the degenerate input
    EXPECT_NO_THROW(pcd.ComputeConvexHull(true));

    // Hard-coded test
    pcd.SetPointPositions(core::Tensor::Init<double>(
            {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}}));
    mesh = pcd.ComputeConvexHull();
    auto point_indices = mesh.GetVertexAttr("point_indices");
    EXPECT_EQ(point_indices.GetDtype(), core::Int32);
    EXPECT_EQ(point_indices.ToFlatVector<int>(),
              std::vector<int>({2, 3, 0, 1}));
    EXPECT_TRUE(mesh.GetVertexPositions().AllEqual(
            pcd.GetPointPositions().IndexGet({point_indices.To(core::Int64)})));

    // Hard-coded test
    pcd.SetPointPositions(core::Tensor::Init<double>({{0.5, 0.5, 0.5},
                                                      {0, 0, 0},
                                                      {0, 0, 1},
                                                      {0, 1, 0},
                                                      {0, 1, 1},
                                                      {1, 0, 0},
                                                      {1, 0, 1},
                                                      {1, 1, 0},
                                                      {1, 1, 1}}));
    mesh = pcd.ComputeConvexHull();
    point_indices = mesh.GetVertexAttr("point_indices");
    EXPECT_EQ(point_indices.ToFlatVector<int>(),
              std::vector<int>({7, 3, 1, 5, 6, 2, 8, 4}));
    EXPECT_TRUE(mesh.GetVertexPositions().AllEqual(
            pcd.GetPointPositions().IndexGet({point_indices.To(core::Int64)})));
    ExpectEQ(mesh.GetTriangleIndices().ToFlatVector<int>(),
             std::vector<int>{1, 0, 2,  //
                              0, 3, 2,  //
                              3, 4, 2,  //
                              4, 5, 2,  //
                              0, 4, 3,  //
                              4, 0, 6,  //
                              7, 1, 2,  //
                              5, 7, 2,  //
                              7, 0, 1,  //
                              0, 7, 6,  //
                              4, 7, 5,  //
                              7, 4, 6});
}

}  // namespace tests
}  // namespace open3d
