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

#include "open3d/t/geometry/PointCloud.h"

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "tests/UnitTest.h"

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
    EXPECT_FALSE(pcd.HasPoints());
    EXPECT_FALSE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());

    // Default device.
    EXPECT_EQ(pcd.GetDevice(), core::Device("CPU:0"));
}

TEST_P(PointCloudPermuteDevices, ConstructFromPoints) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    core::Tensor points = core::Tensor::Ones({10, 3}, dtype, device);
    core::Tensor single_point = core::Tensor::Ones({3}, dtype, device);

    t::geometry::PointCloud pcd(points);
    EXPECT_TRUE(pcd.HasPoints());
    EXPECT_EQ(pcd.GetPoints().GetLength(), 10);
}

TEST_P(PointCloudPermuteDevices, ConstructFromPointDict) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor points = core::Tensor::Ones({10, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({10, 3}, dtype, device) * 0.5;
    core::Tensor normals = core::Tensor::Ones({10, 3}, dtype, device) * 0.25;
    std::unordered_map<std::string, core::Tensor> point_dict{
            {"points", points},
            {"colors", colors},
            {"normals", normals},
    };

    t::geometry::PointCloud pcd(point_dict);
    EXPECT_TRUE(pcd.HasPoints());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_TRUE(pcd.HasPointNormals());

    EXPECT_TRUE(pcd.GetPoints().AllClose(
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
                                       {2, 3}, core::Dtype::Float32, device);
    pcd.SetPoints(points);

    EXPECT_FALSE(pcd.IsEmpty());
    EXPECT_TRUE(pcd.HasPoints());
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

    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, src_device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, src_device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, src_device) * 3;

    t::geometry::PointCloud pcd(src_device);

    pcd.SetPoints(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    // Copy is created on the dst_device.
    t::geometry::PointCloud pcd_copy = pcd.To(dst_device, /*copy=*/true);

    EXPECT_EQ(pcd_copy.GetDevice(), dst_device);
    EXPECT_EQ(pcd_copy.GetPoints().GetDtype(), pcd.GetPoints().GetDtype());
}

TEST_P(PointCloudPermuteDevices, Copy) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, device) * 3;

    t::geometry::PointCloud pcd(device);

    pcd.SetPoints(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    // Copy is on the same device as source.
    t::geometry::PointCloud pcd_copy = pcd.Clone();

    // Copy does not share the same memory with source (deep copy).
    EXPECT_FALSE(pcd_copy.GetPoints().IsSame(pcd.GetPoints()));
    EXPECT_FALSE(pcd_copy.GetPointColors().IsSame(pcd.GetPointColors()));
    EXPECT_FALSE(
            pcd_copy.GetPointAttr("labels").IsSame(pcd.GetPointAttr("labels")));

    // Copy has the same attributes and values as source.
    EXPECT_TRUE(pcd_copy.GetPoints().AllClose(pcd.GetPoints()));
    EXPECT_TRUE(pcd_copy.GetPoints().AllClose(pcd.GetPoints()));
    EXPECT_TRUE(pcd_copy.GetPointAttr("labels").AllClose(
            pcd.GetPointAttr("labels")));
    EXPECT_ANY_THROW(pcd_copy.GetPointNormals());
}

TEST_P(PointCloudPermuteDevices, Transform) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud pcd(device);
    core::Tensor transformation(
            std::vector<float>{1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1},
            {4, 4}, dtype, device);

    pcd.SetPoints(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));
    pcd.SetPointNormals(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));
    pcd.Transform(transformation);
    EXPECT_EQ(pcd.GetPoints().ToFlatVector<float>(),
              std::vector<float>({3, 3, 2}));
    EXPECT_EQ(pcd.GetPointNormals().ToFlatVector<float>(),
              std::vector<float>({2, 2, 1}));
}

TEST_P(PointCloudPermuteDevices, Translate) {
    core::Device device = GetParam();
    t::geometry::PointCloud pcd(device);
    core::Tensor translation(std::vector<float>{10, 20, 30}, {3},
                             core::Dtype::Float32, device);

    // Relative.
    pcd.SetPoints(core::Tensor(std::vector<float>{0, 1, 2, 6, 7, 8}, {2, 3},
                               core::Dtype::Float32, device));
    pcd.Translate(translation, /*relative=*/true);
    EXPECT_EQ(pcd.GetPoints().ToFlatVector<float>(),
              std::vector<float>({10, 21, 32, 16, 27, 38}));

    // Non-relative.
    pcd.SetPoints(core::Tensor(std::vector<float>{0, 1, 2, 6, 7, 8}, {2, 3},
                               core::Dtype::Float32, device));
    pcd.Translate(translation, /*relative=*/false);
    EXPECT_EQ(pcd.GetPoints().ToFlatVector<float>(),
              std::vector<float>({7, 17, 27, 13, 23, 33}));
}

TEST_P(PointCloudPermuteDevices, Scale) {
    core::Device device = GetParam();
    t::geometry::PointCloud pcd(device);
    core::Tensor points =
            core::Tensor(std::vector<float>{0, 0, 0, 1, 1, 1, 2, 2, 2}, {3, 3},
                         core::Dtype::Float32, device);
    pcd.SetPoints(points);
    core::Tensor center(std::vector<float>{1, 1, 1}, {3}, core::Dtype::Float32,
                        device);
    float scale = 4;
    pcd.Scale(scale, center);
    EXPECT_EQ(points.ToFlatVector<float>(),
              std::vector<float>({-3, -3, -3, 1, 1, 1, 5, 5, 5}));
}

TEST_P(PointCloudPermuteDevices, Rotate) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud pcd(device);
    core::Tensor rotation(std::vector<float>{1, 1, 0, 0, 1, 1, 0, 1, 0}, {3, 3},
                          dtype, device);
    core::Tensor center = core::Tensor::Ones({3}, dtype, device);

    pcd.SetPoints(
            core::Tensor(std::vector<float>{2, 2, 2}, {1, 3}, dtype, device));
    pcd.SetPointNormals(
            core::Tensor(std::vector<float>{1, 1, 1}, {1, 3}, dtype, device));

    pcd.Rotate(rotation, center);
    EXPECT_EQ(pcd.GetPoints().ToFlatVector<float>(),
              std::vector<float>({3, 3, 2}));
    EXPECT_EQ(pcd.GetPointNormals().ToFlatVector<float>(),
              std::vector<float>({2, 2, 1}));
}

TEST_P(PointCloudPermuteDevices, FromLegacyPointCloud) {
    core::Device device = GetParam();
    geometry::PointCloud legacy_pcd;
    legacy_pcd.points_ = std::vector<Eigen::Vector3d>{Eigen::Vector3d(0, 0, 0),
                                                      Eigen::Vector3d(0, 0, 0)};
    legacy_pcd.colors_ = std::vector<Eigen::Vector3d>{Eigen::Vector3d(1, 1, 1),
                                                      Eigen::Vector3d(1, 1, 1)};

    // Float32: vector3d will be converted to float32.
    core::Dtype dtype = core::Dtype::Float32;
    t::geometry::PointCloud pcd = t::geometry::PointCloud::FromLegacyPointCloud(
            legacy_pcd, dtype, device);
    EXPECT_TRUE(pcd.HasPoints());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());
    EXPECT_TRUE(pcd.GetPoints().AllClose(
            core::Tensor::Zeros({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));

    // Float64 case.
    dtype = core::Dtype::Float64;
    pcd = t::geometry::PointCloud::FromLegacyPointCloud(legacy_pcd, dtype,
                                                        device);
    EXPECT_TRUE(pcd.HasPoints());
    EXPECT_TRUE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointNormals());
    EXPECT_TRUE(pcd.GetPoints().AllClose(
            core::Tensor::Zeros({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));
}

TEST_P(PointCloudPermuteDevices, ToLegacyPointCloud) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud pcd({
            {"points", core::Tensor::Ones({2, 3}, dtype, device)},
            {"colors", core::Tensor::Ones({2, 3}, dtype, device) * 2},
    });

    geometry::PointCloud legacy_pcd = pcd.ToLegacyPointCloud();
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
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud pcd({
            {"points", core::Tensor::Ones({2, 3}, dtype, device)},
            {"colors", core::Tensor::Ones({2, 3}, dtype, device) * 2},
            {"labels", core::Tensor::Ones({2, 3}, dtype, device) * 3},
    });

    EXPECT_TRUE(pcd.GetPoints().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device)));
    EXPECT_TRUE(pcd.GetPointColors().AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 2));
    EXPECT_TRUE(pcd.GetPointAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, dtype, device) * 3));
    EXPECT_ANY_THROW(pcd.GetPointNormals());

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPoints(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPointColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = pcd.GetPointAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::Tensor& tl = pcd.GetPointNormals(); (void)tl);
}

TEST_P(PointCloudPermuteDevices, Setters) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype, device);
    core::Tensor colors = core::Tensor::Ones({2, 3}, dtype, device) * 2;
    core::Tensor labels = core::Tensor::Ones({2, 3}, dtype, device) * 3;

    t::geometry::PointCloud pcd(device);

    pcd.SetPoints(points);
    pcd.SetPointColors(colors);
    pcd.SetPointAttr("labels", labels);

    EXPECT_TRUE(pcd.GetPoints().AllClose(
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

        EXPECT_ANY_THROW(pcd.SetPoints(cpu_points));
        EXPECT_ANY_THROW(pcd.SetPointColors(cpu_colors));
        EXPECT_ANY_THROW(pcd.SetPointAttr("labels", cpu_labels));
    }
}

TEST_P(PointCloudPermuteDevices, Has) {
    core::Device device = GetParam();
    core::Dtype dtype = core::Dtype::Float32;

    t::geometry::PointCloud pcd(device);
    EXPECT_FALSE(pcd.HasPoints());
    EXPECT_FALSE(pcd.HasPointColors());
    EXPECT_FALSE(pcd.HasPointAttr("labels"));

    pcd.SetPoints(core::Tensor::Ones({10, 3}, dtype, device));
    EXPECT_TRUE(pcd.HasPoints());

    // Different size.
    pcd.SetPointColors(core::Tensor::Ones({5, 3}, dtype, device));
    EXPECT_FALSE(pcd.HasPointColors());

    // Same size.
    pcd.SetPointColors(core::Tensor::Ones({10, 3}, dtype, device));
    EXPECT_TRUE(pcd.HasPointColors());
}

}  // namespace tests
}  // namespace open3d
