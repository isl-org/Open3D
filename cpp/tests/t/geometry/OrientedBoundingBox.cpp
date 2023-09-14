// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/t/geometry/BoundingVolume.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class OrientedBoundingBoxPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(OrientedBoundingBox,
                         OrientedBoundingBoxPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class OrientedBoundingBoxPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        OrientedBoundingBox,
        OrientedBoundingBoxPermuteDevicePairs,
        testing::ValuesIn(OrientedBoundingBoxPermuteDevicePairs::TestCases()));

TEST_P(OrientedBoundingBoxPermuteDevices, ConstructorNoArg) {
    t::geometry::OrientedBoundingBox obb;

    // Inherited from Geometry3D.
    EXPECT_EQ(obb.GetGeometryType(),
              t::geometry::Geometry::GeometryType::OrientedBoundingBox);
    EXPECT_EQ(obb.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(obb.IsEmpty());
    EXPECT_TRUE(obb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, core::Device("CPU:0"))));
    EXPECT_TRUE(obb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, core::Device("CPU:0"))));
    EXPECT_TRUE(obb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, core::Device("CPU:0"))));

    EXPECT_EQ(obb.GetDevice(), core::Device("CPU:0"));

    // Print Information.
    EXPECT_EQ(obb.ToString(), "OrientedBoundingBox[Float32, CPU:0]");
}

TEST_P(OrientedBoundingBoxPermuteDevices, Constructor) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent_err = core::Tensor::Init<float>({-1, 1, 1}, device);
    core::Tensor extent = core::Tensor::Init<float>({2, 2, 2}, device);
    core::Tensor rotation_err = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);

    // Attempt to construct with invalid extent and rotation.
    EXPECT_THROW(t::geometry::OrientedBoundingBox(center, rotation_err, extent),
                 std::runtime_error);
    EXPECT_THROW(t::geometry::OrientedBoundingBox(center, rotation, extent_err),
                 std::runtime_error);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    // Public members.
    EXPECT_FALSE(obb.IsEmpty());
    EXPECT_TRUE(obb.GetMinBound().AllClose(
            core::Tensor::Init<float>({-2, -2, -2}, device)));
    EXPECT_TRUE(obb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, device)));
    EXPECT_TRUE(obb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));

    EXPECT_EQ(obb.GetDevice(), device);
}

TEST_P(OrientedBoundingBoxPermuteDevicePairs, CopyDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    const core::Dtype dtype = core::Float32;

    core::Tensor center = core::Tensor::Ones({3}, dtype, src_device);
    core::Tensor extent = core::Tensor::Ones({3}, dtype, src_device) * 2;
    core::Tensor rotation = core::Tensor::Eye(3, dtype, src_device);
    core::Tensor color = core::Tensor::Ones({3}, dtype, src_device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);
    obb.SetColor(color);

    // Copy is created on the dst_device.
    t::geometry::OrientedBoundingBox obb_copy =
            obb.To(dst_device, /*copy=*/true);

    EXPECT_EQ(obb_copy.GetDevice(), dst_device);
    EXPECT_EQ(obb_copy.GetExtent().GetDtype(), obb.GetExtent().GetDtype());
}

TEST_P(OrientedBoundingBoxPermuteDevices, Setters) {
    core::Device device = GetParam();

    t::geometry::OrientedBoundingBox obb(device);

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device);
    core::Tensor color = core::Tensor::Init<float>({0.0, 0.0, 0.0}, device);

    // SetCenter.
    obb.SetCenter(center);
    EXPECT_TRUE(obb.GetCenter().AllClose(center));

    // SetExtent.
    obb.SetExtent(extent);
    EXPECT_TRUE(obb.GetExtent().AllClose(extent));

    // SetRotation.
    obb.SetRotation(rotation);
    EXPECT_TRUE(obb.GetRotation().AllClose(rotation));

    // SetColor.
    obb.SetColor(color);
    EXPECT_TRUE(obb.GetColor().AllClose(color));
}

TEST_P(OrientedBoundingBoxPermuteDevices, Translate) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    obb.Translate(core::Tensor::Init<float>({1, 1, 1}, device), true);

    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, device)));

    obb.Translate(core::Tensor::Init<float>({1, 1, 1}, device), false);

    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, Rotate) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);
    auto obb_copy = obb.Clone();

    core::Tensor rotation_apply = core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device);

    obb.Rotate(rotation_apply);
    obb_copy.Rotate(rotation_apply,
                    core::Tensor::Zeros({3}, core::Float32, device));

    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({-1, -1, -1}, device)));
    EXPECT_TRUE(obb.GetRotation().AllClose(core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device)));
    EXPECT_TRUE(obb_copy.GetCenter().AllClose(
            core::Tensor::Init<float>({-1, 1, 1}, device)));
    EXPECT_TRUE(obb_copy.GetRotation().AllClose(core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, Transform) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    core::Tensor transformation = core::Tensor::Init<float>(
            {{1, 0, 0, 1}, {0, -1, 0, 2}, {0, 0, -1, 3}, {0, 0, 0, 1}}, device);

    obb.Transform(transformation);

    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({0, 1, 2}, device)));
    EXPECT_TRUE(obb.GetRotation().AllClose(core::Tensor::Init<float>(
            {{1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, Scale) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);
    auto obb_copy = obb.Clone();

    obb.Scale(2.0);
    obb_copy.Scale(2.0, core::Tensor::Zeros({3}, core::Float32, device));

    EXPECT_TRUE(obb.GetExtent().AllClose(
            core::Tensor::Init<float>({2, 2, 2}, device)));
    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({-1, -1, -1}, device)));
    EXPECT_TRUE(obb_copy.GetCenter().AllClose(
            core::Tensor::Init<float>({-2, -2, -2}, device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, GetBoxPoints) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1., -1., -1.}, device);
    core::Tensor extent = core::Tensor::Init<float>({0.0, 0.0, 1.0}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    auto box_points = obb.GetBoxPoints();

    EXPECT_TRUE(
            box_points.AllClose(core::Tensor::Init<float>({{-1.0, -1.0, -1.5},
                                                           {-1.0, -1.0, -1.5},
                                                           {-1.0, -1.0, -1.5},
                                                           {-1.0, -1.0, -0.5},
                                                           {-1.0, -1.0, -0.5},
                                                           {-1.0, -1.0, -0.5},
                                                           {-1.0, -1.0, -0.5},
                                                           {-1.0, -1.0, -1.5}},
                                                          device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, GetPointIndicesWithinBoundingBox) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({0.5, 0.5, 0.5}, device);
    core::Tensor rotation = core::Tensor::Eye(3, core::Float32, device);
    core::Tensor extent = core::Tensor::Init<float>({1, 1, 1}, device);
    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    core::Tensor points = core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                                     {-0.2, 0.2, 0.5},
                                                     {0.9, 0.2, 0.4},
                                                     {0.3, 0.6, 0.8},
                                                     {0.2, 0.4, 0.2},
                                                     {1.2, 0.3, 0.5}},
                                                    device);

    core::Tensor indices = obb.GetPointIndicesWithinBoundingBox(points);

    EXPECT_TRUE(indices.AllClose(
            core::Tensor::Init<int64_t>({0, 2, 3, 4}, device)));
}

TEST_P(OrientedBoundingBoxPermuteDevices, LegacyConversion) {
    core::Device device = GetParam();

    core::Tensor center = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor extent = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor rotation =
            core::Tensor::Init<float>({{0.770062, -0.231286, -0.594569},
                                       {-0.584733, 0.116831, -0.802769},
                                       {0.255133, 0.965845, -0.0452729}},
                                      device);

    t::geometry::OrientedBoundingBox obb(center, rotation, extent);

    auto legacy_obb = obb.ToLegacy();
    ExpectEQ(legacy_obb.center_, Eigen::Vector3d(-1, -1, -1));
    ExpectEQ(legacy_obb.extent_, Eigen::Vector3d(1, 1, 1));
    ExpectEQ(legacy_obb.color_, Eigen::Vector3d(1, 1, 1));

    // In Legacy, the data-type is eigen-double, so the created aabb is of
    // type Float64.
    auto obb_new = t::geometry::OrientedBoundingBox::FromLegacy(
            legacy_obb, core::Float64, device);
    EXPECT_TRUE(obb_new.GetCenter().AllClose(
            core::Tensor::Init<double>({-1, -1, -1}, device)));
    EXPECT_TRUE(obb_new.GetExtent().AllClose(
            core::Tensor::Init<double>({1, 1, 1}, device)));
    EXPECT_TRUE(obb_new.GetRotation().AllClose(
            core::Tensor::Init<double>({{0.770062, -0.231286, -0.594569},
                                        {-0.584733, 0.116831, -0.802769},
                                        {0.255133, 0.965845, -0.0452729}},
                                       device),
            1e-5, 1e-5));
}

TEST_P(OrientedBoundingBoxPermuteDevices, CreateFromPoints) {
    core::Device device = GetParam();

    core::Tensor points = core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                                     {0.9, 0.2, 0.4},
                                                     {0.3, 0.6, 0.8},
                                                     {0.2, 0.4, 0.2}},
                                                    device);
    t::geometry::OrientedBoundingBox obb =
            t::geometry::OrientedBoundingBox::CreateFromPoints(points);

    EXPECT_TRUE(obb.GetCenter().AllClose(
            core::Tensor::Init<float>({0.376834, 0.383993, 0.438357}, device)));
    EXPECT_TRUE(obb.GetExtent().AllClose(
            core::Tensor::Init<float>({0.936462, 0.593233, 0.345308}, device)));
    EXPECT_TRUE(obb.GetRotation().AllClose(
            core::Tensor::Init<float>({{0.77006164, -0.5847325, 0.25513324},
                                       {-0.23128562, 0.11683135, 0.96584543},
                                       {-0.59456878, -0.80276917, -0.04527287}},
                                      device)));
}

}  // namespace tests
}  // namespace open3d
