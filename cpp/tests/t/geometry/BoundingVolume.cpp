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

#include "open3d/t/geometry/BoundingVolume.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class AxisAlignedBoundingBoxPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(AxisAlignedBoundingBox,
                         AxisAlignedBoundingBoxPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class AxisAlignedBoundingBoxPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        AxisAlignedBoundingBox,
        AxisAlignedBoundingBoxPermuteDevicePairs,
        testing::ValuesIn(
                AxisAlignedBoundingBoxPermuteDevicePairs::TestCases()));

TEST_P(AxisAlignedBoundingBoxPermuteDevices, ConstructorNoArg) {
    t::geometry::AxisAlignedBoundingBox aabb;

    // Inherited from Geometry3D.
    EXPECT_EQ(aabb.GetGeometryType(),
              t::geometry::Geometry::GeometryType::AxisAlignedBoundingBox);
    EXPECT_EQ(aabb.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(aabb.IsEmpty());
    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, core::Device("CPU:0"))));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, core::Device("CPU:0"))));
    EXPECT_TRUE(aabb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, core::Device("CPU:0"))));

    EXPECT_EQ(aabb.GetDevice(), core::Device("CPU:0"));

    // Print Information.
    EXPECT_EQ(aabb.ToString(), "AxisAlignedBoundingBox[Float32, CPU:0]");
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Constructor) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 1}, device);

    // Attempt to construct with invalid min/max bound.
    EXPECT_THROW(t::geometry::AxisAlignedBoundingBox(max_bound, min_bound),
                 std::runtime_error);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    // Public members.
    EXPECT_FALSE(aabb.IsEmpty());
    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({-1, -1, -1}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));
    EXPECT_TRUE(aabb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));

    EXPECT_EQ(aabb.GetDevice(), device);
}

TEST_P(AxisAlignedBoundingBoxPermuteDevicePairs, CopyDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();

    const core::Dtype dtype = core::Float32;

    core::Tensor min_bound = core::Tensor::Ones({3}, dtype, src_device);
    core::Tensor max_bound = core::Tensor::Ones({3}, dtype, src_device) * 2;
    core::Tensor color = core::Tensor::Ones({3}, dtype, src_device);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);
    aabb.SetColor(color);

    // Copy is created on the dst_device.
    t::geometry::AxisAlignedBoundingBox aabb_copy =
            aabb.To(dst_device, /*copy=*/true);

    EXPECT_EQ(aabb_copy.GetDevice(), dst_device);
    EXPECT_EQ(aabb_copy.GetMinBound().GetDtype(),
              aabb.GetMinBound().GetDtype());
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Clone_Clear_IsEmpty) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 1}, device);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    EXPECT_EQ(aabb.GetDevice(), device);
    EXPECT_TRUE(aabb.GetMinBound().AllClose(min_bound));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(max_bound));
    EXPECT_TRUE(aabb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));

    // Clear.
    aabb.Clear();
    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, device)));
    EXPECT_TRUE(aabb.GetColor().AllClose(
            core::Tensor::Init<float>({1, 1, 1}, device)));

    // IsEmpty.
    EXPECT_TRUE(aabb.IsEmpty());
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Setters) {
    core::Device device = GetParam();

    t::geometry::AxisAlignedBoundingBox aabb(device);

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1.0, 1.0, 1.0}, device);
    core::Tensor color = core::Tensor::Init<float>({0.0, 0.0, 0.0}, device);

    // SetMinBound.
    aabb.SetMinBound(min_bound);
    EXPECT_TRUE(aabb.GetMinBound().AllClose(min_bound));

    // SetMaxBound.
    aabb.SetMaxBound(max_bound);
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(max_bound));

    // SetColor.
    aabb.SetColor(color);
    EXPECT_TRUE(aabb.GetColor().AllClose(color));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, GetProperties) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -2}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 2}, device);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    // GetCenter.
    EXPECT_TRUE(aabb.GetCenter().AllClose(
            core::Tensor::Init<float>({0, 0, 0}, device)));

    // GetExtent.
    EXPECT_TRUE(aabb.GetExtent().AllClose(
            core::Tensor::Init<float>({2, 2, 4}, device)));

    // GetHalfExtent.
    EXPECT_TRUE(aabb.GetHalfExtent().AllClose(
            core::Tensor::Init<float>({1, 1, 2}, device)));

    // GetMaxExtent.
    EXPECT_EQ(aabb.GetMaxExtent(), 4);

    // Volume.
    EXPECT_EQ(aabb.Volume(), 16);

    // GetXPercentage.
    EXPECT_EQ(aabb.GetXPercentage(0), 0.5);

    // GetYPercentage.
    EXPECT_EQ(aabb.GetYPercentage(0), 0.5);

    // GetZPercentage.
    EXPECT_EQ(aabb.GetZPercentage(0), 0.5);
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Translate) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -2}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 2}, device);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    aabb.Translate(core::Tensor::Init<float>({1, 1, 1}, device), true);

    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0, 0, -1}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({2, 2, 3}, device)));

    aabb.Translate(core::Tensor::Init<float>({1, 1, 1}, device), false);

    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0, 0, -1}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({2, 2, 3}, device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Scale) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -2}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 2}, device);

    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    aabb.Scale(0.5, aabb.GetCenter());

    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({-0.5, -0.5, -1}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0.5, 0.5, 1}, device)));

    aabb.Scale(2.0, aabb.GetCenter());

    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({-1, -1, -2}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({1, 1, 2}, device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, Operator) {
    core::Device device = GetParam();

    t::geometry::AxisAlignedBoundingBox aabb1(device);
    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -2}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 2}, device);
    t::geometry::AxisAlignedBoundingBox aabb2(min_bound, max_bound);

    aabb1 += aabb2;

    EXPECT_TRUE(aabb1.GetMinBound().AllClose(
            core::Tensor::Init<float>({-1, -1, -2}, device)));
    EXPECT_TRUE(aabb1.GetMaxBound().AllClose(
            core::Tensor::Init<float>({1, 1, 2}, device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, GetBoxPoints) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 1}, device);
    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    auto box_points = aabb.GetBoxPoints();

    EXPECT_TRUE(box_points.AllClose(core::Tensor::Init<float>({{-1, -1, -1},
                                                               {1, -1, -1},
                                                               {-1, 1, -1},
                                                               {-1, -1, 1},
                                                               {1, 1, 1},
                                                               {-1, 1, 1},
                                                               {1, -1, 1},
                                                               {1, 1, -1}},
                                                              device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, GetPointIndicesWithinBoundingBox) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 1}, device);
    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    core::Tensor points = core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                                     {0.9, 0.2, 0.4},
                                                     {0.3, 0.6, 0.8},
                                                     {0.2, 0.4, 0.2}},
                                                    device);

    core::Tensor indices = aabb.GetPointIndicesWithinBoundingBox(points);

    EXPECT_TRUE(indices.AllClose(
            core::Tensor::Init<int64_t>({0, 1, 2, 3}, device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, LegacyConversion) {
    core::Device device = GetParam();

    core::Tensor min_bound = core::Tensor::Init<float>({-1, -1, -1}, device);
    core::Tensor max_bound = core::Tensor::Init<float>({1, 1, 1}, device);
    t::geometry::AxisAlignedBoundingBox aabb(min_bound, max_bound);

    auto legacy_aabb = aabb.ToLegacy();
    ExpectEQ(legacy_aabb.min_bound_, Eigen::Vector3d(-1, -1, -1));
    ExpectEQ(legacy_aabb.max_bound_, Eigen::Vector3d(1, 1, 1));
    ExpectEQ(legacy_aabb.color_, Eigen::Vector3d(1, 1, 1));

    // In Legacy, the data-type is eigen-double, so the created aabb is of type
    // Float64.
    auto aabb_new = t::geometry::AxisAlignedBoundingBox::FromLegacy(
            legacy_aabb, core::Float64, device);
    EXPECT_TRUE(aabb_new.GetMinBound().AllClose(
            core::Tensor::Init<double>({-1, -1, -1}, device)));
    EXPECT_TRUE(aabb_new.GetMaxBound().AllClose(
            core::Tensor::Init<double>({1, 1, 1}, device)));
}

TEST_P(AxisAlignedBoundingBoxPermuteDevices, CreateFromPoints) {
    core::Device device = GetParam();

    core::Tensor points = core::Tensor::Init<float>({{0.1, 0.3, 0.9},
                                                     {0.9, 0.2, 0.4},
                                                     {0.3, 0.6, 0.8},
                                                     {0.2, 0.4, 0.2}},
                                                    device);
    t::geometry::AxisAlignedBoundingBox aabb =
            t::geometry::AxisAlignedBoundingBox::CreateFromPoints(points);

    EXPECT_TRUE(aabb.GetMinBound().AllClose(
            core::Tensor::Init<float>({0.1, 0.2, 0.2}, device)));
    EXPECT_TRUE(aabb.GetMaxBound().AllClose(
            core::Tensor::Init<float>({0.9, 0.6, 0.9}, device)));
}

}  // namespace tests
}  // namespace open3d
