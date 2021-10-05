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

#include "open3d/t/geometry/LineSet.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/TensorCheck.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class LineSetPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(LineSet,
                         LineSetPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class LineSetPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        LineSet,
        LineSetPermuteDevicePairs,
        testing::ValuesIn(LineSetPermuteDevicePairs::TestCases()));

TEST_P(LineSetPermuteDevices, DefaultConstructor) {
    t::geometry::LineSet lineset;

    // Inherited from Geometry3D.
    EXPECT_EQ(lineset.GetGeometryType(),
              t::geometry::Geometry::GeometryType::LineSet);
    EXPECT_EQ(lineset.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(lineset.IsEmpty());
    EXPECT_FALSE(lineset.HasPointPositions());
    EXPECT_FALSE(lineset.HasLineIndices());
    EXPECT_FALSE(lineset.HasLineColors());

    // Default device.
    EXPECT_EQ(lineset.GetDevice(), core::Device("CPU:0"));

    // ToString
    EXPECT_EQ(lineset.ToString(), R"(LineSet on CPU:0
[0 points ()] Attributes: None.
[0 lines ()] Attributes: None.)");
}

TEST_P(LineSetPermuteDevices, ConstructFromPointPositions) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor points = core::Tensor::Ones({10, 3}, core::Float32, device);
    core::Tensor single_point = core::Tensor::Ones({3}, core::Float32, device);

    core::Tensor lines = core::Tensor::Ones({10, 2}, core::Int64, device);
    core::Tensor single_line = core::Tensor::Ones({2}, core::Int64, device);

    t::geometry::LineSet lineset(points, lines);

    EXPECT_TRUE(lineset.HasPointPositions());
    EXPECT_EQ(lineset.GetPointPositions().GetLength(), 10);
    EXPECT_EQ(lineset.GetLineIndices().GetLength(), 10);
}

TEST_P(LineSetPermuteDevices, Getters) {
    using ::testing::AnyOf;
    core::Device device = GetParam();

    core::Tensor points = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor point_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor lines = core::Tensor::Ones({2, 2}, core::Int64, device);
    core::Tensor line_colors =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor line_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::LineSet lineset(points, lines);
    lineset.SetLineColors(line_colors);
    lineset.SetPointAttr("labels", point_labels);
    lineset.SetLineAttr("labels", line_labels);

    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device)));
    EXPECT_TRUE(lineset.GetPointAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));

    EXPECT_TRUE(lineset.GetLineIndices().AllClose(
            core::Tensor::Ones({2, 2}, core::Int64, device)));
    EXPECT_TRUE(lineset.GetLineColors().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2));
    EXPECT_TRUE(lineset.GetLineAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = lineset.GetPointPositions();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = lineset.GetPointAttr("labels");
                    (void)tl);

    EXPECT_NO_THROW(const core::Tensor& tl = lineset.GetLineIndices();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = lineset.GetLineColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = lineset.GetLineAttr("labels");
                    (void)tl);

    // ToString
    std::string text = "LineSet on " + device.ToString() +
                       "\n[2 points (Float32)] Attributes: labels (dtype = "
                       "Float32, shape = {2, 3})."
                       "\n[2 lines (Int64)] Attributes: ";
    EXPECT_THAT(lineset.ToString(),  // Compiler dependent output
                AnyOf(text + "labels (dtype = Float32, shape = {2, 3}), colors "
                             "(dtype = Float32, shape = {2, 3}).",
                      text + "colors (dtype = Float32, shape = {2, 3}), labels "
                             "(dtype = Float32, shape = {2, 3})."));
}

TEST_P(LineSetPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // if device is a CUDA device.
    t::geometry::LineSet lineset(device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::Tensor cpu_points =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device);
        core::Tensor cpu_lines =
                core::Tensor::Ones({2, 2}, core::Int64, cpu_device);
        core::Tensor cpu_colors =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 2;
        core::Tensor cpu_labels =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 3;

        EXPECT_ANY_THROW(lineset.SetPointPositions(cpu_points));
        EXPECT_ANY_THROW(lineset.SetLineIndices(cpu_lines));
        EXPECT_ANY_THROW(lineset.SetLineColors(cpu_colors));
        EXPECT_ANY_THROW(lineset.SetPointAttr("labels", cpu_labels));
    }
}

TEST_P(LineSetPermuteDevices, RemoveAttr) {
    core::Device device = GetParam();

    core::Tensor points = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor point_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor lines = core::Tensor::Ones({2, 2}, core::Int64, device);
    core::Tensor line_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::LineSet lineset(points, lines);

    lineset.SetPointAttr("labels", point_labels);
    EXPECT_NO_THROW(lineset.GetPointAttr("labels"));
    lineset.RemovePointAttr("labels");
    EXPECT_ANY_THROW(lineset.GetPointAttr("labels"));

    lineset.SetLineAttr("labels", line_labels);
    EXPECT_NO_THROW(lineset.GetLineAttr("labels"));
    lineset.RemoveLineAttr("labels");
    EXPECT_ANY_THROW(lineset.GetLineAttr("labels"));

    // Not allowed to delete primary key attribute.
    EXPECT_ANY_THROW(lineset.RemovePointAttr("positions"));
    EXPECT_ANY_THROW(lineset.RemoveLineAttr("indices"));
}

TEST_P(LineSetPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::LineSet lineset(device);
    EXPECT_FALSE(lineset.HasPointPositions());
    EXPECT_FALSE(lineset.HasPointAttr("labels"));
    EXPECT_FALSE(lineset.HasLineIndices());
    EXPECT_FALSE(lineset.HasLineColors());
    EXPECT_FALSE(lineset.HasLineAttr("labels"));

    lineset.SetPointPositions(
            core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(lineset.HasPointPositions());
    lineset.SetLineIndices(core::Tensor::Ones({10, 2}, core::Int64, device));
    EXPECT_TRUE(lineset.HasLineIndices());

    // Different size.
    lineset.SetLineColors(core::Tensor::Ones({5, 3}, core::Float32, device));
    EXPECT_FALSE(lineset.HasLineColors());

    // Same size.
    lineset.SetLineColors(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(lineset.HasLineColors());
}

TEST_P(LineSetPermuteDevicePairs, Copy_CopyDevice) {
    core::Device dst_device;
    core::Device src_device;
    std::tie(dst_device, src_device) = GetParam();
    core::Dtype dtype_float = core::Float32;
    core::Dtype dtype_int = core::Int32;

    core::Tensor points = core::Tensor::Ones({2, 3}, dtype_float, src_device);
    core::Tensor lines = core::Tensor::Ones({2, 2}, dtype_int, src_device);
    core::Tensor colors =
            core::Tensor::Ones({2, 3}, dtype_float, src_device) * 2;
    core::Tensor labels =
            core::Tensor::Ones({2, 3}, dtype_float, src_device) * 3;

    t::geometry::LineSet lineset(src_device);

    lineset.SetPointPositions(points);
    lineset.SetPointAttr("labels", labels);
    lineset.SetLineIndices(lines);
    lineset.SetLineColors(colors);

    // Copy is on the same device as source.
    t::geometry::LineSet lineset_copy = lineset.Clone();

    // Copy does not share the same memory with source (deep copy).
    EXPECT_FALSE(lineset_copy.GetPointPositions().IsSame(
            lineset.GetPointPositions()));
    EXPECT_FALSE(lineset_copy.GetLineColors().IsSame(lineset.GetLineColors()));
    EXPECT_FALSE(lineset_copy.GetPointAttr("labels").IsSame(
            lineset.GetPointAttr("labels")));

    // Copy has the same attributes and values as source.
    EXPECT_TRUE(lineset_copy.GetPointPositions().AllClose(
            lineset.GetPointPositions()));
    EXPECT_TRUE(
            lineset_copy.GetLineIndices().AllClose(lineset.GetLineIndices()));
    EXPECT_TRUE(lineset_copy.GetPointAttr("labels").AllClose(
            lineset.GetPointAttr("labels")));

    // Copy is created on the dst_device.
    t::geometry::LineSet lineset_copy_dev =
            lineset.To(dst_device, /*copy=*/true);

    EXPECT_EQ(lineset_copy_dev.GetDevice(), dst_device);
    // CopyDevice has the same attributes and values as source.
    EXPECT_TRUE(lineset_copy_dev.GetPointPositions()
                        .To(src_device)
                        .AllClose(lineset.GetPointPositions()));
    EXPECT_TRUE(lineset_copy_dev.GetLineIndices()
                        .To(src_device)
                        .AllClose(lineset.GetLineIndices()));
    EXPECT_TRUE(lineset_copy_dev.GetPointAttr("labels")
                        .To(src_device)
                        .AllClose(lineset.GetPointAttr("labels")));
}

TEST_P(LineSetPermuteDevices, GetMinBound_GetMaxBound_GetCenter) {
    core::Device device = GetParam();
    t::geometry::LineSet lineset(device);

    core::Tensor points = core::Tensor(std::vector<float>{1, 2, 3, 4, 5, 6},
                                       {2, 3}, core::Float32, device);
    lineset.SetPointPositions(points);

    EXPECT_FALSE(lineset.IsEmpty());
    EXPECT_TRUE(lineset.HasPointPositions());
    EXPECT_FALSE(lineset.HasLineIndices());
    EXPECT_EQ(lineset.GetMinBound().ToFlatVector<float>(),
              std::vector<float>({1, 2, 3}));
    EXPECT_EQ(lineset.GetMaxBound().ToFlatVector<float>(),
              std::vector<float>({4, 5, 6}));
    EXPECT_EQ(lineset.GetCenter().ToFlatVector<float>(),
              std::vector<float>({2.5, 3.5, 4.5}));
}

TEST_P(LineSetPermuteDevices, Transform) {
    core::Device device = GetParam();

    t::geometry::LineSet lineset(device);
    core::Tensor transformation = core::Tensor::Init<float>(
            {{1, 1, 0, 1}, {0, 1, 1, 1}, {0, 1, 0, 1}, {0, 0, 0, 1}}, device);

    lineset.SetPointPositions(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    lineset.Transform(transformation);
    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
}

TEST_P(LineSetPermuteDevices, Translate) {
    core::Device device = GetParam();

    t::geometry::LineSet lineset(device);
    core::Tensor translation = core::Tensor::Init<float>({10, 20, 30}, device);

    // Relative.
    lineset.SetPointPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));

    lineset.Translate(translation, /*relative=*/true);

    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{10, 21, 32}, {16, 27, 38}}, device)));

    // Non-relative.
    lineset.SetPointPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));
    lineset.Translate(translation, /*relative=*/false);

    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{7, 17, 27}, {13, 23, 33}}, device)));
}

TEST_P(LineSetPermuteDevices, Scale) {
    core::Device device = GetParam();

    t::geometry::LineSet lineset(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);
    double scale = 4;

    lineset.SetPointPositions(core::Tensor::Init<float>(
            {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}, device));

    lineset.Scale(scale, center);
    EXPECT_TRUE(lineset.GetPointPositions().AllClose(core::Tensor::Init<float>(
            {{-3, -3, -3}, {1, 1, 1}, {5, 5, 5}}, device)));
}

TEST_P(LineSetPermuteDevices, Rotate) {
    core::Device device = GetParam();

    t::geometry::LineSet lineset(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);

    lineset.SetPointPositions(
            core::Tensor::Init<float>({{2, 2, 2}, {2, 2, 2}}, device));

    lineset.Rotate(rotation, center);
    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
}

TEST_P(LineSetPermuteDevices, FromLegacy) {
    core::Device device = GetParam();
    geometry::LineSet legacy_lineset;
    legacy_lineset.points_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)};
    legacy_lineset.lines_ = std::vector<Eigen::Vector2i>{Eigen::Vector2i(3, 3),
                                                         Eigen::Vector2i(3, 3)};
    legacy_lineset.colors_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)};

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;
    t::geometry::LineSet lineset = t::geometry::LineSet::FromLegacy(
            legacy_lineset, float_dtype, int_dtype, device);

    EXPECT_TRUE(lineset.HasPointPositions());
    EXPECT_TRUE(lineset.HasLineIndices());
    EXPECT_TRUE(lineset.HasLineColors());

    EXPECT_NO_THROW(
            core::AssertTensorDtype(lineset.GetPointPositions(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(lineset.GetLineIndices(), int_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(lineset.GetLineColors(), float_dtype));

    EXPECT_TRUE(lineset.GetPointPositions().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 0));
    EXPECT_TRUE(lineset.GetLineIndices().AllClose(
            core::Tensor::Ones({2, 2}, int_dtype, device) * 3));
    EXPECT_TRUE(lineset.GetLineColors().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 1));
}

TEST_P(LineSetPermuteDevices, ToLegacy) {
    core::Device device = GetParam();

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;

    t::geometry::LineSet lineset(device);
    lineset.SetPointPositions(core::Tensor::Ones({2, 3}, float_dtype, device) *
                              0);
    lineset.SetLineIndices(core::Tensor::Ones({2, 2}, int_dtype, device) * 3);
    lineset.SetLineColors(core::Tensor::Ones({2, 3}, float_dtype, device) * 1);

    geometry::LineSet legacy_lineset = lineset.ToLegacy();
    EXPECT_EQ(legacy_lineset.points_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)}));
    EXPECT_EQ(legacy_lineset.lines_,
              std::vector<Eigen::Vector2i>(
                      {Eigen::Vector2i(3, 3), Eigen::Vector2i(3, 3)}));
    EXPECT_EQ(legacy_lineset.colors_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)}));
}

}  // namespace tests
}  // namespace open3d
