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

#include "open3d/t/geometry/TriangleMesh.h"

#include <gmock/gmock.h>

#include "core/CoreTest.h"
#include "open3d/core/TensorCheck.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    t::geometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              t::geometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasVertexPositions());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasTriangleIndices());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default device.
    EXPECT_EQ(mesh.GetDevice(), core::Device("CPU:0"));
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromVertices) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor vertices = core::Tensor::Ones({10, 3}, core::Float32, device);
    core::Tensor single_vertex = core::Tensor::Ones({3}, core::Float32, device);

    core::Tensor triangles = core::Tensor::Ones({10, 3}, core::Int64, device);
    core::Tensor single_triangle = core::Tensor::Ones({3}, core::Int64, device);

    t::geometry::TriangleMesh mesh(vertices, triangles);

    EXPECT_TRUE(mesh.HasVertexPositions());
    EXPECT_EQ(mesh.GetVertexPositions().GetLength(), 10);
    EXPECT_EQ(mesh.GetTriangleIndices().GetLength(), 10);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_colors =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_normals =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetVertexAttr("labels", vertex_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device)));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetVertexAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetVertexNormals());

    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(
            core::Tensor::Ones({2, 3}, core::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexPositions();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::Tensor& tl = mesh.GetVertexNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleIndices();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, ToString) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_colors =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_normals =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 2;
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetTriangleNormals(triangle_normals);

    std::string text =
            "TriangleMesh on " + device.ToString() +
            " [2 vertices (Float32) and 2 triangles (Int64)]."
            "\nVertex Attributes: colors (dtype = Float32, shape = {2, 3})."
            "\nTriangle Attributes: normals (dtype = Float32, shape = {2, "
            "3}).";

    EXPECT_STREQ(mesh.ToString().c_str(), text.c_str());

    mesh.RemoveVertexAttr("colors");
    mesh.RemoveTriangleAttr("normals");

    // Mesh with only primary attributes.
    std::string text_2 = "TriangleMesh on " + device.ToString() +
                         " [2 vertices (Float32) and 2 triangles (Int64)]."
                         "\nVertex Attributes: None."
                         "\nTriangle Attributes: None.";

    EXPECT_STREQ(mesh.ToString().c_str(), text_2.c_str());

    // Empty mesh.
    mesh.Clear();
    std::string text_3 = "TriangleMesh on " + device.ToString() +
                         " [0 vertices and 0 triangles].";
    EXPECT_STREQ(mesh.ToString().c_str(), text_3.c_str());
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // if device is a CUDA device.
    t::geometry::TriangleMesh mesh(device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::Tensor cpu_vertices =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device);
        core::Tensor cpu_colors =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 2;
        core::Tensor cpu_labels =
                core::Tensor::Ones({2, 3}, core::Float32, cpu_device) * 3;

        EXPECT_ANY_THROW(mesh.SetVertexPositions(cpu_vertices));
        EXPECT_ANY_THROW(mesh.SetVertexColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetVertexAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, RemoveAttr) {
    core::Device device = GetParam();

    core::Tensor vertices = core::Tensor::Ones({2, 3}, core::Float32, device);
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    core::Tensor triangles = core::Tensor::Ones({2, 3}, core::Int64, device);
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);

    mesh.SetVertexAttr("labels", vertex_labels);

    EXPECT_NO_THROW(mesh.GetVertexAttr("labels"));
    mesh.RemoveVertexAttr("labels");
    EXPECT_ANY_THROW(mesh.GetVertexAttr("labels"));

    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_NO_THROW(mesh.GetTriangleAttr("labels"));
    mesh.RemoveTriangleAttr("labels");
    EXPECT_ANY_THROW(mesh.GetTriangleAttr("labels"));

    // Not allowed to delete primary key attribute.
    EXPECT_ANY_THROW(mesh.RemoveVertexAttr("positions"));
    EXPECT_ANY_THROW(mesh.RemoveTriangleAttr("indices"));
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    EXPECT_FALSE(mesh.HasVertexPositions());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasVertexAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangleIndices());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetVertexPositions(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasVertexPositions());
    mesh.SetTriangleIndices(core::Tensor::Ones({10, 3}, core::Int64, device));
    EXPECT_TRUE(mesh.HasTriangleIndices());

    // Different size.
    mesh.SetVertexColors(core::Tensor::Ones({5, 3}, core::Float32, device));
    EXPECT_FALSE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::Tensor::Ones({5, 3}, core::Float32, device));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetVertexColors(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(core::Tensor::Ones({10, 3}, core::Float32, device));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

TEST_P(TriangleMeshPermuteDevices, Transform) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor transformation = core::Tensor::Init<float>(
            {{1, 1, 0, 1}, {0, 1, 1, 1}, {0, 1, 0, 1}, {0, 0, 0, 1}}, device);

    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Transform(transformation);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{2, 2, 1}, {2, 2, 1}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Translate) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor translation = core::Tensor::Init<float>({10, 20, 30}, device);

    // Relative.
    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Translate(translation, /*relative=*/true);

    // Normals do not translate.
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device)));

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{10, 21, 32}, {16, 27, 38}}, device)));

    // Non-relative.
    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{0, 1, 2}, {6, 7, 8}}, device));
    mesh.Translate(translation, /*relative=*/false);

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{7, 17, 27}, {13, 23, 33}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Scale) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);
    double scale = 4;

    mesh.SetVertexPositions(core::Tensor::Init<float>(
            {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}, device));

    mesh.Scale(scale, center);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(core::Tensor::Init<float>(
            {{-3, -3, -3}, {1, 1, 1}, {5, 5, 5}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, Rotate) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    core::Tensor rotation = core::Tensor::Init<float>(
            {{1, 1, 0}, {0, 1, 1}, {0, 1, 0}}, device);
    core::Tensor center = core::Tensor::Ones({3}, core::Dtype::Float32, device);

    mesh.SetVertexPositions(
            core::Tensor::Init<float>({{2, 2, 2}, {2, 2, 2}}, device));
    mesh.SetVertexNormals(
            core::Tensor::Init<float>({{1, 1, 1}, {1, 1, 1}}, device));

    mesh.Rotate(rotation, center);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Init<float>({{3, 3, 2}, {3, 3, 2}}, device)));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Init<float>({{2, 2, 1}, {2, 2, 1}}, device)));
}

TEST_P(TriangleMeshPermuteDevices, FromLegacy) {
    core::Device device = GetParam();
    geometry::TriangleMesh legacy_mesh;
    legacy_mesh.vertices_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)};
    legacy_mesh.vertex_colors_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)};
    legacy_mesh.vertex_normals_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(2, 2, 2), Eigen::Vector3d(2, 2, 2)};
    legacy_mesh.triangles_ = std::vector<Eigen::Vector3i>{
            Eigen::Vector3i(3, 3, 3), Eigen::Vector3i(3, 3, 3)};
    legacy_mesh.triangle_normals_ = std::vector<Eigen::Vector3d>{
            Eigen::Vector3d(4, 4, 4), Eigen::Vector3d(4, 4, 4)};
    legacy_mesh.triangle_uvs_ = std::vector<Eigen::Vector2d>{
            Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(0.2, 0.3),
            Eigen::Vector2d(0.4, 0.5), Eigen::Vector2d(0.6, 0.7),
            Eigen::Vector2d(0.8, 0.9), Eigen::Vector2d(1.0, 1.1)};

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;
    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::FromLegacy(
            legacy_mesh, float_dtype, int_dtype, device);

    EXPECT_TRUE(mesh.HasVertexPositions());
    EXPECT_TRUE(mesh.HasVertexColors());
    EXPECT_TRUE(mesh.HasVertexNormals());
    EXPECT_TRUE(mesh.HasTriangleIndices());
    EXPECT_TRUE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleColors());
    EXPECT_TRUE(mesh.HasTriangleAttr("texture_uvs"));
    EXPECT_FALSE(mesh.HasVertexAttr("texture_uvs"));

    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexPositions(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexPositions(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexColors(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetVertexNormals(), float_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetTriangleIndices(), int_dtype));
    EXPECT_NO_THROW(
            core::AssertTensorDtype(mesh.GetTriangleNormals(), float_dtype));
    EXPECT_NO_THROW(core::AssertTensorDtype(mesh.GetTriangleAttr("texture_uvs"),
                                            float_dtype));

    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 0));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 1));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(
            core::Tensor::Ones({2, 3}, int_dtype, device) * 3));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 4));
    EXPECT_TRUE(mesh.GetTriangleAttr("texture_uvs")
                        .AllClose(core::Tensor::Arange(0., 1.1, 0.1,
                                                       float_dtype, device)
                                          .Reshape({-1, 3, 2})));
}

TEST_P(TriangleMeshPermuteDevices, ToLegacy) {
    using ::testing::ElementsAreArray;
    using ::testing::FloatEq;
    using ::testing::Pointwise;
    core::Device device = GetParam();

    core::Dtype float_dtype = core::Float32;
    core::Dtype int_dtype = core::Int64;

    t::geometry::TriangleMesh mesh(device);
    mesh.SetVertexPositions(core::Tensor::Ones({2, 3}, float_dtype, device) *
                            0);
    mesh.SetVertexColors(core::Tensor::Ones({2, 3}, float_dtype, device) * 1);
    mesh.SetVertexNormals(core::Tensor::Ones({2, 3}, float_dtype, device) * 2);
    mesh.SetTriangleIndices(core::Tensor::Ones({2, 3}, int_dtype, device) * 3);
    mesh.SetTriangleNormals(core::Tensor::Ones({2, 3}, float_dtype, device) *
                            4);
    mesh.SetTriangleAttr("texture_uvs",
                         core::Tensor::Arange(0., 1.1, 0.1, float_dtype, device)
                                 .Reshape({-1, 3, 2}));

    geometry::TriangleMesh legacy_mesh = mesh.ToLegacy();
    EXPECT_EQ(legacy_mesh.vertices_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)}));
    EXPECT_EQ(legacy_mesh.vertex_colors_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(1, 1, 1)}));
    EXPECT_EQ(legacy_mesh.vertex_normals_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(2, 2, 2), Eigen::Vector3d(2, 2, 2)}));
    EXPECT_EQ(legacy_mesh.triangles_,
              std::vector<Eigen::Vector3i>(
                      {Eigen::Vector3i(3, 3, 3), Eigen::Vector3i(3, 3, 3)}));
    EXPECT_EQ(legacy_mesh.triangle_normals_,
              std::vector<Eigen::Vector3d>(
                      {Eigen::Vector3d(4, 4, 4), Eigen::Vector3d(4, 4, 4)}));
    EXPECT_THAT(legacy_mesh.triangle_uvs_,
                ElementsAreArray({Pointwise(FloatEq(), {0.0, 0.1}),
                                  Pointwise(FloatEq(), {0.2, 0.3}),
                                  Pointwise(FloatEq(), {0.4, 0.5}),
                                  Pointwise(FloatEq(), {0.6, 0.7}),
                                  Pointwise(FloatEq(), {0.8, 0.9}),
                                  Pointwise(FloatEq(), {1.0, 1.1})}));
}

TEST_P(TriangleMeshPermuteDevices, CreateBox) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with default parameters.
    t::geometry::TriangleMesh box_default =
            t::geometry::TriangleMesh::CreateBox();

    core::Tensor vertex_positions_default =
            core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                       {1.0, 0.0, 0.0},
                                       {0.0, 0.0, 1.0},
                                       {1.0, 0.0, 1.0},
                                       {0.0, 1.0, 0.0},
                                       {1.0, 1.0, 0.0},
                                       {0.0, 1.0, 1.0},
                                       {1.0, 1.0, 1.0}});

    core::Tensor triangle_indices_default =
            core::Tensor::Init<int64_t>({{4, 7, 5},
                                         {4, 6, 7},
                                         {0, 2, 4},
                                         {2, 6, 4},
                                         {0, 1, 2},
                                         {1, 3, 2},
                                         {1, 5, 7},
                                         {1, 7, 3},
                                         {2, 3, 7},
                                         {2, 7, 6},
                                         {0, 4, 1},
                                         {1, 4, 5}});

    EXPECT_TRUE(box_default.GetVertexPositions().AllClose(
            vertex_positions_default));
    EXPECT_TRUE(box_default.GetTriangleIndices().AllClose(
            triangle_indices_default));

    // Test with custom parameters.
    t::geometry::TriangleMesh box_custom = t::geometry::TriangleMesh::CreateBox(
            2, 3, 4, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {2.0, 0.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {2.0, 0.0, 4.0},
                                        {0.0, 3.0, 0.0},
                                        {2.0, 3.0, 0.0},
                                        {0.0, 3.0, 4.0},
                                        {2.0, 3.0, 4.0}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{4, 7, 5},
                                         {4, 6, 7},
                                         {0, 2, 4},
                                         {2, 6, 4},
                                         {0, 1, 2},
                                         {1, 3, 2},
                                         {1, 5, 7},
                                         {1, 7, 3},
                                         {2, 3, 7},
                                         {2, 7, 6},
                                         {0, 4, 1},
                                         {1, 4, 5}},
                                        device);

    EXPECT_TRUE(
            box_custom.GetVertexPositions().AllClose(vertex_positions_custom));
    EXPECT_TRUE(
            box_custom.GetTriangleIndices().AllClose(triangle_indices_custom));
}

}  // namespace tests
}  // namespace open3d
