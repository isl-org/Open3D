// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/TriangleMesh.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/CoreTest.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorCheck.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "open3d/visualization/utility/Draw.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevicesWithSYCL {};
INSTANTIATE_TEST_SUITE_P(
        TriangleMesh,
        TriangleMeshPermuteDevices,
        testing::ValuesIn(TriangleMeshPermuteDevices::TestCases()));

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
                         " [0 vertices and 0 triangles].\nVertex Attributes: "
                         "None.\nTriangle Attributes: None.";
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
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

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
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

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

TEST_P(TriangleMeshPermuteDevices, NormalizeNormals) {
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh =
            open3d::geometry::TriangleMesh::CreateSphere(1.0, 3);
    t::geometry::TriangleMesh t_mesh = t::geometry::TriangleMesh::FromLegacy(
            *mesh, core::Float64, core::Int64, device);

    mesh->ComputeTriangleNormals(false);
    mesh->NormalizeNormals();
    t_mesh.ComputeTriangleNormals(false);
    t_mesh.NormalizeNormals();

    EXPECT_TRUE(t_mesh.GetTriangleNormals().AllClose(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    mesh->triangle_normals_, core::Dtype::Float64, device)));
}

TEST_P(TriangleMeshPermuteDevices, ComputeTriangleNormals) {
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh =
            open3d::geometry::TriangleMesh::CreateSphere(1.0, 3);
    t::geometry::TriangleMesh t_mesh = t::geometry::TriangleMesh::FromLegacy(
            *mesh, core::Float64, core::Int64, device);

    mesh->ComputeTriangleNormals();
    t_mesh.ComputeTriangleNormals();
    EXPECT_TRUE(t_mesh.GetTriangleNormals().AllClose(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    mesh->triangle_normals_, core::Dtype::Float64, device)));
}

TEST_P(TriangleMeshPermuteDevices, ComputeVertexNormals) {
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    std::shared_ptr<open3d::geometry::TriangleMesh> mesh =
            open3d::geometry::TriangleMesh::CreateSphere(1.0, 3);
    t::geometry::TriangleMesh t_mesh = t::geometry::TriangleMesh::FromLegacy(
            *mesh, core::Float64, core::Int64, device);

    mesh->ComputeVertexNormals();
    t_mesh.ComputeVertexNormals();

    EXPECT_TRUE(t_mesh.GetVertexNormals().AllClose(
            core::eigen_converter::EigenVector3dVectorToTensor(
                    mesh->vertex_normals_, core::Dtype::Float64, device)));
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

    legacy_mesh.materials_.emplace_back();
    legacy_mesh.materials_.front().first = "Mat1";
    auto& mat = legacy_mesh.materials_.front().second;
    mat.baseColor = mat.baseColor.CreateRGB(1, 1, 1);

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
    EXPECT_TRUE(mesh.HasMaterial());
    EXPECT_TRUE(mesh.GetMaterial().GetBaseColor() ==
                Eigen::Vector4f(1, 1, 1, 1));
    EXPECT_TRUE(mesh.GetMaterial().GetBaseMetallic() == 0.0f);
    EXPECT_TRUE(mesh.GetMaterial().GetBaseRoughness() == 1.0f);
    EXPECT_TRUE(mesh.GetMaterial().GetBaseReflectance() == 0.5f);
    EXPECT_TRUE(mesh.GetMaterial().GetBaseClearcoat() == 0.0f);
    EXPECT_TRUE(mesh.GetMaterial().GetBaseClearcoatRoughness() == 0.0f);
    EXPECT_TRUE(mesh.GetMaterial().GetAnisotropy() == 0.0f);
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
    mesh.GetMaterial().SetDefaultProperties();

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

    auto mat_iterator = std::find_if(
            legacy_mesh.materials_.begin(), legacy_mesh.materials_.end(),
            [](const auto& pair) -> bool { return pair.first == "Mat1"; });
    EXPECT_TRUE(mat_iterator != legacy_mesh.materials_.end());
    auto& mat = mat_iterator->second;
    EXPECT_TRUE(Eigen::Vector4f(mat.baseColor.f4) ==
                Eigen::Vector4f(1, 1, 1, 1));
    EXPECT_TRUE(mat.baseMetallic == 0.0);
    EXPECT_TRUE(mat.baseRoughness == 1.0);
    EXPECT_TRUE(mat.baseReflectance == 0.5);
    EXPECT_TRUE(mat.baseClearCoat == 0.0);
    EXPECT_TRUE(mat.baseClearCoatRoughness == 0.0);
    EXPECT_TRUE(mat.baseAnisotropy == 0.0);
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

TEST_P(TriangleMeshPermuteDevices, CreateSphere) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh sphere_custom =
            t::geometry::TriangleMesh::CreateSphere(1, 3, float_dtype_custom,
                                                    int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 1.0},
                                        {0.0, 0.0, -1.0},
                                        {0.866025, 0, 0.5},
                                        {0.433013, 0.75, 0.5},
                                        {-0.433013, 0.75, 0.5},
                                        {-0.866025, 0.0, 0.5},
                                        {-0.433013, -0.75, 0.5},
                                        {0.433013, -0.75, 0.5},
                                        {0.866025, 0.0, -0.5},
                                        {0.433013, 0.75, -0.5},
                                        {-0.433013, 0.75, -0.5},
                                        {-0.866025, 0.0, -0.5},
                                        {-0.433013, -0.75, -0.5},
                                        {0.433013, -0.75, -0.5}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},   {1, 9, 8},   {0, 3, 4},   {1, 10, 9}, {0, 4, 5},
             {1, 11, 10}, {0, 5, 6},   {1, 12, 11}, {0, 6, 7},  {1, 13, 12},
             {0, 7, 2},   {1, 8, 13},  {8, 3, 2},   {8, 9, 3},  {9, 4, 3},
             {9, 10, 4},  {10, 5, 4},  {10, 11, 5}, {11, 6, 5}, {11, 12, 6},
             {12, 7, 6},  {12, 13, 7}, {13, 2, 7},  {13, 8, 2}},
            device);

    EXPECT_TRUE(sphere_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(sphere_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateTetrahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh tetrahedron_custom =
            t::geometry::TriangleMesh::CreateTetrahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{1.88562, 0.0, -0.666667},
                                        {-0.942809, 1.63299, -0.666667},
                                        {-0.942809, -1.63299, -0.666667},
                                        {0.0, 0.0, 2}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 1}, {0, 3, 2}, {0, 1, 3}, {1, 2, 3}}, device);
    EXPECT_TRUE(tetrahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(tetrahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateOctahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh octahedron_custom =
            t::geometry::TriangleMesh::CreateOctahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{2.0, 0.0, 0.0},
                                        {0.0, 2.0, 0.0},
                                        {0.0, 0.0, 2.0},
                                        {-2.0, 0.0, 0.0},
                                        {0.0, -2.0, 0.0},
                                        {0.0, 0.0, -2.0}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{0, 1, 2},
                                         {1, 3, 2},
                                         {3, 4, 2},
                                         {4, 0, 2},
                                         {0, 5, 1},
                                         {1, 5, 3},
                                         {3, 5, 4},
                                         {4, 5, 0}},
                                        device);
    EXPECT_TRUE(octahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(octahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateIcosahedron) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh icosahedron_custom =
            t::geometry::TriangleMesh::CreateIcosahedron(
                    2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{-2.0, 0.0, 3.23607},
                                        {2.0, 0.0, 3.23607},
                                        {2.0, 0.0, -3.23607},
                                        {-2.0, 0.0, -3.23607},
                                        {0.0, -3.23607, 2.0},
                                        {0.0, 3.23607, 2.0},
                                        {0.0, 3.23607, -2.0},
                                        {0.0, -3.23607, -2.0},
                                        {-3.23607, -2.0, 0.0},
                                        {3.23607, -2.0, 0.0},
                                        {3.23607, 2.0, 0.0},
                                        {-3.23607, 2.0, 0.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 4, 1},  {0, 1, 5},  {1, 4, 9},  {1, 9, 10}, {1, 10, 5},
             {0, 8, 4},  {0, 11, 8}, {0, 5, 11}, {5, 6, 11}, {5, 10, 6},
             {4, 8, 7},  {4, 7, 9},  {3, 6, 2},  {3, 2, 7},  {2, 6, 10},
             {2, 10, 9}, {2, 9, 7},  {3, 11, 6}, {3, 8, 11}, {3, 7, 8}},
            device);
    EXPECT_TRUE(icosahedron_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(icosahedron_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateCylinder) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh cylinder_custom =
            t::geometry::TriangleMesh::CreateCylinder(
                    1, 2, 3, 3, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 1.0},
                                        {0.0, 0.0, -1.0},
                                        {1.0, 0.0, 1.0},
                                        {-0.5, 0.866025, 1.0},
                                        {-0.5, -0.866025, 1.0},
                                        {1.0, 0.0, 0.333333},
                                        {-0.5, 0.866025, 0.333333},
                                        {-0.5, -0.866025, 0.333333},
                                        {1.0, 0.0, -0.333333},
                                        {-0.5, 0.866025, -0.333333},
                                        {-0.5, -0.866025, -0.333333},
                                        {1.0, 0.0, -1.0},
                                        {-0.5, 0.866025, -1.0},
                                        {-0.5, -0.866025, -1.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},   {1, 12, 11},  {0, 3, 4},   {1, 13, 12}, {0, 4, 2},
             {1, 11, 13}, {5, 3, 2},    {5, 6, 3},   {6, 4, 3},   {6, 7, 4},
             {7, 2, 4},   {7, 5, 2},    {8, 6, 5},   {8, 9, 6},   {9, 7, 6},
             {9, 10, 7},  {10, 5, 7},   {10, 8, 5},  {11, 9, 8},  {11, 12, 9},
             {12, 10, 9}, {12, 13, 10}, {13, 8, 10}, {13, 11, 8}},
            device);
    EXPECT_TRUE(cylinder_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(cylinder_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateCone) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh cone_custom =
            t::geometry::TriangleMesh::CreateCone(
                    2, 4, 3, 2, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {2.0, 0.0, 0.0},
                                        {-1.0, 1.73205, 0.0},
                                        {-1.0, -1.73205, 0.0},
                                        {1.0, 0.0, 2.0},
                                        {-0.5, 0.866025, 2},
                                        {-0.5, -0.866025, 2}},
                                       device);

    core::Tensor triangle_indices_custom =
            core::Tensor::Init<int32_t>({{0, 3, 2},
                                         {1, 5, 6},
                                         {0, 4, 3},
                                         {1, 6, 7},
                                         {0, 2, 4},
                                         {1, 7, 5},
                                         {6, 2, 3},
                                         {6, 5, 2},
                                         {7, 3, 4},
                                         {7, 6, 3},
                                         {5, 4, 2},
                                         {5, 7, 4}},
                                        device);
    EXPECT_TRUE(
            cone_custom.GetVertexPositions().AllClose(vertex_positions_custom));
    EXPECT_TRUE(
            cone_custom.GetTriangleIndices().AllClose(triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateTorus) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh torus_custom =
            t::geometry::TriangleMesh::CreateTorus(
                    2, 1, 6, 3, float_dtype_custom, int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{3.0, 0.0, 0.0},
                                        {1.5, 0.0, 0.866025},
                                        {1.5, 0.0, -0.866025},
                                        {1.5, 2.59808, 0.0},
                                        {0.75, 1.29904, 0.866025},
                                        {0.75, 1.29904, -0.866025},
                                        {-1.5, 2.59808, 0},
                                        {-0.75, 1.29904, 0.866025},
                                        {-0.75, 1.29904, -0.866025},
                                        {-3.0, 0.0, 0.0},
                                        {-1.5, 0.0, 0.866025},
                                        {-1.5, 0.0, -0.866025},
                                        {-1.5, -2.59808, 0.0},
                                        {-0.75, -1.29904, 0.866025},
                                        {-0.75, -1.29904, -0.866025},
                                        {1.5, -2.59808, 0.0},
                                        {0.75, -1.29904, 0.866025},
                                        {0.75, -1.29904, -0.866025}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{3, 4, 0},    {0, 4, 1},    {4, 5, 1},    {1, 5, 2},
             {5, 3, 2},    {2, 3, 0},    {6, 7, 3},    {3, 7, 4},
             {7, 8, 4},    {4, 8, 5},    {8, 6, 5},    {5, 6, 3},
             {9, 10, 6},   {6, 10, 7},   {10, 11, 7},  {7, 11, 8},
             {11, 9, 8},   {8, 9, 6},    {12, 13, 9},  {9, 13, 10},
             {13, 14, 10}, {10, 14, 11}, {14, 12, 11}, {11, 12, 9},
             {15, 16, 12}, {12, 16, 13}, {16, 17, 13}, {13, 17, 14},
             {17, 15, 14}, {14, 15, 12}, {0, 1, 15},   {15, 1, 16},
             {1, 2, 16},   {16, 2, 17},  {2, 0, 17},   {17, 0, 15}},
            device);
    EXPECT_TRUE(torus_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(torus_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateArrow) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh arrow_custom =
            t::geometry::TriangleMesh::CreateArrow(1, 2, 4, 2, 4, 1, 1,
                                                   float_dtype_custom,
                                                   int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.0, 0.0, 4.0},
                                        {0.0, 0.0, 0.0},
                                        {1.0, 0.0, 4.0},
                                        {0.0, 1.0, 4.0},
                                        {-1.0, 0.0, 4.0},
                                        {0.0, -1.0, 4.0},
                                        {1.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0},
                                        {-1.0, 0.0, 0.0},
                                        {0.0, -1.0, 0.0},
                                        {0.0, 0.0, 4.0},
                                        {0.0, 0.0, 6.0},
                                        {2.0, 0.0, 4.0},
                                        {0.0, 2.0, 4.0},
                                        {-2.0, 0.0, 4.0},
                                        {0.0, -2.0, 4.0}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 2, 3},    {1, 7, 6},    {0, 3, 4},    {1, 8, 7},
             {0, 4, 5},    {1, 9, 8},    {0, 5, 2},    {1, 6, 9},
             {6, 3, 2},    {6, 7, 3},    {7, 4, 3},    {7, 8, 4},
             {8, 5, 4},    {8, 9, 5},    {9, 2, 5},    {9, 6, 2},
             {10, 13, 12}, {11, 12, 13}, {10, 14, 13}, {11, 13, 14},
             {10, 15, 14}, {11, 14, 15}, {10, 12, 15}, {11, 15, 12}},
            device);
    EXPECT_TRUE(arrow_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(arrow_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, CreateMobius) {
    core::Device device = GetParam();
    core::Dtype float_dtype_custom = core::Float64;
    core::Dtype int_dtype_custom = core::Int32;

    // Test with custom parameters.
    t::geometry::TriangleMesh mobius_custom =
            t::geometry::TriangleMesh::CreateMobius(10, 2, 1, 1, 1, 1, 1,
                                                    float_dtype_custom,
                                                    int_dtype_custom, device);

    core::Tensor vertex_positions_custom =
            core::Tensor::Init<double>({{0.5, 0.0, 0.0},
                                        {1.5, 0.0, 0.0},
                                        {0.424307, 0.308277, -0.154508},
                                        {1.19373, 0.867294, 0.154508},
                                        {0.184017, 0.566346, -0.293893},
                                        {0.434017, 1.33577, 0.293893},
                                        {-0.218199, 0.671548, -0.404508},
                                        {-0.399835, 1.23057, 0.404508},
                                        {-0.684017, 0.496967, -0.475528},
                                        {-0.934017, 0.678603, 0.475528},
                                        {-1.0, 0.0, -0.5},
                                        {-1.0, 0.0, 0.5},
                                        {-0.934017, -0.678603, -0.475528},
                                        {-0.684017, -0.496967, 0.475528},
                                        {-0.399835, -1.23057, -0.404508},
                                        {-0.218199, -0.671548, 0.404508},
                                        {0.434017, -1.33577, -0.293893},
                                        {0.184017, -0.566346, 0.293893},
                                        {1.19373, -0.867294, -0.154508},
                                        {0.424307, -0.308277, 0.154508}},
                                       device);

    core::Tensor triangle_indices_custom = core::Tensor::Init<int32_t>(
            {{0, 3, 1},    {0, 2, 3},    {3, 2, 4},    {3, 4, 5},
             {4, 7, 5},    {4, 6, 7},    {7, 6, 8},    {7, 8, 9},
             {8, 11, 9},   {8, 10, 11},  {11, 10, 12}, {11, 12, 13},
             {12, 15, 13}, {12, 14, 15}, {15, 14, 16}, {15, 16, 17},
             {16, 19, 17}, {16, 18, 19}, {18, 19, 1},  {1, 19, 0}},
            device);
    EXPECT_TRUE(mobius_custom.GetVertexPositions().AllClose(
            vertex_positions_custom));
    EXPECT_TRUE(mobius_custom.GetTriangleIndices().AllClose(
            triangle_indices_custom));
}

TEST_P(TriangleMeshPermuteDevices, SelectFacesByMask) {
    // check that an exception is thrown if the mesh is empty
    t::geometry::TriangleMesh mesh_empty;
    core::Tensor mask_empty =
            core::Tensor::Zeros({12}, core::Bool, mesh_empty.GetDevice());
    core::Tensor mask_full =
            core::Tensor::Ones({12}, core::Bool, mesh_empty.GetDevice());

    // check completely empty mesh
    EXPECT_TRUE(mesh_empty.SelectFacesByMask(mask_empty).IsEmpty());
    EXPECT_TRUE(mesh_empty.SelectFacesByMask(mask_full).IsEmpty());

    // check mesh w/o triangles
    core::Tensor cpu_vertices =
            core::Tensor::Ones({2, 3}, core::Float32, mesh_empty.GetDevice());
    mesh_empty.SetVertexPositions(cpu_vertices);
    EXPECT_TRUE(mesh_empty.SelectFacesByMask(mask_empty).IsEmpty());
    EXPECT_TRUE(mesh_empty.SelectFacesByMask(mask_full).IsEmpty());

    // create box with normals, colors and labels defined.
    t::geometry::TriangleMesh box = t::geometry::TriangleMesh::CreateBox();
    core::Tensor vertex_colors = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                            {1.0, 1.0, 1.0},
                                                            {2.0, 2.0, 2.0},
                                                            {3.0, 3.0, 3.0},
                                                            {4.0, 4.0, 4.0},
                                                            {5.0, 5.0, 5.0},
                                                            {6.0, 6.0, 6.0},
                                                            {7.0, 7.0, 7.0}});
    ;
    core::Tensor vertex_labels = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                            {1.0, 1.0, 1.0},
                                                            {2.0, 2.0, 2.0},
                                                            {3.0, 3.0, 3.0},
                                                            {4.0, 4.0, 4.0},
                                                            {5.0, 5.0, 5.0},
                                                            {6.0, 6.0, 6.0},
                                                            {7.0, 7.0, 7.0}}) *
                                 10;
    ;
    core::Tensor triangle_labels =
            core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                       {1.0, 1.0, 1.0},
                                       {2.0, 2.0, 2.0},
                                       {3.0, 3.0, 3.0},
                                       {4.0, 4.0, 4.0},
                                       {5.0, 5.0, 5.0},
                                       {6.0, 6.0, 6.0},
                                       {7.0, 7.0, 7.0},
                                       {8.0, 8.0, 8.0},
                                       {9.0, 9.0, 9.0},
                                       {10.0, 10.0, 10.0},
                                       {11.0, 11.0, 11.0}}) *
            100;
    box.SetVertexColors(vertex_colors);
    box.SetVertexAttr("labels", vertex_labels);
    box.ComputeTriangleNormals();
    box.SetTriangleAttr("labels", triangle_labels);

    // empty index list
    EXPECT_TRUE(box.SelectFacesByMask(mask_empty).IsEmpty());

    // set the expected value
    core::Tensor expected_verts = core::Tensor::Init<float>({{0.0, 0.0, 1.0},
                                                             {1.0, 0.0, 1.0},
                                                             {0.0, 1.0, 1.0},
                                                             {1.0, 1.0, 1.0}});
    core::Tensor expected_vert_colors =
            core::Tensor::Init<float>({{2.0, 2.0, 2.0},
                                       {3.0, 3.0, 3.0},
                                       {6.0, 6.0, 6.0},
                                       {7.0, 7.0, 7.0}});
    core::Tensor expected_vert_labels =
            core::Tensor::Init<float>({{20.0, 20.0, 20.0},
                                       {30.0, 30.0, 30.0},
                                       {60.0, 60.0, 60.0},
                                       {70.0, 70.0, 70.0}});
    core::Tensor expected_tris =
            core::Tensor::Init<int64_t>({{0, 1, 3}, {0, 3, 2}});
    core::Tensor tris_mask =
            core::Tensor::Init<bool>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0});
    core::Tensor expected_tri_normals =
            box.GetTriangleNormals().IndexGet({tris_mask});
    core::Tensor expected_tri_labels = core::Tensor::Init<float>(
            {{800.0, 800.0, 800.0}, {900.0, 900.0, 900.0}});

    // check basic case
    t::geometry::TriangleMesh selected = box.SelectFacesByMask(tris_mask);

    EXPECT_TRUE(selected.GetVertexPositions().AllClose(expected_verts));
    EXPECT_TRUE(selected.GetVertexColors().AllClose(expected_vert_colors));
    EXPECT_TRUE(
            selected.GetVertexAttr("labels").AllClose(expected_vert_labels));
    EXPECT_TRUE(selected.GetTriangleIndices().AllClose(expected_tris));
    EXPECT_TRUE(selected.GetTriangleNormals().AllClose(expected_tri_normals));
    EXPECT_TRUE(
            selected.GetTriangleAttr("labels").AllClose(expected_tri_labels));

    // Check that initial mesh is unchanged.
    t::geometry::TriangleMesh box_untouched =
            t::geometry::TriangleMesh::CreateBox();
    EXPECT_TRUE(box.GetVertexPositions().AllClose(
            box_untouched.GetVertexPositions()));
    EXPECT_TRUE(box.GetTriangleIndices().AllClose(
            box_untouched.GetTriangleIndices()));
}

TEST_P(TriangleMeshPermuteDevices, SelectByIndex) {
    // check that an exception is thrown if the mesh is empty
    t::geometry::TriangleMesh mesh_empty;
    core::Tensor indices_empty = core::Tensor::Init<int64_t>({});

    // check completely empty mesh
    EXPECT_TRUE(mesh_empty.SelectByIndex(indices_empty).IsEmpty());
    EXPECT_TRUE(mesh_empty.SelectByIndex(core::Tensor::Init<int64_t>({0}))
                        .IsEmpty());

    // check mesh w/o triangles
    core::Tensor vertices_no_tris_orig =
            core::Tensor::Ones({2, 3}, core::Float32, mesh_empty.GetDevice());
    core::Tensor expected_vertices_no_tris_orig =
            core::Tensor::Ones({1, 3}, core::Float32, mesh_empty.GetDevice());
    mesh_empty.SetVertexPositions(vertices_no_tris_orig);
    t::geometry::TriangleMesh selected_no_tris_orig =
            mesh_empty.SelectByIndex(core::Tensor::Init<int64_t>({0}));
    EXPECT_TRUE(selected_no_tris_orig.GetVertexPositions().AllClose(
            expected_vertices_no_tris_orig));

    // create box with normals, colors and labels defined.
    t::geometry::TriangleMesh box = t::geometry::TriangleMesh::CreateBox();
    core::Tensor vertex_colors = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                            {1.0, 1.0, 1.0},
                                                            {2.0, 2.0, 2.0},
                                                            {3.0, 3.0, 3.0},
                                                            {4.0, 4.0, 4.0},
                                                            {5.0, 5.0, 5.0},
                                                            {6.0, 6.0, 6.0},
                                                            {7.0, 7.0, 7.0}});
    ;
    core::Tensor vertex_labels = core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                                            {1.0, 1.0, 1.0},
                                                            {2.0, 2.0, 2.0},
                                                            {3.0, 3.0, 3.0},
                                                            {4.0, 4.0, 4.0},
                                                            {5.0, 5.0, 5.0},
                                                            {6.0, 6.0, 6.0},
                                                            {7.0, 7.0, 7.0}}) *
                                 10;
    ;
    core::Tensor triangle_labels =
            core::Tensor::Init<float>({{0.0, 0.0, 0.0},
                                       {1.0, 1.0, 1.0},
                                       {2.0, 2.0, 2.0},
                                       {3.0, 3.0, 3.0},
                                       {4.0, 4.0, 4.0},
                                       {5.0, 5.0, 5.0},
                                       {6.0, 6.0, 6.0},
                                       {7.0, 7.0, 7.0},
                                       {8.0, 8.0, 8.0},
                                       {9.0, 9.0, 9.0},
                                       {10.0, 10.0, 10.0},
                                       {11.0, 11.0, 11.0}}) *
            100;
    box.SetVertexColors(vertex_colors);
    box.SetVertexAttr("labels", vertex_labels);
    box.ComputeTriangleNormals();
    box.SetTriangleAttr("labels", triangle_labels);

    // empty index list
    EXPECT_TRUE(box.SelectByIndex(indices_empty).IsEmpty());

    // set the expected value
    core::Tensor expected_verts = core::Tensor::Init<float>({{0.0, 0.0, 1.0},
                                                             {1.0, 0.0, 1.0},
                                                             {0.0, 1.0, 1.0},
                                                             {1.0, 1.0, 1.0}});
    core::Tensor expected_vert_colors =
            core::Tensor::Init<float>({{2.0, 2.0, 2.0},
                                       {3.0, 3.0, 3.0},
                                       {6.0, 6.0, 6.0},
                                       {7.0, 7.0, 7.0}});
    core::Tensor expected_vert_labels =
            core::Tensor::Init<float>({{20.0, 20.0, 20.0},
                                       {30.0, 30.0, 30.0},
                                       {60.0, 60.0, 60.0},
                                       {70.0, 70.0, 70.0}});
    core::Tensor expected_tris =
            core::Tensor::Init<int64_t>({{0, 1, 3}, {0, 3, 2}});
    core::Tensor tris_mask =
            core::Tensor::Init<bool>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0});
    core::Tensor expected_tri_normals =
            box.GetTriangleNormals().IndexGet({tris_mask});
    core::Tensor expected_tri_labels = core::Tensor::Init<float>(
            {{800.0, 800.0, 800.0}, {900.0, 900.0, 900.0}});

    // check basic case
    core::Tensor indices = core::Tensor::Init<int64_t>({2, 3, 6, 7});
    t::geometry::TriangleMesh selected_basic = box.SelectByIndex(indices);

    EXPECT_TRUE(selected_basic.GetVertexPositions().AllClose(expected_verts));
    EXPECT_TRUE(
            selected_basic.GetVertexColors().AllClose(expected_vert_colors));
    EXPECT_TRUE(selected_basic.GetVertexAttr("labels").AllClose(
            expected_vert_labels));
    EXPECT_TRUE(selected_basic.GetTriangleIndices().AllClose(expected_tris));
    EXPECT_TRUE(
            selected_basic.GetTriangleNormals().AllClose(expected_tri_normals));
    EXPECT_TRUE(selected_basic.GetTriangleAttr("labels").AllClose(
            expected_tri_labels));

    // check duplicated indices case
    core::Tensor indices_duplicate =
            core::Tensor::Init<int16_t>({2, 2, 3, 3, 6, 7, 7});
    t::geometry::TriangleMesh selected_duplicate =
            box.SelectByIndex(indices_duplicate);
    EXPECT_TRUE(
            selected_duplicate.GetVertexPositions().AllClose(expected_verts));
    EXPECT_TRUE(selected_duplicate.GetVertexColors().AllClose(
            expected_vert_colors));
    EXPECT_TRUE(selected_duplicate.GetVertexAttr("labels").AllClose(
            expected_vert_labels));
    EXPECT_TRUE(
            selected_duplicate.GetTriangleIndices().AllClose(expected_tris));
    EXPECT_TRUE(selected_duplicate.GetTriangleNormals().AllClose(
            expected_tri_normals));
    EXPECT_TRUE(selected_duplicate.GetTriangleAttr("labels").AllClose(
            expected_tri_labels));

    core::Tensor indices_negative =
            core::Tensor::Init<int64_t>({2, -4, 3, 6, 7});
    t::geometry::TriangleMesh selected_negative =
            box.SelectByIndex(indices_negative);
    EXPECT_TRUE(
            selected_negative.GetVertexPositions().AllClose(expected_verts));
    EXPECT_TRUE(selected_negative.GetTriangleIndices().AllClose(expected_tris));

    // select with empty triangles as result
    // set the expected value
    core::Tensor expected_verts_no_tris = core::Tensor::Init<float>(
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}});
    core::Tensor expected_vert_colors_no_tris = core::Tensor::Init<float>(
            {{0.0, 0.0, 0.0}, {3.0, 3.0, 3.0}, {4.0, 4.0, 4.0}});
    core::Tensor expected_vert_labels_no_tris = core::Tensor::Init<float>(
            {{0.0, 0.0, 0.0}, {30.0, 30.0, 30.0}, {40.0, 40.0, 40.0}});

    core::Tensor indices_no_tris = core::Tensor::Init<int64_t>({0, 3, 4});
    t::geometry::TriangleMesh selected_no_tris =
            box.SelectByIndex(indices_no_tris);

    EXPECT_TRUE(selected_no_tris.GetVertexPositions().AllClose(
            expected_verts_no_tris));
    EXPECT_TRUE(selected_no_tris.GetVertexColors().AllClose(
            expected_vert_colors_no_tris));
    EXPECT_TRUE(selected_no_tris.GetVertexAttr("labels").AllClose(
            expected_vert_labels_no_tris));
    EXPECT_FALSE(selected_no_tris.HasTriangleIndices());

    // check that initial mesh is unchanged
    t::geometry::TriangleMesh box_untouched =
            t::geometry::TriangleMesh::CreateBox();
    EXPECT_TRUE(box.GetVertexPositions().AllClose(
            box_untouched.GetVertexPositions()));
    EXPECT_TRUE(box.GetTriangleIndices().AllClose(
            box_untouched.GetTriangleIndices()));
}

TEST_P(TriangleMeshPermuteDevices, RemoveUnreferencedVertices) {
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    t::geometry::TriangleMesh mesh_empty{device};

    // check completely empty mesh
    EXPECT_TRUE(mesh_empty.RemoveUnreferencedVertices().IsEmpty());

    // check mesh w/o triangles
    core::Tensor vertices_no_tris_orig =
            core::Tensor::Ones({2, 3}, core::Float32, device);
    mesh_empty.SetVertexPositions(vertices_no_tris_orig);
    EXPECT_TRUE(mesh_empty.RemoveUnreferencedVertices().IsEmpty());

    // Torus
    core::Tensor verts = core::Tensor::Init<double>(
            {
                    {0, 0, 0}, /* 0 */
                    {3.0, 0.0, 0.0},
                    {1.5, 0.0, 0.866025},
                    {1, 2, 3}, /* 3 */
                    {1.5, 0.0, -0.866025},
                    {1.5, 2.59808, 0.0},
                    {0.75, 1.29904, 0.866025},
                    {0.75, 1.29904, -0.866025},
                    {-1.5, 2.59808, 0},
                    {-0.75, 1.29904, 0.866025},
                    {-0.75, 1.29904, -0.866025},
                    {-3.0, 0.0, 0.0},
                    {-1.5, 0.0, 0.866025},
                    {-1.5, 0.0, -0.866025},
                    {-1.5, -2.59808, 0.0},
                    {-0.75, -1.29904, 0.866025},
                    {-0.75, -1.29904, -0.866025},
                    {4, 5, 6}, /* 17 */
                    {1.5, -2.59808, 0.0},
                    {0.75, -1.29904, 0.866025},
                    {0.75, -1.29904, -0.866025},
                    {7, 8, 9} /* 21 */
            },
            device);

    core::Tensor tris = core::Tensor::Init<int32_t>(
            {{5, 6, 1},    {1, 6, 2},    {6, 7, 2},    {2, 7, 4},
             {7, 5, 4},    {4, 5, 1},    {8, 9, 5},    {5, 9, 6},
             {9, 10, 6},   {6, 10, 7},   {10, 8, 7},   {7, 8, 5},
             {11, 12, 8},  {8, 12, 9},   {12, 13, 9},  {9, 13, 10},
             {13, 11, 10}, {10, 11, 8},  {14, 15, 11}, {11, 15, 12},
             {15, 16, 12}, {12, 16, 13}, {16, 14, 13}, {13, 14, 11},
             {18, 19, 14}, {14, 19, 15}, {19, 20, 15}, {15, 20, 16},
             {20, 18, 16}, {16, 18, 14}, {1, 2, 18},   {18, 2, 19},
             {2, 4, 19},   {19, 4, 20},  {4, 1, 20},   {20, 1, 18}},
            device);
    t::geometry::TriangleMesh torus{verts, tris};
    core::Tensor vertex_colors = core::Tensor::Init<float>(
            {{0.0, 0.0, 0.0},    {1.0, 1.0, 1.0},    {2.0, 2.0, 2.0},
             {3.0, 3.0, 3.0},    {4.0, 4.0, 4.0},    {5.0, 5.0, 5.0},
             {6.0, 6.0, 6.0},    {7.0, 7.0, 7.0},    {8.0, 8.0, 8.0},
             {9.0, 9.0, 9.0},    {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0},
             {12.0, 12.0, 12.0}, {13.0, 13.0, 13.0}, {14.0, 14.0, 14.0},
             {15.0, 15.0, 15.0}, {16.0, 16.0, 16.0}, {17.0, 17.0, 17.0},
             {18.0, 18.0, 18.0}, {19.0, 19.0, 19.0}, {20.0, 20.0, 20.0},
             {21.0, 21.0, 21.0}},
            device);
    core::Tensor vertex_labels =
            core::Tensor::Init<float>(
                    {{0.0, 0.0, 0.0},    {1.0, 1.0, 1.0},    {2.0, 2.0, 2.0},
                     {3.0, 3.0, 3.0},    {4.0, 4.0, 4.0},    {5.0, 5.0, 5.0},
                     {6.0, 6.0, 6.0},    {7.0, 7.0, 7.0},    {8.0, 8.0, 8.0},
                     {9.0, 9.0, 9.0},    {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0},
                     {12.0, 12.0, 12.0}, {13.0, 13.0, 13.0}, {14.0, 14.0, 14.0},
                     {15.0, 15.0, 15.0}, {16.0, 16.0, 16.0}, {17.0, 17.0, 17.0},
                     {18.0, 18.0, 18.0}, {19.0, 19.0, 19.0}, {20.0, 20.0, 20.0},
                     {21.0, 21.0, 21.0}},
                    device) *
            10;

    core::Tensor triangle_labels =
            core::Tensor::Init<float>({{0.0, 0.0, 0.0},    {1.0, 1.0, 1.0},
                                       {2.0, 2.0, 2.0},    {3.0, 3.0, 3.0},
                                       {4.0, 4.0, 4.0},    {5.0, 5.0, 5.0},
                                       {6.0, 6.0, 6.0},    {7.0, 7.0, 7.0},
                                       {8.0, 8.0, 8.0},    {9.0, 9.0, 9.0},
                                       {10.0, 10.0, 10.0}, {11.0, 11.0, 11.0},
                                       {12.0, 12.0, 12.0}, {13.0, 13.0, 13.0},
                                       {14.0, 14.0, 14.0}, {15.0, 15.0, 15.0},
                                       {16.0, 16.0, 16.0}, {17.0, 17.0, 17.0},
                                       {18.0, 18.0, 18.0}, {19.0, 19.0, 19.0},
                                       {20.0, 20.0, 20.0}, {21.0, 21.0, 21.0},
                                       {22.0, 22.0, 22.0}, {23.0, 23.0, 23.0},
                                       {24.0, 24.0, 24.0}, {25.0, 25.0, 25.0},
                                       {26.0, 26.0, 26.0}, {27.0, 27.0, 27.0},
                                       {28.0, 28.0, 28.0}, {29.0, 29.0, 29.0},
                                       {30.0, 30.0, 30.0}, {31.0, 31.0, 31.0},
                                       {32.0, 32.0, 32.0}, {33.0, 33.0, 33.0},
                                       {34.0, 34.0, 34.0}, {35.0, 35.0, 35.0}},
                                      device) *
            100;
    torus.SetVertexColors(vertex_colors);
    torus.SetVertexAttr("labels", vertex_labels);
    torus.ComputeVertexNormals();
    torus.ComputeTriangleNormals();

    // set the expected value
    core::Tensor verts_mask = core::Tensor::Init<bool>(
            {0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0},
            device);
    core::Tensor expected_verts =
            torus.GetVertexPositions().IndexGet({verts_mask});
    core::Tensor expected_tris =
            t::geometry::TriangleMesh::CreateTorus(2, 1, 6, 3, core::Float32,
                                                   core::Int32, device)
                    .GetTriangleIndices();
    core::Tensor expected_vert_normals =
            torus.GetVertexNormals().IndexGet({verts_mask});
    core::Tensor expected_tri_normals = torus.GetTriangleNormals().Clone();
    core::Tensor expected_vert_labels =
            torus.GetVertexAttr("labels").IndexGet({verts_mask});
    core::Tensor expected_vert_colors =
            torus.GetVertexAttr("colors").IndexGet({verts_mask});

    torus.RemoveUnreferencedVertices();

    EXPECT_TRUE(torus.GetVertexPositions().AllClose(expected_verts));
    EXPECT_TRUE(torus.GetVertexNormals().AllClose(expected_vert_normals));
    EXPECT_TRUE(torus.GetVertexColors().AllClose(expected_vert_colors));
    EXPECT_TRUE(torus.GetVertexAttr("labels").AllClose(expected_vert_labels));
    EXPECT_TRUE(torus.GetTriangleIndices().AllClose(expected_tris));
    EXPECT_TRUE(torus.GetTriangleNormals().AllClose(expected_tri_normals));
}

TEST_P(TriangleMeshPermuteDevices, ProjectImagesToAlbedo) {
    using namespace t::geometry;
    using ::testing::AnyOf;
    using ::testing::ElementsAre;
    using ::testing::FloatEq;
    core::Device device = GetParam();
    if (!device.IsCPU() || !Image::HAVE_IPP) GTEST_SKIP() << "Not Implemented!";
    TriangleMesh sphere =
            TriangleMesh::FromLegacy(*geometry::TriangleMesh::CreateSphere(
                    1.0, 20, /*create_uv_map=*/true));
    core::Tensor view[3] = {core::Tensor::Zeros({192, 256, 3}, core::Float32),
                            core::Tensor::Zeros({192, 256, 3}, core::Float32),
                            core::Tensor::Zeros({192, 256, 3}, core::Float32)};
    view[0].Slice(2, 0, 1, 1).Fill(1.0);  // red
    view[1].Slice(2, 1, 2, 1).Fill(1.0);  // green
    view[2].Slice(2, 2, 3, 1).Fill(1.0);  // blue
    core::Tensor intrinsic_matrix = core::Tensor::Init<float>(
            {{256, 0, 128}, {0, 256, 96}, {0, 0, 1}}, device);
    core::Tensor extrinsic_matrix[3] = {
            core::Tensor::Init<float>(
                    {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 3}, {0, 0, 0, 1}},
                    device),
            core::Tensor::Init<float>({{-0.5, 0, -0.8660254, 0},
                                       {0, 1, 0, 0},
                                       {0.8660254, 0, -0.5, 3},
                                       {0, 0, 0, 1}},
                                      device),
            core::Tensor::Init<float>({{-0.5, 0, 0.8660254, 0},
                                       {0, 1, 0, 0},
                                       {-0.8660254, 0, -0.5, 3},
                                       {0, 0, 0, 1}},
                                      device),
    };

    Image albedo = sphere.ProjectImagesToAlbedo(
            {Image(view[0]), Image(view[1]), Image(view[2])},
            {intrinsic_matrix, intrinsic_matrix, intrinsic_matrix},
            {extrinsic_matrix[0], extrinsic_matrix[1], extrinsic_matrix[2]},
            256, true);

    // visualization::Draw(
    //         {std::shared_ptr<TriangleMesh>(&sphere, [](TriangleMesh*) {})},
    //         "ProjectImagesToAlbedo", 1024, 768);

    // Round to uint8_t for tests due to numerical errors between platforms
    EXPECT_THAT(albedo.AsTensor()
                        .To(core::Float32)
                        .Mean({0, 1})
                        .Round()
                        .To(core::UInt8)
                        .ToFlatVector<uint8_t>(),
                ElementsAre(88, 67, 64));
}  // namespace tests

TEST_P(TriangleMeshPermuteDevices, ComputeTriangleAreas) {
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    t::geometry::TriangleMesh mesh_empty;
    EXPECT_NO_THROW(mesh_empty.ComputeTriangleAreas());

    t::geometry::TriangleMesh t_mesh =
            t::geometry::TriangleMesh::CreateSphere(1.0, 3).To(device);
    core::Tensor ref_areas = core::Tensor::Init<float>(
            {0.39031237489989984, 0.39031237489989995, 0.39031237489989973,
             0.39031237489989995, 0.39031237489989984, 0.3903123748999,
             0.3903123748998997,  0.3903123748998999,  0.39031237489989973,
             0.39031237489989995, 0.3903123748999,     0.3903123748999002,
             0.4330127018922192,  0.43301270189221924, 0.43301270189221924,
             0.43301270189221924, 0.43301270189221924, 0.4330127018922193,
             0.4330127018922191,  0.43301270189221913, 0.4330127018922192,
             0.43301270189221924, 0.4330127018922195,  0.43301270189221963},
            device);
    t_mesh.ComputeTriangleAreas();
    EXPECT_TRUE(t_mesh.GetTriangleAttr("areas").AllClose(ref_areas));
}

TEST_P(TriangleMeshPermuteDevices, RemoveNonManifoldEdges) {
    using ::testing::UnorderedElementsAreArray;
    core::Device device = GetParam();
    if (device.IsSYCL()) GTEST_SKIP() << "Not Implemented!";

    t::geometry::TriangleMesh mesh_empty(device);
    EXPECT_TRUE(mesh_empty.RemoveNonManifoldEdges().IsEmpty());

    core::Tensor verts = core::Tensor::Init<float>(
            {
                    {0.0, 0.0, 0.0},
                    {1.0, 0.0, 0.0},
                    {0.0, 0.0, 1.0},
                    {1.0, 0.0, 1.0},
                    {0.0, 1.0, 0.0},
                    {1.0, 1.0, 0.0},
                    {0.0, 1.0, 1.0},
                    {1.0, 1.0, 1.0},
                    {0.0, -0.2, 0.0},
            },
            device);

    mesh_empty.SetVertexPositions(verts);
    EXPECT_TRUE(mesh_empty.GetVertexPositions().AllClose(verts));

    core::Tensor tris = core::Tensor::Init<int64_t>(
            {{4, 7, 5}, {8, 0, 1}, {8, 0, 1}, {8, 0, 1}, {4, 6, 7}, {0, 2, 4},
             {2, 6, 4}, {0, 1, 2}, {1, 3, 2}, {1, 5, 7}, {8, 0, 2}, {8, 0, 2},
             {8, 0, 1}, {1, 7, 3}, {2, 3, 7}, {2, 7, 6}, {8, 0, 2}, {6, 6, 7},
             {0, 4, 1}, {8, 0, 4}, {1, 4, 5}},
            device);

    core::Tensor tri_labels = tris * 10;

    t::geometry::TriangleMesh mesh(verts, tris);
    mesh.SetTriangleAttr("labels", tri_labels);

    geometry::TriangleMesh legacy_mesh = mesh.ToLegacy();
    core::Tensor expected_edges =
            core::eigen_converter::EigenVector2iVectorToTensor(
                    legacy_mesh.GetNonManifoldEdges(), core::Int64, device);
    EXPECT_TRUE(mesh.GetNonManifoldEdges().AllClose(expected_edges));

    expected_edges = core::eigen_converter::EigenVector2iVectorToTensor(
            legacy_mesh.GetNonManifoldEdges(true), core::Int64, device);
    EXPECT_TRUE(mesh.GetNonManifoldEdges(true).AllClose(expected_edges));
    EXPECT_THAT(
            core::eigen_converter::TensorToEigenVector2iVector(
                    mesh.GetNonManifoldEdges(false)),
            UnorderedElementsAreArray(std::vector<Eigen::Vector2i>{{0, 8},
                                                                   {1, 8},
                                                                   {0, 1},
                                                                   {6, 7},
                                                                   {0, 2},
                                                                   {0, 4},
                                                                   {6, 6},
                                                                   {4, 8},
                                                                   {2, 8}}));

    mesh.RemoveNonManifoldEdges();

    EXPECT_TRUE(mesh.GetNonManifoldEdges(true).AllClose(
            core::Tensor({0, 2}, core::Int64, device)));

    EXPECT_TRUE(mesh.GetNonManifoldEdges(false).AllClose(
            core::Tensor({0, 2}, core::Int64, device)));

    t::geometry::TriangleMesh box =
            t::geometry::TriangleMesh::CreateBox().To(device);
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(verts));
    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(box.GetTriangleIndices()));
    core::Tensor expected_labels = tri_labels.IndexGet(
            {core::Tensor::Init<bool>({1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                                       0, 0, 1, 1, 1, 0, 0, 1, 0, 1},
                                      device)});
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AllClose(expected_labels));
}

TEST_P(TriangleMeshPermuteDevices, SamplePointsUniformly) {
    auto mesh_empty = t::geometry::TriangleMesh();
    EXPECT_THROW(mesh_empty.SamplePointsUniformly(100), std::runtime_error);

    core::Tensor vertices =
            core::Tensor::Init<float>({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
    core::Tensor triangles = core::Tensor::Init<int32_t>({{0, 1, 2}});

    auto mesh_simple = t::geometry::TriangleMesh(vertices, triangles);

    int64_t n_points = 2;
    auto pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple.GetPointPositions().GetLength() == n_points);
    EXPECT_FALSE(pcd_simple.HasPointColors());
    EXPECT_FALSE(pcd_simple.HasPointNormals());

    core::Tensor colors =
            core::Tensor::Init<float>({{1, 0, 0}, {1, 0, 0}, {1, 0, 0}});
    core::Tensor normals =
            core::Tensor::Init<float>({{0, 1, 0}, {0, 1, 0}, {0, 1, 0}});
    mesh_simple.SetVertexColors(colors);
    mesh_simple.SetVertexNormals(normals);
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple.GetPointPositions().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointColors().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointNormals().GetLength() == n_points);

    core::Tensor ref_point_colors = core::Tensor::Init<float>({1, 0, 0});
    core::Tensor ref_point_normals = core::Tensor::Init<float>({0, 1, 0});

    for (int64_t pidx = 0; pidx < n_points; ++pidx) {
        EXPECT_TRUE(
                pcd_simple.GetPointColors()[pidx].AllClose(ref_point_colors));
        EXPECT_TRUE(
                pcd_simple.GetPointNormals()[pidx].AllClose(ref_point_normals));
    }

    // use triangle normal instead of the vertex normals
    EXPECT_FALSE(mesh_simple.HasTriangleNormals());
    // Use Float64 positions and normals and uint8_t colors
    mesh_simple.SetVertexNormals(normals.To(core::Float64));
    mesh_simple.SetVertexPositions(vertices.To(core::Float64));
    // new triangle normals
    ref_point_normals = core::Tensor::Init<double>({0, 0, 1});
    colors = core::Tensor::Init<uint8_t>(
            {{255, 0, 0}, {255, 0, 0}, {255, 0, 0}});
    mesh_simple.SetVertexColors(colors);
    pcd_simple = mesh_simple.SamplePointsUniformly(
            n_points, /*use_triangle_normal=*/true);
    // the mesh now has triangle normals as a side effect.
    EXPECT_TRUE(mesh_simple.HasTriangleNormals());
    EXPECT_TRUE(pcd_simple.GetPointPositions().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointColors().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointNormals().GetLength() == n_points);

    for (int64_t pidx = 0; pidx < n_points; ++pidx) {
        // colors are still the same (Float32)
        EXPECT_TRUE(
                pcd_simple.GetPointColors()[pidx].AllClose(ref_point_colors));
        EXPECT_TRUE(
                pcd_simple.GetPointNormals()[pidx].AllClose(ref_point_normals));
    }

    // use triangle normal, this time the mesh has no vertex normals
    mesh_simple.RemoveVertexAttr("normals");
    pcd_simple = mesh_simple.SamplePointsUniformly(
            n_points, /*use_triangle_normal=*/true);
    EXPECT_TRUE(pcd_simple.GetPointPositions().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointColors().GetLength() == n_points);
    EXPECT_TRUE(pcd_simple.GetPointNormals().GetLength() == n_points);

    for (int64_t pidx = 0; pidx < n_points; ++pidx) {
        EXPECT_TRUE(
                pcd_simple.GetPointColors()[pidx].AllClose(ref_point_colors));
        EXPECT_TRUE(
                pcd_simple.GetPointNormals()[pidx].AllClose(ref_point_normals));
    }
}

}  // namespace tests
}  // namespace open3d
