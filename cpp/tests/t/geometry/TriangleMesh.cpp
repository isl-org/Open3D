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

#include "open3d/t/geometry/TriangleMesh.h"

#include "core/CoreTest.h"
#include "open3d/core/TensorList.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

class TriangleMeshPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(TriangleMesh,
                         TriangleMeshPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

class TriangleMeshPermuteDevicePairs : public PermuteDevicePairs {};
INSTANTIATE_TEST_SUITE_P(
        TriangleMesh,
        TriangleMeshPermuteDevicePairs,
        testing::ValuesIn(TriangleMeshPermuteDevicePairs::TestCases()));

TEST_P(TriangleMeshPermuteDevices, DefaultConstructor) {
    t::geometry::TriangleMesh mesh;

    // Inherited from Geometry3D.
    EXPECT_EQ(mesh.GetGeometryType(),
              t::geometry::Geometry::GeometryType::TriangleMesh);
    EXPECT_EQ(mesh.Dimension(), 3);

    // Public members.
    EXPECT_TRUE(mesh.IsEmpty());
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Default device.
    EXPECT_EQ(mesh.GetDevice(), core::Device("CPU:0"));
}

TEST_P(TriangleMeshPermuteDevices, ConstructFromVertices) {
    core::Device device = GetParam();

    // Prepare data.
    core::Tensor vertices =
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device);
    core::Tensor single_vertex =
            core::Tensor::Ones({3}, core::Dtype::Float32, device);

    core::Tensor triangles =
            core::Tensor::Ones({10, 3}, core::Dtype::Int64, device);
    core::Tensor single_triangle =
            core::Tensor::Ones({3}, core::Dtype::Int64, device);

    t::geometry::TriangleMesh mesh(vertices, triangles);

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_EQ(mesh.GetVertices().GetLength(), 10);
    EXPECT_EQ(mesh.GetTriangles().GetLength(), 10);
}

TEST_P(TriangleMeshPermuteDevices, Getters) {
    core::Device device = GetParam();

    core::Tensor vertices =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor vertex_colors =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2;
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3;

    core::Tensor triangles =
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device);
    core::Tensor triangle_normals =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2;
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3;

    t::geometry::TriangleMesh mesh(vertices, triangles);
    mesh.SetVertexColors(vertex_colors);
    mesh.SetVertexAttr("labels", vertex_labels);
    mesh.SetTriangleNormals(triangle_normals);
    mesh.SetTriangleAttr("labels", triangle_labels);

    EXPECT_TRUE(mesh.GetVertices().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device)));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetVertexAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));
    EXPECT_ANY_THROW(mesh.GetVertexNormals());

    EXPECT_TRUE(mesh.GetTriangles().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device)));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 2));
    EXPECT_TRUE(mesh.GetTriangleAttr("labels").AllClose(
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3));

    // Const getters. (void)tl gets rid of the unused variables warning.
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertices(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexColors(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetVertexAttr("labels");
                    (void)tl);
    EXPECT_ANY_THROW(const core::Tensor& tl = mesh.GetVertexNormals();
                     (void)tl);

    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangles(); (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleNormals();
                    (void)tl);
    EXPECT_NO_THROW(const core::Tensor& tl = mesh.GetTriangleAttr("labels");
                    (void)tl);
}

TEST_P(TriangleMeshPermuteDevices, Setters) {
    core::Device device = GetParam();

    // Setters are already tested in Getters' unit tests. Here we test that
    // mismatched device should throw an exception. This test is only effective
    // is device is a CUDA device.
    t::geometry::TriangleMesh mesh(device);
    core::Device cpu_device = core::Device("CPU:0");
    if (cpu_device != device) {
        core::Tensor cpu_vertices =
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device);
        core::Tensor cpu_colors =
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                2;
        core::Tensor cpu_labels =
                core::Tensor::Ones({2, 3}, core::Dtype::Float32, cpu_device) *
                3;

        EXPECT_ANY_THROW(mesh.SetVertices(cpu_vertices));
        EXPECT_ANY_THROW(mesh.SetVertexColors(cpu_colors));
        EXPECT_ANY_THROW(mesh.SetVertexAttr("labels", cpu_labels));
    }
}

TEST_P(TriangleMeshPermuteDevices, RemoveAttr) {
    core::Device device = GetParam();

    core::Tensor vertices =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device);
    core::Tensor vertex_labels =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3;

    core::Tensor triangles =
            core::Tensor::Ones({2, 3}, core::Dtype::Int64, device);
    core::Tensor triangle_labels =
            core::Tensor::Ones({2, 3}, core::Dtype::Float32, device) * 3;

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
    EXPECT_ANY_THROW(mesh.RemoveVertexAttr("vertices"));
    EXPECT_ANY_THROW(mesh.RemoveTriangleAttr("triangles"));
}

TEST_P(TriangleMeshPermuteDevices, Has) {
    core::Device device = GetParam();

    t::geometry::TriangleMesh mesh(device);
    EXPECT_FALSE(mesh.HasVertices());
    EXPECT_FALSE(mesh.HasVertexColors());
    EXPECT_FALSE(mesh.HasVertexNormals());
    EXPECT_FALSE(mesh.HasVertexAttr("labels"));
    EXPECT_FALSE(mesh.HasTriangles());
    EXPECT_FALSE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleAttr("labels"));

    mesh.SetVertices(core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    EXPECT_TRUE(mesh.HasVertices());
    mesh.SetTriangles(core::Tensor::Ones({10, 3}, core::Dtype::Int64, device));
    EXPECT_TRUE(mesh.HasTriangles());

    // Different size.
    mesh.SetVertexColors(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device));
    EXPECT_FALSE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(
            core::Tensor::Ones({5, 3}, core::Dtype::Float32, device));
    EXPECT_FALSE(mesh.HasTriangleNormals());

    // Same size.
    mesh.SetVertexColors(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    EXPECT_TRUE(mesh.HasVertexColors());
    mesh.SetTriangleNormals(
            core::Tensor::Ones({10, 3}, core::Dtype::Float32, device));
    EXPECT_TRUE(mesh.HasTriangleNormals());
}

TEST_P(TriangleMeshPermuteDevices, FromLegacyTriangleMesh) {
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

    core::Dtype float_dtype = core::Dtype::Float32;
    core::Dtype int_dtype = core::Dtype::Int64;
    t::geometry::TriangleMesh mesh =
            t::geometry::TriangleMesh::FromLegacyTriangleMesh(
                    legacy_mesh, float_dtype, int_dtype, device);

    EXPECT_TRUE(mesh.HasVertices());
    EXPECT_TRUE(mesh.HasVertexColors());
    EXPECT_TRUE(mesh.HasVertexNormals());
    EXPECT_TRUE(mesh.HasTriangles());
    EXPECT_TRUE(mesh.HasTriangleNormals());
    EXPECT_FALSE(mesh.HasTriangleColors());

    EXPECT_NO_THROW(mesh.GetVertices().AssertDtype(float_dtype));
    EXPECT_NO_THROW(mesh.GetVertices().AssertDtype(float_dtype));
    EXPECT_NO_THROW(mesh.GetVertexColors().AssertDtype(float_dtype));
    EXPECT_NO_THROW(mesh.GetVertexNormals().AssertDtype(float_dtype));
    EXPECT_NO_THROW(mesh.GetTriangles().AssertDtype(int_dtype));
    EXPECT_NO_THROW(mesh.GetTriangleNormals().AssertDtype(float_dtype));

    EXPECT_TRUE(mesh.GetVertices().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 0));
    EXPECT_TRUE(mesh.GetVertexColors().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 1));
    EXPECT_TRUE(mesh.GetVertexNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 2));
    EXPECT_TRUE(mesh.GetTriangles().AllClose(
            core::Tensor::Ones({2, 3}, int_dtype, device) * 3));
    EXPECT_TRUE(mesh.GetTriangleNormals().AllClose(
            core::Tensor::Ones({2, 3}, float_dtype, device) * 4));
}

TEST_P(TriangleMeshPermuteDevices, ToLegacyTriangleMesh) {
    core::Device device = GetParam();

    core::Dtype float_dtype = core::Dtype::Float32;
    core::Dtype int_dtype = core::Dtype::Int64;

    t::geometry::TriangleMesh mesh(device);
    mesh.SetVertices(core::Tensor::Ones({2, 3}, float_dtype, device) * 0);
    mesh.SetVertexColors(core::Tensor::Ones({2, 3}, float_dtype, device) * 1);
    mesh.SetVertexNormals(core::Tensor::Ones({2, 3}, float_dtype, device) * 2);
    mesh.SetTriangles(core::Tensor::Ones({2, 3}, int_dtype, device) * 3);
    mesh.SetTriangleNormals(core::Tensor::Ones({2, 3}, float_dtype, device) *
                            4);

    geometry::TriangleMesh legacy_mesh = mesh.ToLegacyTriangleMesh();
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
}

}  // namespace tests
}  // namespace open3d
