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

#include "open3d/t/io/TriangleMeshIO.h"

#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshIO, CreateMeshFromFile) {
    auto mesh =
            t::io::CreateMeshFromFile(utility::GetDataPathCommon("knot.ply"));
    EXPECT_EQ(mesh->GetTriangleIndices().GetLength(), 2880);
    EXPECT_EQ(mesh->GetVertexPositions().GetLength(), 1440);
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshPLY) {
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(utility::GetDataPathCommon("knot.ply"),
                                        mesh));
    std::string file_name = utility::GetDataPathCommon("test_mesh.ply");
    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name, mesh));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name, mesh_read));
    EXPECT_TRUE(
            mesh.GetTriangleIndices().AllClose(mesh_read.GetTriangleIndices()));
    EXPECT_TRUE(
            mesh.GetVertexPositions().AllClose(mesh_read.GetVertexPositions()));
    std::remove(file_name.c_str());
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshOBJ) {
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(
            utility::GetDataPathDownload("tests/cube.obj"), mesh));

    core::Tensor vertices = core::Tensor::Init<float>(
            {{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0},
             {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 1.0},
             {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0},
             {1.0, 1.0, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0},
             {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0},
             {1.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
             {1.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0},
             {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}});
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(vertices));

    core::Tensor triangles = core::Tensor::Init<int64_t>({{0, 1, 2},
                                                          {0, 3, 1},
                                                          {4, 5, 6},
                                                          {4, 7, 5},
                                                          {8, 9, 10},
                                                          {8, 11, 9},
                                                          {12, 13, 14},
                                                          {12, 14, 15},
                                                          {16, 17, 18},
                                                          {16, 18, 19},
                                                          {20, 21, 22},
                                                          {20, 22, 23}});
    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(triangles));

    std::string file_name = utility::GetDataPathCommon("test_mesh.obj");
    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name, mesh));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name, mesh_read));
    EXPECT_TRUE(
            mesh.GetVertexPositions().AllClose(mesh_read.GetVertexPositions()));
    EXPECT_TRUE(
            mesh.GetTriangleIndices().AllClose(mesh_read.GetTriangleIndices()));

    std::remove(file_name.c_str());
}

// TODO: Add tests for triangle_uvs, materials, triangle_material_ids and
// textures once these are supported.
TEST(TriangleMeshIO, TriangleMeshLegecyCompatibility) {
    t::geometry::TriangleMesh mesh_tensor, mesh_tensor_read;
    geometry::TriangleMesh mesh_legacy, mesh_legacy_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(
            utility::GetDataPathDownload("tests/cube.obj"), mesh_tensor));
    EXPECT_TRUE(io::ReadTriangleMesh(
            utility::GetDataPathDownload("tests/cube.obj"), mesh_legacy));

    EXPECT_EQ(mesh_tensor.GetTriangleIndices().GetLength(),
              static_cast<int64_t>(mesh_legacy.triangles_.size()));
    EXPECT_EQ(mesh_tensor.GetVertexPositions().GetLength(),
              static_cast<int64_t>(mesh_legacy.vertices_.size()));
    EXPECT_EQ(mesh_tensor.GetVertexNormals().GetLength(),
              static_cast<int64_t>(mesh_legacy.vertex_normals_.size()));

    std::string file_name_tensor =
            utility::GetDataPathCommon("test_mesh_tensor.obj");
    std::string file_name_legacy =
            utility::GetDataPathCommon("test_mesh_legacy.obj");

    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name_tensor, mesh_tensor));
    EXPECT_TRUE(io::WriteTriangleMesh(file_name_legacy, mesh_legacy));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name_tensor, mesh_tensor_read));
    EXPECT_TRUE(io::ReadTriangleMesh(file_name_legacy, mesh_legacy_read));

    EXPECT_EQ(mesh_tensor_read.GetTriangleIndices().GetLength(),
              static_cast<int64_t>(mesh_legacy_read.triangles_.size()));
    EXPECT_EQ(mesh_tensor_read.GetVertexPositions().GetLength(),
              static_cast<int64_t>(mesh_legacy_read.vertices_.size()));
    EXPECT_EQ(mesh_tensor_read.GetVertexNormals().GetLength(),
              static_cast<int64_t>(mesh_legacy_read.vertex_normals_.size()));
    std::remove(file_name_tensor.c_str());
    std::remove(file_name_legacy.c_str());
    std::string file_name_legacy_mtl =
            utility::GetDataPathCommon("test_mesh_legacy.mtl");
    std::remove(file_name_legacy_mtl.c_str());
}

}  // namespace tests
}  // namespace open3d
