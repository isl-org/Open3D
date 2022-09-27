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

#include "open3d/data/Dataset.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshIO, CreateMeshFromFile) {
    data::KnotMesh knot_data;
    auto mesh = t::io::CreateMeshFromFile(knot_data.GetPath());
    EXPECT_EQ(mesh->GetTriangleIndices().GetLength(), 2880);
    EXPECT_EQ(mesh->GetVertexPositions().GetLength(), 1440);
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshPLY) {
    data::KnotMesh knot_data;
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(knot_data.GetPath(), mesh));
    std::string file_name =
            utility::filesystem::GetTempDirectoryPath() + "/test_mesh.ply";
    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name, mesh));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name, mesh_read));
    EXPECT_TRUE(
            mesh.GetTriangleIndices().AllClose(mesh_read.GetTriangleIndices()));
    EXPECT_TRUE(
            mesh.GetVertexPositions().AllClose(mesh_read.GetVertexPositions()));
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshOBJ) {
    auto cube_mesh = t::geometry::TriangleMesh::CreateBox();

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/cube.obj";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, cube_mesh));
    t::geometry::TriangleMesh mesh, mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh));

    core::Tensor vertices = core::Tensor::Init<float>({{0.0, 1.0, 0.0},
                                                       {1.0, 1.0, 1.0},
                                                       {1.0, 1.0, 0.0},
                                                       {0.0, 1.0, 1.0},
                                                       {0.0, 0.0, 0.0},
                                                       {0.0, 0.0, 1.0},
                                                       {1.0, 0.0, 0.0},
                                                       {1.0, 0.0, 1.0}});
    EXPECT_TRUE(mesh.GetVertexPositions().AllClose(vertices));

    core::Tensor triangles = core::Tensor::Init<int64_t>({{0, 1, 2},
                                                          {0, 3, 1},
                                                          {4, 5, 0},
                                                          {5, 3, 0},
                                                          {4, 6, 5},
                                                          {6, 7, 5},
                                                          {6, 2, 1},
                                                          {6, 1, 7},
                                                          {5, 7, 1},
                                                          {5, 1, 3},
                                                          {4, 0, 6},
                                                          {6, 0, 2}});
    EXPECT_TRUE(mesh.GetTriangleIndices().AllClose(triangles));
}

// TODO: Add tests for triangle_uvs, materials, triangle_material_ids and
// textures once these are supported.
TEST(TriangleMeshIO, TriangleMeshLegecyCompatibility) {
    t::geometry::TriangleMesh mesh_tensor, mesh_tensor_read;
    geometry::TriangleMesh mesh_legacy, mesh_legacy_read;
    data::BunnyMesh bunny_mesh;
    EXPECT_TRUE(t::io::ReadTriangleMesh(bunny_mesh.GetPath(), mesh_tensor));
    EXPECT_TRUE(io::ReadTriangleMesh(bunny_mesh.GetPath(), mesh_legacy));

    EXPECT_EQ(mesh_tensor.GetTriangleIndices().GetLength(),
              static_cast<int64_t>(mesh_legacy.triangles_.size()));
    EXPECT_EQ(mesh_tensor.GetVertexPositions().GetLength(),
              static_cast<int64_t>(mesh_legacy.vertices_.size()));

    const std::string tmp_path = utility::filesystem::GetTempDirectoryPath();
    std::string file_name_tensor = tmp_path + "/test_mesh_tensor.obj";
    std::string file_name_legacy = tmp_path + "/test_mesh_legacy.obj";

    EXPECT_TRUE(t::io::WriteTriangleMesh(file_name_tensor, mesh_tensor));
    EXPECT_TRUE(io::WriteTriangleMesh(file_name_legacy, mesh_legacy));
    EXPECT_TRUE(t::io::ReadTriangleMesh(file_name_tensor, mesh_tensor_read));
    EXPECT_TRUE(io::ReadTriangleMesh(file_name_legacy, mesh_legacy_read));

    EXPECT_EQ(mesh_tensor_read.GetTriangleIndices().GetLength(),
              static_cast<int64_t>(mesh_legacy_read.triangles_.size()));
    EXPECT_EQ(mesh_tensor_read.GetVertexPositions().GetLength(),
              static_cast<int64_t>(mesh_legacy_read.vertices_.size()));
}

}  // namespace tests
}  // namespace open3d
