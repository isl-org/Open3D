// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/TriangleMeshIO.h"

#include "open3d/core/Tensor.h"
#include "open3d/data/Dataset.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

// Unit box with 2×2 albedo, roughness, and metallic maps and per-triangle UVs.
// Used by all material round-trip tests.
t::geometry::TriangleMesh MakeTexturedBoxPBR() {
    auto mesh = t::geometry::TriangleMesh::CreateBox();
    const int64_t n = mesh.GetTriangleIndices().GetLength();
    mesh.SetTriangleAttr("texture_uvs",
                         core::Tensor::Zeros({n, 3, 2}, core::Float32));
    mesh.GetMaterial().SetDefaultProperties();
    mesh.GetMaterial().SetAlbedoMap(t::geometry::Image(
            core::Tensor::Ones({2, 2, 3}, core::UInt8) * 200));
    mesh.GetMaterial().SetRoughnessMap(t::geometry::Image(
            core::Tensor::Ones({2, 2, 1}, core::UInt8) * 180));
    mesh.GetMaterial().SetMetallicMap(t::geometry::Image(
            core::Tensor::Ones({2, 2, 1}, core::UInt8) * 64));
    return mesh;
}

}  // namespace

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
    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));

    // Vertex and triangle counts must be preserved.
    EXPECT_EQ(mesh_read.GetVertexPositions().GetLength(),
              cube_mesh.GetVertexPositions().GetLength());
    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              cube_mesh.GetTriangleIndices().GetLength());
}

// OBJ round-trip with PBR material.  The OBJ/MTL format has no PBR material
// properties, so roughness/metallic maps are skipped on export (a warning is
// logged).  Only albedo (map_Kd) is written to the MTL and read back.
TEST(TriangleMeshIO, ReadWriteTriangleMeshOBJMaterial) {
    t::geometry::TriangleMesh mesh = MakeTexturedBoxPBR();

    const std::string tmp =
            utility::filesystem::GetTempDirectoryPath() + "/obj_material";
    const std::string filename = tmp + ".obj";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, mesh));

    // Albedo sidecar PNG must be written and referenced by the MTL.
    EXPECT_TRUE(utility::filesystem::FileExists(tmp + "_albedo.png"));

    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));

    EXPECT_EQ(mesh_read.GetVertexPositions().GetLength(),
              mesh.GetVertexPositions().GetLength());
    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              mesh.GetTriangleIndices().GetLength());
    // Per-triangle UVs round-trip via ASSIMP's per-vertex UV export.
    EXPECT_TRUE(mesh_read.HasTriangleAttr("texture_uvs"));
    EXPECT_TRUE(mesh_read.GetMaterial().IsValid());
    EXPECT_TRUE(mesh_read.GetMaterial().HasAlbedoMap());
    EXPECT_EQ(mesh_read.GetMaterial().GetAlbedoMap().GetRows(), 2);
    EXPECT_EQ(mesh_read.GetMaterial().GetAlbedoMap().GetCols(), 2);
}

// GLB round-trip with PBR material: textures are embedded (no sidecar files).
// The ao_rough_metal map (G=roughness, B=metallic) is written to ASSIMP 6's
// aiTextureType_DIFFUSE_ROUGHNESS slot, which the glTF2 exporter maps to
// metallicRoughnessTexture.  On read-back ASSIMP 6 stores it in the dedicated
// aiTextureType_GLTF_METALLIC_ROUGHNESS slot, which our reader loads as
// ao_rough_metal.  A combined map is set directly here to isolate this path
// from CombineRoughnessMetallic.
TEST(TriangleMeshIO, ReadWriteTriangleMeshGLBMaterial) {
    auto mesh = t::geometry::TriangleMesh::CreateBox();
    const int64_t n = mesh.GetTriangleIndices().GetLength();
    mesh.SetTriangleAttr("texture_uvs",
                         core::Tensor::Zeros({n, 3, 2}, core::Float32));
    mesh.GetMaterial().SetDefaultProperties();
    mesh.GetMaterial().SetAlbedoMap(t::geometry::Image(
            core::Tensor::Ones({2, 2, 3}, core::UInt8) * 200));
    mesh.GetMaterial().SetAORoughnessMetalMap(t::geometry::Image(
            core::Tensor::Ones({2, 2, 4}, core::UInt8) * 128));

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/glb_material.glb";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, mesh));

    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));

    EXPECT_EQ(mesh_read.GetVertexPositions().GetLength(),
              mesh.GetVertexPositions().GetLength());
    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              mesh.GetTriangleIndices().GetLength());
    EXPECT_TRUE(mesh_read.GetMaterial().HasAlbedoMap());
    EXPECT_EQ(mesh_read.GetMaterial().GetAlbedoMap().GetRows(), 2);
    EXPECT_EQ(mesh_read.GetMaterial().GetAlbedoMap().GetCols(), 2);
    EXPECT_TRUE(mesh_read.GetMaterial().HasAORoughnessMetalMap());
    EXPECT_EQ(mesh_read.GetMaterial().GetAORoughnessMetalMap().GetRows(), 2);
    EXPECT_EQ(mesh_read.GetMaterial().GetAORoughnessMetalMap().GetCols(), 2);
}

// FBX round-trip with PBR material: geometry and albedo sidecar are verified;
// full material fidelity is best-effort for the ASSIMP FBX exporter.
TEST(TriangleMeshIO, ReadWriteTriangleMeshFBXMaterial) {
    t::geometry::TriangleMesh mesh = MakeTexturedBoxPBR();

    const std::string tmp =
            utility::filesystem::GetTempDirectoryPath() + "/fbx_material";
    const std::string filename = tmp + ".fbx";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, mesh));

    // Sidecar textures must be written even for FBX.
    EXPECT_TRUE(utility::filesystem::FileExists(tmp + "_albedo.png"));

    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));
    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              mesh.GetTriangleIndices().GetLength());
    EXPECT_EQ(mesh_read.GetVertexPositions().GetLength(),
              mesh.GetVertexPositions().GetLength());
}

// STL round-trip: triangle count only.  STL stores 3 vertices per face (no
// shared-vertex support), so after ASSIMP's JoinIdenticalVertices pass the
// vertex count differs from the original (e.g. 24 for a box, not 8).
TEST(TriangleMeshIO, ReadWriteTriangleMeshSTL) {
    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::CreateBox();

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/box.stl";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, mesh));

    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));

    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              mesh.GetTriangleIndices().GetLength());
}

// FBX round-trip: geometry counts only (ASSIMP FBX exporter is best-effort).
TEST(TriangleMeshIO, ReadWriteTriangleMeshFBX) {
    t::geometry::TriangleMesh mesh = t::geometry::TriangleMesh::CreateBox();

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/box.fbx";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, mesh));

    t::geometry::TriangleMesh mesh_read;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh_read));

    EXPECT_EQ(mesh_read.GetTriangleIndices().GetLength(),
              mesh.GetTriangleIndices().GetLength());
    EXPECT_EQ(mesh_read.GetVertexPositions().GetLength(),
              mesh.GetVertexPositions().GetLength());
}

TEST(TriangleMeshIO, ReadWriteTriangleMeshNPZ) {
    auto cube_mesh = t::geometry::TriangleMesh::CreateBox();

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() + "/cube.npz";
    EXPECT_TRUE(t::io::WriteTriangleMesh(filename, cube_mesh));
    t::geometry::TriangleMesh mesh;
    EXPECT_TRUE(t::io::ReadTriangleMesh(filename, mesh));
    EXPECT_TRUE(
            mesh.GetVertexPositions().AllClose(cube_mesh.GetVertexPositions()));
    EXPECT_TRUE(
            mesh.GetTriangleIndices().AllClose(cube_mesh.GetTriangleIndices()));
}

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
