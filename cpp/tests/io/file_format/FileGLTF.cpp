// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(FileGLTF, WriteReadTriangleMeshFromGLTF) {
    geometry::TriangleMesh tm_gt;
    tm_gt.vertices_ = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    tm_gt.triangles_ = {{0, 1, 2}};
    tm_gt.ComputeVertexNormals();

    const std::string tmp_gltf_path =
            utility::filesystem::GetTempDirectoryPath() + "/tmp.gltf";
    io::WriteTriangleMesh(tmp_gltf_path, tm_gt);

    geometry::TriangleMesh tm_test;
    io::ReadTriangleMeshOptions opt;
    opt.print_progress = false;
    io::ReadTriangleMesh(tmp_gltf_path, tm_test, opt);

    ExpectEQ(tm_gt.vertices_, tm_test.vertices_);
    ExpectEQ(tm_gt.triangles_, tm_test.triangles_);
    ExpectEQ(tm_gt.vertex_normals_, tm_test.vertex_normals_);
}

// NOTE: Temporarily disabled because of a mismatch between GLB export
// (TinyGLTF) and GLB import (ASSIMP)
// TEST(FileGLTF, WriteReadTriangleMeshFromGLB) {
//     geometry::TriangleMesh tm_gt;
//     tm_gt.vertices_ = {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}};
//     tm_gt.triangles_ = {{0, 1, 2}};
//     tm_gt.ComputeVertexNormals();
//     const std::string tmp_glb_path =
//          utility::filesystem::GetTempDirectoryPath() + "/tmp.glb";
//     io::WriteTriangleMesh(tmp_glb_path, tm_gt);
//
//     geometry::TriangleMesh tm_test;
//     io::ReadTriangleMesh(tmp_glb_path, tm_test, false);
//
//     ExpectEQ(tm_gt.vertices_, tm_test.vertices_);
//     ExpectEQ(tm_gt.triangles_, tm_test.triangles_);
//     ExpectEQ(tm_gt.vertex_normals_, tm_test.vertex_normals_);
// }

}  // namespace tests
}  // namespace open3d
