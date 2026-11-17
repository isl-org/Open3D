// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <json/json.h>

#include <cstdint>
#include <memory>

#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

class FileGLTFIndexBounds : public testing::TestWithParam<std::string> {};

TEST_P(FileGLTFIndexBounds, WriteTriangleMesh) {
    struct TestCase {
        size_t vertex_count;
        std::vector<Eigen::Vector3i> triangles;
        int min_index;
        int max_index;
        int component_type;
    };
    // Two triangles isolate accessor bounds from index-buffer alignment.
    const std::vector<TestCase> cases = {
            {6, {{0, 1, 2}, {3, 4, 5}}, 0, 5, 5123},
            {4, {{0, 2, 1}, {3, 1, 2}}, 0, 3, 5123},
            {6, {{2, 3, 4}, {3, 5, 4}}, 2, 5, 5123},
            {20, {{2, 7, 11}, {2, 11, 17}}, 2, 17, 5123},
            {65539, {{1, 2, 65536}, {2, 3, 65536}}, 1, 65536, 5125},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(GetParam());
        SCOPED_TRACE(test_case.vertex_count);
        SCOPED_TRACE(test_case.min_index);
        geometry::TriangleMesh mesh;
        mesh.vertices_.resize(test_case.vertex_count, Eigen::Vector3d::Zero());
        mesh.triangles_ = test_case.triangles;
        const std::string path = utility::filesystem::GetTempDirectoryPath() +
                                 "/open3d_index_bounds" + GetParam();
        ASSERT_TRUE(io::WriteTriangleMesh(path, mesh));

        std::vector<char> bytes;
        std::string error;
        const bool read_success =
                utility::filesystem::FReadToBuffer(path, bytes, &error);
        EXPECT_TRUE(utility::filesystem::RemoveFile(path));
        ASSERT_TRUE(read_success) << error;
        size_t json_offset = 0;
        size_t json_length = bytes.size();
        if (GetParam() == ".glb") {
            // The first GLB chunk is JSON, after the 12-byte file header
            // and an 8-byte chunk header with a little-endian length.
            ASSERT_GE(bytes.size(), 20);
            ASSERT_EQ(std::string(bytes.data(), 4), "glTF");
            ASSERT_EQ(std::string(bytes.data() + 16, 4), "JSON");
            json_offset = 20;
            json_length = 0;
            for (size_t i = 0; i < 4; ++i) {
                json_length |= uint32_t(uint8_t(bytes[12 + i])) << (8 * i);
            }
            ASSERT_LE(json_length, bytes.size() - json_offset);
        }

        Json::Value model;
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        ASSERT_TRUE(reader->parse(bytes.data() + json_offset,
                                  bytes.data() + json_offset + json_length,
                                  &model, &error))
                << error;
        const auto accessor_index =
                model["meshes"][0]["primitives"][0]["indices"].asUInt();
        const auto& accessor = model["accessors"][accessor_index];
        EXPECT_EQ(accessor["componentType"].asInt(), test_case.component_type);
        EXPECT_EQ(accessor["count"].asUInt(), 6);
        ASSERT_EQ(accessor["min"].size(), 1);
        ASSERT_EQ(accessor["max"].size(), 1);
        EXPECT_EQ(accessor["min"][0].asInt(), test_case.min_index);
        EXPECT_EQ(accessor["max"][0].asInt(), test_case.max_index);
    }
}

INSTANTIATE_TEST_SUITE_P(FileGLTF,
                         FileGLTFIndexBounds,
                         testing::Values(".gltf", ".glb"));

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
