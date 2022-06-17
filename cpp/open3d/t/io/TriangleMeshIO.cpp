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

#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           geometry::TriangleMesh &,
                           const open3d::io::ReadTriangleMeshOptions &)>>
        file_extension_to_trianglemesh_read_function{};

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::TriangleMesh &,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool)>>
        file_extension_to_trianglemesh_write_function{};

std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename, bool print_progress) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    open3d::io::ReadTriangleMeshOptions opt;
    opt.print_progress = print_progress;
    ReadTriangleMesh(filename, *mesh, opt);
    return mesh;
}

// TODO:
// 1. Currently, the tensor triangle mesh implementation has no provision for
// triangle_uvs,  materials, triangle_material_ids and textures which are
// supported by the legacy. These can be added as custom attributes (level 2)
// approach. Please check legacy file formats(e.g. FileOBJ.cpp) for more
// information.
// 2. Add these properties to the legacy to tensor mesh and tensor to legacy
// mesh conversion.
// 3. Update the documentation with information on how to access these
// additional attributes from tensor based triangle mesh.
// 4. Implement read/write tensor triangle mesh with various file formats.
// 5. Compare with legacy triangle mesh and add corresponding unit tests.

bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      open3d::io::ReadTriangleMeshOptions params) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }

    auto map_itr =
            file_extension_to_trianglemesh_read_function.find(filename_ext);
    bool success = false;
    if (map_itr == file_extension_to_trianglemesh_read_function.end()) {
        open3d::geometry::TriangleMesh legacy_mesh;
        success = open3d::io::ReadTriangleMesh(filename, legacy_mesh, params);
        if (!success) {
            return false;
        }
        mesh = geometry::TriangleMesh::FromLegacy(legacy_mesh);
    } else {
        success = map_itr->second(filename, mesh, params);
        utility::LogDebug(
                "Read geometry::TriangleMesh: {:d} triangles and {:d} "
                "vertices.",
                mesh.GetTriangleIndices().GetLength(),
                mesh.GetVertexPositions().GetLength());
        if (mesh.HasVertexPositions() && !mesh.HasTriangleIndices()) {
            utility::LogWarning(
                    "geometry::TriangleMesh appears to be a "
                    "geometry::PointCloud "
                    "(only contains vertices, but no triangles).");
        }
    }
    return success;
}

bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/,
                       bool write_vertex_normals /* = true*/,
                       bool write_vertex_colors /* = true*/,
                       bool write_triangle_uvs /* = true*/,
                       bool print_progress /* = false*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_write_function.end()) {
        return open3d::io::WriteTriangleMesh(
                filename, mesh.ToLegacy(), write_ascii, compressed,
                write_vertex_normals, write_vertex_colors, write_triangle_uvs,
                print_progress);
    }
    bool success = map_itr->second(filename, mesh, write_ascii, compressed,
                                   write_vertex_normals, write_vertex_colors,
                                   write_triangle_uvs, print_progress);
    utility::LogDebug(
            "Write geometry::TriangleMesh: {:d} triangles and {:d} vertices.",
            mesh.GetTriangleIndices().GetLength(),
            mesh.GetVertexPositions().GetLength());
    return success;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
