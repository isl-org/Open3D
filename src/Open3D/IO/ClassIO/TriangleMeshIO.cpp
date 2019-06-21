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

#include "Open3D/IO/ClassIO/TriangleMeshIO.h"

#include <unordered_map>

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::TriangleMesh &)>>
        file_extension_to_trianglemesh_read_function{
                {"ply", ReadTriangleMeshFromPLY},
                {"stl", ReadTriangleMeshFromSTL},
                {"obj", ReadTriangleMeshFromOBJ},
                {"off", ReadTriangleMeshFromOFF},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::TriangleMesh &,
                           const bool,
                           const bool)>>
        file_extension_to_trianglemesh_write_function{
                {"ply", WriteTriangleMeshToPLY},
                {"stl", WriteTriangleMeshToSTL},
                {"obj", WriteTriangleMeshToOBJ},
                {"off", WriteTriangleMeshToOFF},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    ReadTriangleMesh(filename, *mesh);
    return mesh;
}

bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::PrintWarning(
                "Read geometry::TriangleMesh failed: unknown file "
                "extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_read_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_read_function.end()) {
        utility::PrintWarning(
                "Read geometry::TriangleMesh failed: unknown file "
                "extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, mesh);
    utility::PrintDebug(
            "Read geometry::TriangleMesh: %d triangles and %d vertices.\n",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    if (mesh.HasVertices() && !mesh.HasTriangles()) {
        utility::PrintWarning(
                "geometry::TriangleMesh appears to be a geometry::PointCloud "
                "(only contains "
                "vertices, but no triangles).\n");
    }
    return success;
}

bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::PrintWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_write_function.end()) {
        utility::PrintWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, mesh, write_ascii, compressed);
    utility::PrintDebug(
            "Write geometry::TriangleMesh: %d triangles and %d vertices.\n",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
