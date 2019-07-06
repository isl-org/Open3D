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
                           const bool,
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
                       bool compressed /* = false*/,
                       bool write_vertex_normals /* = true*/,
                       bool write_vertex_colors /* = true*/) {
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
    bool success = map_itr->second(filename, mesh, write_ascii, compressed,
                                   write_vertex_normals, write_vertex_colors);
    utility::PrintDebug(
            "Write geometry::TriangleMesh: %d triangles and %d vertices.\n",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    return success;
}

int PointInclusionInPolygonTest(unsigned int nvert,
                                double *vertx,
                                double *verty,
                                double testx,
                                double testy) {
    int i, j, c = 0;
    for (i = 0, j = nvert - 1; i < nvert; j = i++) {
        if (((verty[i] > testy) != (verty[j] > testy)) &&
            (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) /
                                     (verty[j] - verty[i]) +
                             vertx[i]))
            c = !c;
    }
    return c;
}

bool AddTrianglesByEarClipping(geometry::TriangleMesh &mesh,
                               std::vector<unsigned int> &indices) {
    unsigned int n = indices.size();
    Eigen::Vector3d faceNormal = Eigen::Vector3d::Zero();
    if (n > 3) {
        for (int i = 0; i < n; i++) {
            const Eigen::Vector3d &v1 = mesh.vertices_[indices[(i + 1) % n]] -
                                        mesh.vertices_[indices[i % n]];
            const Eigen::Vector3d &v2 = mesh.vertices_[indices[(i + 2) % n]] -
                                        mesh.vertices_[indices[(i + 1) % n]];
            faceNormal += v1.cross(v2);
        }
        double l = std::sqrt(faceNormal.dot(faceNormal));
        faceNormal *= (1.0 / l);
    }

    bool foundEar = true;
    while (n > 3) {
        if (!foundEar) {
            // If no ear is found after all indices are looped through, the
            // polygon is not triangulable
            return false;
        }

        foundEar = false;
        for (int i = 1; i < n - 2; i++) {
            const Eigen::Vector3d &v1 =
                    mesh.vertices_[indices[i]] - mesh.vertices_[indices[i - 1]];
            const Eigen::Vector3d &v2 =
                    mesh.vertices_[indices[i + 1]] - mesh.vertices_[indices[i]];
            Eigen::Vector3d v_cross = v1.cross(v2);
            bool isConvex = (faceNormal.dot(v1.cross(v2)) > 0.0);
            bool isEar = true;
            if (isConvex) {
                // If convex, check if vertex is an ear
                // (no vertices within triangle v[i-1], v[i], v[i+1])
                double triangle_vertices_x[3];
                double triangle_vertices_y[3];
                for (int j = 0; j < 3; j++) {
                    const Eigen::Vector3d &v1 =
                            mesh.vertices_[indices[i + j - 1]];
                    const Eigen::Vector3d &v2 =
                            mesh.vertices_[indices[i + j - 1]];
                    triangle_vertices_x[j] = v1(0);
                    triangle_vertices_y[j] = v2(1);
                }

                for (int j = 0; j < n; j++) {
                    if (j == i - 1 || j == i || j == i + 1) {
                        continue;
                    }
                    const Eigen::Vector3d &v1 = mesh.vertices_[indices[j]];
                    const Eigen::Vector3d &v2 = mesh.vertices_[indices[j]];
                    double test_vertex_x = v1(0);
                    double test_vertex_y = v2(1);
                    int vertexIsInside = PointInclusionInPolygonTest(
                            3, triangle_vertices_x, triangle_vertices_y,
                            test_vertex_x, test_vertex_y);
                    if (vertexIsInside) {
                        isEar = false;
                        break;
                    }
                }

                if (isEar) {
                    foundEar = true;

                    mesh.triangles_.push_back(Eigen::Vector3i(
                            indices[i - 1], indices[i], indices[i + 1]));

                    // Remove ear vertex from indices
                    std::vector<unsigned int> buffer;
                    for (int j = 0; j < n; j++) {
                        if (j != i) {
                            buffer.push_back(indices[j]);
                        }
                    }
                    indices = buffer;
                    n = indices.size();

                    break;
                }
            }
        }
    }
    mesh.triangles_.push_back(
            Eigen::Vector3i(indices[0], indices[1], indices[2]));

    return true;
}

}  // namespace io
}  // namespace open3d
