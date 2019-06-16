// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <fstream>
#include <numeric>
#include <vector>

#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/Console.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace open3d {
namespace io {

bool ReadTriangleMeshFromOBJ(const std::string& filename,
                             geometry::TriangleMesh& mesh) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filename.c_str());

    if (!warn.empty()) {
        utility::PrintWarning("Read OBJ failed: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        utility::PrintError("Read OBJ failed: %s\n", err.c_str());
    }

    if (!ret) {
        return false;
    }

    mesh.Clear();

    // copy vertex and vertex_color data
    for (int vidx = 0; vidx < attrib.vertices.size(); vidx += 3) {
        tinyobj::real_t vx = attrib.vertices[vidx + 0];
        tinyobj::real_t vy = attrib.vertices[vidx + 1];
        tinyobj::real_t vz = attrib.vertices[vidx + 2];
        mesh.vertices_.push_back(Eigen::Vector3d(vx, vy, vz));
    }
    for (int vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
        tinyobj::real_t r = attrib.colors[vidx + 0];
        tinyobj::real_t g = attrib.colors[vidx + 1];
        tinyobj::real_t b = attrib.colors[vidx + 2];
        mesh.vertex_colors_.push_back(Eigen::Vector3d(r, g, b));
    }

    // resize normal data and create bool indicator vector
    mesh.vertex_normals_.resize(mesh.vertices_.size());
    std::vector<bool> normals_indicator(mesh.vertices_.size(), false);

    // copy face data and copy normals data
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) {
                utility::PrintWarning(
                        "Read OBJ failed: facet with number of vertices not "
                        "equal to 3\n");
                return false;
            }

            Eigen::Vector3i facet;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                int vidx = idx.vertex_index;
                facet(v) = vidx;

                if (!normals_indicator[vidx] &&
                    (3 * idx.normal_index + 2) < attrib.normals.size()) {
                    tinyobj::real_t nx =
                            attrib.normals[3 * idx.normal_index + 0];
                    tinyobj::real_t ny =
                            attrib.normals[3 * idx.normal_index + 1];
                    tinyobj::real_t nz =
                            attrib.normals[3 * idx.normal_index + 2];
                    mesh.vertex_normals_[vidx](0) = nx;
                    mesh.vertex_normals_[vidx](1) = ny;
                    mesh.vertex_normals_[vidx](2) = nz;
                    normals_indicator[vidx] = true;
                }
            }
            mesh.triangles_.push_back(facet);
            index_offset += fv;
        }
    }

    // if not all normals have been set, then remove the vertex normals
    bool all_normals_set =
            std::accumulate(normals_indicator.begin(), normals_indicator.end(),
                            true, [](bool a, bool b) { return a && b; });
    if (!all_normals_set) {
        mesh.vertex_normals_.clear();
    }
    return true;
}

bool WriteTriangleMeshToOBJ(const std::string& filename,
                            const geometry::TriangleMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/) {
    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);

    if (!file) {
        utility::PrintWarning("Write OBJ failed: unable to open file.\n");
        return false;
    }

    if (mesh.HasTriangleNormals()) {
        utility::PrintWarning("Write OBJ can not include triangle normals.\n");
    }

    file << "# Created by Open3D \n";
    file << "# number of vertices: " << mesh.vertices_.size() << "\n";
    file << "# number of triangles: " << mesh.triangles_.size() << "\n";
    utility::ResetConsoleProgress(
            mesh.vertices_.size() + mesh.triangles_.size(), "Writing OBJ: ");
    bool has_vertex_normals = mesh.HasVertexNormals();
    bool has_vertex_colors = mesh.HasVertexColors();
    for (int vidx = 0; vidx < mesh.vertices_.size(); ++vidx) {
        const Eigen::Vector3d& vertex = mesh.vertices_[vidx];
        file << "v " << vertex(0) << " " << vertex(1) << " " << vertex(2);
        if (has_vertex_colors) {
            const Eigen::Vector3d& color = mesh.vertex_colors_[vidx];
            file << " " << color(0) << " " << color(1) << " " << color(2);
        }
        file << "\n";

        if (has_vertex_normals) {
            const Eigen::Vector3d& normal = mesh.vertex_normals_[vidx];
            file << "vn " << normal(0) << " " << normal(1) << " " << normal(2)
                 << "\n";
        }

        utility::AdvanceConsoleProgress();
    }

    for (int tidx = 0; tidx < mesh.triangles_.size(); ++tidx) {
        const Eigen::Vector3i& triangle = mesh.triangles_[tidx];
        if (has_vertex_normals) {
            file << "f " << triangle(0) + 1 << "//" << triangle(0) + 1 << " "
                 << triangle(1) + 1 << "//" << triangle(1) + 1 << " "
                 << triangle(2) + 1 << "//" << triangle(2) + 1 << "\n";
        } else {
            file << "f " << triangle(0) + 1 << " " << triangle(1) + 1 << " "
                 << triangle(2) + 1 << "\n";
        }
        utility::AdvanceConsoleProgress();
    }

    return true;
}

}  // namespace io
}  // namespace open3d
