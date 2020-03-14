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

#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

namespace open3d {
namespace io {

bool ReadTriangleMeshFromOBJ(const std::string& filename,
                             geometry::TriangleMesh& mesh,
                             bool print_progress) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    std::string mtl_base_path =
            utility::filesystem::GetFileParentDirectory(filename);
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filename.c_str(), mtl_base_path.c_str());
    if (!warn.empty()) {
        utility::LogWarning("Read OBJ failed: {}", warn);
    }
    if (!err.empty()) {
        utility::LogWarning("Read OBJ failed: {}", err);
    }

    if (!ret) {
        return false;
    }

    mesh.Clear();

    // copy vertex and data
    for (size_t vidx = 0; vidx < attrib.vertices.size(); vidx += 3) {
        tinyobj::real_t vx = attrib.vertices[vidx + 0];
        tinyobj::real_t vy = attrib.vertices[vidx + 1];
        tinyobj::real_t vz = attrib.vertices[vidx + 2];
        mesh.vertices_.push_back(Eigen::Vector3d(vx, vy, vz));
    }

    for (size_t vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
        tinyobj::real_t r = attrib.colors[vidx + 0];
        tinyobj::real_t g = attrib.colors[vidx + 1];
        tinyobj::real_t b = attrib.colors[vidx + 2];
        mesh.vertex_colors_.push_back(Eigen::Vector3d(r, g, b));
    }

    // resize normal data and create bool indicator vector
    mesh.vertex_normals_.resize(mesh.vertices_.size());
    std::vector<bool> normals_indicator(mesh.vertices_.size(), false);

    // copy face data and copy normals data
    // append face-wise uv data
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) {
                utility::LogWarning(
                        "Read OBJ failed: facet with number of vertices not "
                        "equal to 3");
                return false;
            }

            Eigen::Vector3i facet;
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                int vidx = idx.vertex_index;
                facet(v) = vidx;

                if (!attrib.normals.empty() && !normals_indicator[vidx] &&
                    (3 * idx.normal_index + 2) < int(attrib.normals.size())) {
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

                if (!attrib.texcoords.empty() &&
                    2 * idx.texcoord_index + 1 < int(attrib.texcoords.size())) {
                    tinyobj::real_t tx =
                            attrib.texcoords[2 * idx.texcoord_index + 0];
                    tinyobj::real_t ty =
                            attrib.texcoords[2 * idx.texcoord_index + 1];
                    mesh.triangle_uvs_.push_back(Eigen::Vector2d(tx, ty));
                }
            }
            mesh.triangles_.push_back(facet);
            mesh.triangle_material_ids_.push_back(
                    shapes[s].mesh.material_ids[f]);
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

    // if not all triangles have corresponding uvs, then remove uvs
    if (3 * mesh.triangles_.size() != mesh.triangle_uvs_.size()) {
        mesh.triangle_uvs_.clear();
    }

    // Now we assert only one shape is stored, we only select the first
    // diffuse material
    for (auto& material : materials) {
        if (!material.diffuse_texname.empty()) {
            mesh.textures_.push_back(
                    *(io::CreateImageFromFile(mtl_base_path +
                                              material.diffuse_texname)
                              ->FlipVertical()));
        }
    }

    return true;
}

bool WriteTriangleMeshToOBJ(const std::string& filename,
                            const geometry::TriangleMesh& mesh,
                            bool write_ascii /* = false*/,
                            bool compressed /* = false*/,
                            bool write_vertex_normals /* = true*/,
                            bool write_vertex_colors /* = true*/,
                            bool write_triangle_uvs /* = true*/,
                            bool print_progress) {
    std::string object_name = utility::filesystem::GetFileNameWithoutExtension(
            utility::filesystem::GetFileNameWithoutDirectory(filename));

    std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
    if (!file) {
        utility::LogWarning("Write OBJ failed: unable to open file.");
        return false;
    }

    if (mesh.HasTriangleNormals()) {
        utility::LogWarning("Write OBJ can not include triangle normals.");
    }

    file << "# Created by Open3D " << std::endl;
    file << "# object name: " << object_name << std::endl;
    file << "# number of vertices: " << mesh.vertices_.size() << std::endl;
    file << "# number of triangles: " << mesh.triangles_.size() << std::endl;

    // always write material filename in obj file, regardless of uvs or textures
    file << "mtllib " << object_name << ".mtl" << std::endl;

    utility::ConsoleProgressBar progress_bar(
            mesh.vertices_.size() + mesh.triangles_.size(),
            "Writing OBJ: ", print_progress);

    write_vertex_normals = write_vertex_normals && mesh.HasVertexNormals();
    write_vertex_colors = write_vertex_colors && mesh.HasVertexColors();
    for (size_t vidx = 0; vidx < mesh.vertices_.size(); ++vidx) {
        const Eigen::Vector3d& vertex = mesh.vertices_[vidx];
        file << "v " << vertex(0) << " " << vertex(1) << " " << vertex(2);
        if (write_vertex_colors) {
            const Eigen::Vector3d& color = mesh.vertex_colors_[vidx];
            file << " " << color(0) << " " << color(1) << " " << color(2);
        }
        file << std::endl;

        if (write_vertex_normals) {
            const Eigen::Vector3d& normal = mesh.vertex_normals_[vidx];
            file << "vn " << normal(0) << " " << normal(1) << " " << normal(2)
                 << std::endl;
        }

        ++progress_bar;
    }

    // we are less strict and allows writing to uvs without known material
    // potentially this will be useful for exporting conformal map generation
    write_triangle_uvs = write_triangle_uvs && mesh.HasTriangleUvs();

    // we don't compress uvs into vertex-wise representation.
    // loose triangle-wise representation is provided
    if (write_triangle_uvs) {
        for (auto& uv : mesh.triangle_uvs_) {
            file << "vt " << uv(0) << " " << uv(1) << std::endl;
        }
    }

    // write faces with (possibly multiple) material ids
    // map faces with material ids
    std::map<int, std::vector<size_t>> material_id_faces_map;
    if (mesh.HasTriangleMaterialIds()) {
        for (size_t i = 0; i < mesh.triangle_material_ids_.size(); ++i) {
            int mi = mesh.triangle_material_ids_[i];
            auto it = material_id_faces_map.find(mi);
            if (it == material_id_faces_map.end()) {
                material_id_faces_map[mi] = {i};
            } else {
                it->second.push_back(i);
            }
        }
    } else {  // every face falls to the default material
        material_id_faces_map[0].resize(mesh.triangles_.size());
        std::iota(material_id_faces_map[0].begin(),
                  material_id_faces_map[0].end(), 0);
    }

    // enumerate ids and their corresponding faces
    for (auto it = material_id_faces_map.begin();
         it != material_id_faces_map.end(); ++it) {
        // write the mtl name
        std::string mtl_name = object_name + "_" + std::to_string(it->first);
        file << "usemtl " << mtl_name << std::endl;

        // write the corresponding faces
        for (auto tidx : it->second) {
            const Eigen::Vector3i& triangle = mesh.triangles_[tidx];
            if (write_vertex_normals && write_triangle_uvs) {
                file << "f ";
                file << triangle(0) + 1 << "/" << 3 * tidx + 1 << "/"
                     << triangle(0) + 1 << " ";
                file << triangle(1) + 1 << "/" << 3 * tidx + 2 << "/"
                     << triangle(1) + 1 << " ";
                file << triangle(2) + 1 << "/" << 3 * tidx + 3 << "/"
                     << triangle(2) + 1 << std::endl;
            } else if (!write_vertex_normals && write_triangle_uvs) {
                file << "f ";
                file << triangle(0) + 1 << "/" << 3 * tidx + 1 << " ";
                file << triangle(1) + 1 << "/" << 3 * tidx + 2 << " ";
                file << triangle(2) + 1 << "/" << 3 * tidx + 3 << std::endl;
            } else if (write_vertex_normals && !write_triangle_uvs) {
                file << "f " << triangle(0) + 1 << "//" << triangle(0) + 1
                     << " " << triangle(1) + 1 << "//" << triangle(1) + 1 << " "
                     << triangle(2) + 1 << "//" << triangle(2) + 1 << std::endl;
            } else {
                file << "f " << triangle(0) + 1 << " " << triangle(1) + 1 << " "
                     << triangle(2) + 1 << std::endl;
            }
            ++progress_bar;
        }
    }
    // end of writing obj.
    //////

    //////
    // start to write to mtl and texture
    std::string parent_dir =
            utility::filesystem::GetFileParentDirectory(filename);
    std::string mtl_filename = parent_dir + object_name + ".mtl";

    // write headers
    std::ofstream mtl_file(mtl_filename.c_str(), std::ios::out);
    if (!mtl_file) {
        utility::LogWarning(
                "Write OBJ successful, but failed to write material file.");
        return true;
    }
    mtl_file << "# Created by Open3D " << std::endl;
    mtl_file << "# object name: " << object_name << std::endl;

    // write textures (if existing)
    for (size_t i = 0; i < mesh.textures_.size(); ++i) {
        std::string mtl_name = object_name + "_" + std::to_string(i);
        mtl_file << "newmtl " << mtl_name << std::endl;
        mtl_file << "Ka 1.000 1.000 1.000" << std::endl;
        mtl_file << "Kd 1.000 1.000 1.000" << std::endl;
        mtl_file << "Ks 0.000 0.000 0.000" << std::endl;
        if (write_triangle_uvs && mesh.HasTextures()) {
            std::string tex_filename = parent_dir + mtl_name + ".png";
            if (!io::WriteImage(tex_filename,
                                *mesh.textures_[i].FlipVertical())) {
                utility::LogWarning(
                        "Write OBJ successful, but failed to write texture "
                        "file.");
                return true;
            }
            mtl_file << "map_Kd " << mtl_name << ".png\n";
        }
    }

    // write the default material
    if (!mesh.HasTextures()) {
        std::string mtl_name = object_name + "_0";
        mtl_file << "newmtl " << mtl_name << std::endl;
        mtl_file << "Ka 1.000 1.000 1.000" << std::endl;
        mtl_file << "Kd 1.000 1.000 1.000" << std::endl;
        mtl_file << "Ks 0.000 0.000 0.000" << std::endl;
    }

    return true;
}

}  // namespace io
}  // namespace open3d
