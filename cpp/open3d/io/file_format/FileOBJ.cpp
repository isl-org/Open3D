// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <tiny_obj_loader.h>

#include <fstream>
#include <numeric>
#include <vector>

#include "open3d/io/FileFormatIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressBar.h"

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeOBJ(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromOBJ(const std::string& filename,
                             geometry::TriangleMesh& mesh,
                             const ReadTriangleMeshOptions& /*={}*/) {
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

    auto textureLoader = [&mtl_base_path](std::string& relativePath) {
        auto image = io::CreateImageFromFile(mtl_base_path + relativePath);
        return image->HasData() ? image : std::shared_ptr<geometry::Image>();
    };

    using MaterialParameter =
            geometry::TriangleMesh::Material::MaterialParameter;

    mesh.materials_.resize(materials.size());
    for (std::size_t i = 0; i < materials.size(); ++i) {
        auto& material = materials[i];
        mesh.materials_[i].first = material.name;
        auto& meshMaterial = mesh.materials_[i].second;

        meshMaterial.baseColor = MaterialParameter::CreateRGB(
                material.diffuse[0], material.diffuse[1], material.diffuse[2]);

        if (!material.normal_texname.empty()) {
            meshMaterial.normalMap = textureLoader(material.normal_texname);
        } else if (!material.bump_texname.empty()) {
            // try bump, because there is often a misunderstanding of
            // what bump map or normal map is
            meshMaterial.normalMap = textureLoader(material.bump_texname);
        }

        if (!material.ambient_texname.empty()) {
            meshMaterial.ambientOcclusion =
                    textureLoader(material.ambient_texname);
        }

        if (!material.diffuse_texname.empty()) {
            meshMaterial.albedo = textureLoader(material.diffuse_texname);

            // Legacy texture map support
            if (meshMaterial.albedo) {
                mesh.textures_.push_back(*meshMaterial.albedo->FlipVertical());
            }
        }

        if (!material.metallic_texname.empty()) {
            meshMaterial.metallic = textureLoader(material.metallic_texname);
        }

        if (!material.roughness_texname.empty()) {
            meshMaterial.roughness = textureLoader(material.roughness_texname);
        }

        if (!material.sheen_texname.empty()) {
            meshMaterial.reflectance = textureLoader(material.sheen_texname);
        }

        // NOTE: We want defaults of 0.0 and 1.0 for baseMetallic and
        // baseRoughness respectively but 0.0 is a valid value for both and
        // tiny_obj_loader defaults to 0.0 for both. So, we will assume that
        // only if one of the values is greater than 0.0 that there are
        // non-default values set in the .mtl file
        if (material.roughness > 0.f || material.metallic > 0.f) {
            meshMaterial.baseMetallic = material.metallic;
            meshMaterial.baseRoughness = material.roughness;
        }

        if (material.sheen > 0.f) {
            meshMaterial.baseReflectance = material.sheen;
        }

        // NOTE: We will unconditionally copy the following parameters because
        // the TinyObj defaults match Open3D's internal defaults
        meshMaterial.baseClearCoat = material.clearcoat_thickness;
        meshMaterial.baseClearCoatRoughness = material.clearcoat_roughness;
        meshMaterial.baseAnisotropy = material.anisotropy;
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

    utility::ProgressBar progress_bar(
            mesh.vertices_.size() + mesh.triangles_.size(),
            "Writing OBJ: ", print_progress);

    // we are less strict and allows writing to uvs without known material
    // potentially this will be useful for exporting conformal map generation
    write_triangle_uvs = write_triangle_uvs && mesh.HasTriangleUvs();

    // write material filename only when uvs is written or has textures
    if (write_triangle_uvs) {
        file << "mtllib " << object_name << ".mtl" << std::endl;
    }

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
        if (write_triangle_uvs) {
            std::string mtl_name =
                    object_name + "_" + std::to_string(it->first);
            file << "usemtl " << mtl_name << std::endl;
        }

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
    // write mtl file when uvs are written
    if (write_triangle_uvs) {
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

        // write the default material
        if (!mesh.HasTextures()) {
            std::string mtl_name = object_name + "_0";
            mtl_file << "newmtl " << mtl_name << std::endl;
            mtl_file << "Ka 1.000 1.000 1.000" << std::endl;
            mtl_file << "Kd 1.000 1.000 1.000" << std::endl;
            mtl_file << "Ks 0.000 0.000 0.000" << std::endl;
        }
    }

    return true;
}

}  // namespace io
}  // namespace open3d
