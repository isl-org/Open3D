// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/TriangleMeshIO.h"

#include <set>
#include <unordered_map>

#include "open3d/t/io/NumpyIO.h"
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
        file_extension_to_trianglemesh_read_function{
                {"npz", ReadTriangleMeshFromNPZ},
                {"stl", ReadTriangleMeshUsingASSIMP},
                {"obj", ReadTriangleMeshUsingASSIMP},
                {"off", ReadTriangleMeshUsingASSIMP},
                {"gltf", ReadTriangleMeshUsingASSIMP},
                {"glb", ReadTriangleMeshUsingASSIMP},
                {"fbx", ReadTriangleMeshUsingASSIMP},
        };

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
        file_extension_to_trianglemesh_write_function{
                {"npz", WriteTriangleMeshToNPZ},
                {"glb", WriteTriangleMeshUsingASSIMP},
        };

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

bool ReadTriangleMeshFromNPZ(
        const std::string &filename,
        geometry::TriangleMesh &mesh,
        const open3d::io::ReadTriangleMeshOptions &params) {
    auto attribute_map = ReadNpz(filename);

    // At a minimum there should be 'vertices' and 'triangles'
    if (!(attribute_map.count("vertices") > 0) ||
        !(attribute_map.count("triangles") > 0)) {
        utility::LogWarning(
                "Read geometry::TriangleMesh failed: Could not find 'vertices' "
                "or 'triangles' attributes in {}",
                filename);
        return false;
    }

    // Fill mesh with attributes
    for (auto &attr : attribute_map) {
        if (attr.first == "vertices") {
            mesh.SetVertexPositions(attr.second);
        } else if (attr.first == "triangles") {
            mesh.SetTriangleIndices(attr.second);
        } else if (attr.first == "vertex_normals") {
            mesh.SetVertexNormals(attr.second);
        } else if (attr.first == "triangle_normals") {
            mesh.SetTriangleNormals(attr.second);
        } else if (attr.first == "vertex_colors") {
            mesh.SetVertexColors(attr.second);
        } else if (attr.first == "triangle_colors") {
            mesh.SetTriangleColors(attr.second);
        } else if (attr.first == "triangle_texture_uvs") {
            mesh.SetTriangleAttr("texture_uvs", attr.second);
        } else if (attr.first.find("tex_") != std::string::npos) {
            // Get texture map
            auto key = attr.first.substr(4);
            if (!mesh.GetMaterial().IsValid()) {
                mesh.GetMaterial().SetDefaultProperties();
            }
            mesh.GetMaterial().SetTextureMap(key, geometry::Image(attr.second));
            // Note: due to quirk of Open3D shader implementation if we have a
            // metallic texture we need to set the metallic scalar propert to
            // 1.0
            if (key == "metallic") {
                mesh.GetMaterial().SetScalarProperty("metallic", 1.0);
            }
        } else if (attr.first.find("vertex_") != std::string::npos) {
            // Generic vertex attribute
            auto key = attr.first.substr(7);
            mesh.SetVertexAttr(key, attr.second);
        } else if (attr.first.find("triangle_") != std::string::npos) {
            // Generic triangle attribute
            auto key = attr.first.substr(9);
            mesh.SetTriangleAttr(key, attr.second);
        } else if (attr.first == "material_name") {
            if (!mesh.GetMaterial().IsValid()) {
                mesh.GetMaterial().SetDefaultProperties();
            }
            const uint8_t *str_ptr = attr.second.GetDataPtr<uint8_t>();
            std::string mat_name(attr.second.GetShape().GetLength(), 'a');
            std::memcpy((void *)mat_name.data(), str_ptr,
                        attr.second.GetShape().GetLength());
            mesh.GetMaterial().SetMaterialName(mat_name);
        }
    }

    return true;
}

bool WriteTriangleMeshToNPZ(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            const bool write_ascii,
                            const bool compressed,
                            const bool write_vertex_normals,
                            const bool write_vertex_colors,
                            const bool write_triangle_uvs,
                            const bool print_progress) {
    // Sanity checks...
    if (write_ascii) {
        utility::LogWarning(
                "TriangleMesh can't be saved in ASCII fromat as .npz");
        return false;
    }
    if (compressed) {
        utility::LogWarning(
                "TriangleMesh can't be saved in compressed format as .npz");
        return false;
    }

    // Map attribute names to names already used by convention in other software
    std::set<std::string> known_attributes(
            {"positions", "normals", "texture_uvs", "indices", "colors"});

    // Build map of known attributes
    std::unordered_map<std::string, core::Tensor> mesh_attributes;
    if (mesh.HasVertexPositions()) {
        mesh_attributes["vertices"] = mesh.GetVertexPositions();
    }
    if (mesh.HasVertexNormals()) {
        mesh_attributes["vertex_normals"] = mesh.GetVertexNormals();
    }
    if (mesh.HasVertexColors()) {
        mesh_attributes["vertex_colors"] = mesh.GetVertexColors();
    }
    if (mesh.HasTriangleIndices()) {
        mesh_attributes["triangles"] = mesh.GetTriangleIndices();
    }
    if (mesh.HasTriangleNormals()) {
        mesh_attributes["triangle_normals"] = mesh.GetTriangleNormals();
    }
    if (mesh.HasTriangleColors()) {
        mesh_attributes["triangle_colors"] = mesh.GetTriangleColors();
    }
    if (mesh.HasTriangleAttr("texture_uvs")) {
        mesh_attributes["triangle_texture_uvs"] =
                mesh.GetTriangleAttr("texture_uvs");
    }

    // Add "generic" attributes
    for (auto attr : mesh.GetVertexAttr()) {
        if (known_attributes.count(attr.first) > 0) {
            continue;
        }
        std::string key_name("vertex_");
        key_name += attr.first;
        mesh_attributes[key_name] = attr.second;
    }
    for (auto attr : mesh.GetTriangleAttr()) {
        if (known_attributes.count(attr.first) > 0) {
            continue;
        }
        std::string key_name("triangle_");
        key_name += attr.first;
        mesh_attributes[key_name] = attr.second;
    }

    // Output texture maps
    if (mesh.GetMaterial().IsValid()) {
        auto &mat = mesh.GetMaterial();
        // Get material name in Tensor form
        std::vector<uint8_t> mat_name_vec(
                {mat.GetMaterialName().begin(), mat.GetMaterialName().end()});
        core::Tensor mat_name_tensor(std::move(mat_name_vec));
        mesh_attributes["material_name"] = mat_name_tensor;
        for (auto &tex : mat.GetTextureMaps()) {
            std::string key = std::string("tex_") + tex.first;
            mesh_attributes[key] = tex.second.AsTensor();
        }
    }

    WriteNpz(filename, mesh_attributes);

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
