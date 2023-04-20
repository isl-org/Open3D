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

#include "open3d/visualization/rendering/Model.h"

namespace open3d {
namespace visualization {
namespace rendering {

MaterialRecord ConvertMaterial(
        const geometry::TriangleMesh::Material& mat,
        const std::string& name = "Material") {
    MaterialRecord out;
    out.name = name;
    // Base attributes
    out.base_color = Eigen::Vector4f(mat.baseColor.f4);
    out.base_metallic = mat.baseMetallic;
    out.base_roughness = mat.baseRoughness;
    out.base_reflectance = mat.baseReflectance;
    out.base_clearcoat = mat.baseClearCoat;
    out.base_clearcoat_roughness = mat.baseClearCoatRoughness;
    out.base_anisotropy = mat.baseAnisotropy;
    // Image attributes
    out.albedo_img = mat.albedo;
    out.normal_img = mat.normalMap;
    out.ao_img = mat.ambientOcclusion;
    out.metallic_img = mat.metallic;
    out.roughness_img = mat.roughness;
    out.reflectance_img = mat.reflectance;
    out.clearcoat_img = mat.clearCoat;
    out.clearcoat_roughness_img = mat.clearCoatRoughness;
    out.anisotropy_img = mat.anisotropy;
    // Dictionary attributes
    for (const auto& kvpair: mat.floatParameters) {
        out.generic_params[kvpair.first] =
                Eigen::Vector4f(kvpair.second.f4);
    }
    for (const auto& kvpair: mat.additionalMaps) {
        out.generic_imgs[kvpair.first] = kvpair.second;
    }
    return out;
}

// Split mesh into components based on material id
geometry::TriangleMesh GetComponentForMaterial(
        const geometry::TriangleMesh& mesh,
        int mat_idx) {

    geometry::TriangleMesh component;
    for (std::size_t tidx = 0; tidx < mesh.triangles_.size(); ++tidx) {
        if (mesh.triangle_material_ids_[tidx] != mat_idx) {
            continue;
        }
        // Copy over per-triangle attributes
        component.triangles_.emplace_back(mesh.triangles_[tidx]);
        if (mesh.HasTriangleUvs()) {
            component.triangle_uvs_.emplace_back(
                    mesh.triangle_uvs_[tidx]);
        }
        if (mesh.HasTriangleNormals()) {
            component.triangle_normals_.emplace_back(
                    mesh.triangle_normals_[tidx]);
        }
    }
    // Create a mapping of old indices to new indices
    std::unordered_map<int, int> vidx_remap;
    for (const Eigen::Vector3i& tri: component.triangles_) {
        for (int i = 0; i < 3; ++i) {
            if (vidx_remap.count(tri(i)) == 0) {
                vidx_remap[tri(i)] =
                        static_cast<int>(vidx_remap.size());
            }
        }
    }
    component.vertices_.resize(vidx_remap.size());
    if (mesh.HasVertexNormals()) {
        component.vertex_normals_.resize(vidx_remap.size());
    }
    if (mesh.HasVertexColors()) {
        component.vertex_colors_.resize(vidx_remap.size());
    }
    // Remap per-vertex data
    for (std::pair<int, int> remap: vidx_remap) {
        component.vertices_[remap.second] =
                mesh.vertices_[remap.first];
        if (mesh.HasVertexNormals()) {
            component.vertex_normals_[remap.second] =
                    mesh.vertex_normals_[remap.first];
        }
        if (mesh.HasVertexColors()) {
            component.vertex_colors_[remap.second] =
                    mesh.vertex_colors_[remap.first];
        }
    }
    for (auto& triangle : component.triangles_) {
        for (int i = 0; i < 3; ++i) {
            triangle(i) = vidx_remap[triangle(i)];
        }
    }
    return component;
}

void TriangleMeshModel::AddMesh(
        const geometry::TriangleMesh &mesh,
        const std::string &name) {
    // Reused code
    auto add_mesh_with_material =
            [this](const geometry::TriangleMesh& mesh,
                   const std::string& mesh_name,
                   const geometry::TriangleMesh::Material& material,
                   const std::string& material_name){
        MeshInfo mesh_info{
                std::make_shared<geometry::TriangleMesh>(mesh),
                mesh_name + " " + std::to_string(meshes_.size()),
                static_cast<unsigned int>(materials_.size())
        };
        meshes_.emplace_back(std::move(mesh_info));
        // Remove references to other materials
        auto& added_mesh = meshes_.back().mesh;
        added_mesh->triangle_material_ids_ = std::vector<int>(
                added_mesh->triangles_.size(), 0);
        added_mesh->materials_ = {std::make_pair(material_name, material)};
        added_mesh->textures_ = {*material.albedo->FlipVertical()};
        materials_.emplace_back(ConvertMaterial(material, material_name));
    };

    if (mesh.HasTriangleMaterialIds()) {
        // Check if all the material IDs are the same
        auto minmax_iters = std::minmax_element(
                mesh.triangle_material_ids_.begin(),
                mesh.triangle_material_ids_.end());
        if (*minmax_iters.first == *minmax_iters.second) {
            std::pair<std::string, geometry::TriangleMesh::Material> mat =
                    mesh.materials_[*minmax_iters.first];
            add_mesh_with_material(mesh, name, mat.second, mat.first);
        } else {
            // Split the mesh into components, one for each material
            std::unordered_set<int> unique_material_indices;
            unique_material_indices.insert(
                    mesh.triangle_material_ids_.begin(),
                    mesh.triangle_material_ids_.end());
            for (int mat_idx: unique_material_indices) {
                std::pair<std::string, geometry::TriangleMesh::Material> mat =
                        mesh.materials_[mat_idx];
                const std::string component_name =
                        name + " " + std::to_string(mat_idx);
                add_mesh_with_material(
                        mesh, component_name, mat.second, mat.first);
            }
        }
    } else {
        if (mesh.materials_.empty()) {
            utility::LogWarning("No material found for triangle mesh, "
                    "creating default material");
        } else {
            // Search for Texture or DefaultMaterial
            // otherwise use the first material
            auto material_it = std::find_if(
                    mesh.materials_.begin(), mesh.materials_.end(),
                    [](const auto& mat_name_pair){
                return mat_name_pair.first == "Texture";
            });
            if (material_it == mesh.materials_.end()) {
                material_it = std::find_if(
                        mesh.materials_.begin(), mesh.materials_.end(),
                        [](const auto& mat_name_pair){
                    return mat_name_pair.first == "DefaultMaterial";
                });
                if (material_it == mesh.materials_.end()) {
                    material_it = mesh.materials_.begin();
                }
            }
            std::pair<std::string, geometry::TriangleMesh::Material> mat =
                    *material_it;
            add_mesh_with_material(mesh, name, mat.second, mat.first);
        }
    }
}

TriangleMeshModel TriangleMeshModel::FromTriangleMesh(
        const geometry::TriangleMesh &mesh,
        const std::string &name) {
    TriangleMeshModel out;
    out.AddMesh(mesh, name);
    return out;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
