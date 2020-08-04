// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "assimp/Importer.hpp"
#include "assimp/pbrmaterial.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeFBX(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

bool ReadTriangleMeshFromASSIMP(const std::string& filename,
                                geometry::TriangleMesh& mesh,
                                bool print_progress) {
    Assimp::Importer importer;
    const auto* scene = importer.ReadFile(
            filename.c_str(), aiProcessPreset_TargetRealtime_MaxQuality |
                                      aiProcess_OptimizeGraph);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP", filename);
        return false;
    }

    // NOTE: Developer debug printout below. Commented out for now and will
    // eventually be removed entirely
    // utility::LogWarning("Loaded {}\n\tN MESHES: {}\n\tN MATERIALS: {}",
    // filename, scene->mNumMeshes, scene->mNumMaterials);
    // const auto* mesh1 = scene->mMeshes[0];
    // utility::LogWarning(
    //         "MESH: {}\n\tHas Positions: {}\n\tHas Normals: {}\n\tHasFaces: "
    //         "{}\n\tVertexColors: {}\n\tUV Channels: {}",
    //         mesh1->mName.C_Str(), mesh1->HasPositions(), mesh1->HasNormals(),
    //         mesh1->HasFaces(), mesh1->GetNumColorChannels(),
    //         mesh1->GetNumUVChannels());

    mesh.Clear();

    size_t current_vidx = 0;
    // Merge individual meshes in aiScene into a single TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // copy vertex data
        for (size_t vidx = 0; vidx < assimp_mesh->mNumVertices; ++vidx) {
            auto& vertex = assimp_mesh->mVertices[vidx];
            mesh.vertices_.push_back(
                    Eigen::Vector3d(vertex.x, vertex.y, vertex.z));
        }

        // copy face indices data
        for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
            auto& face = assimp_mesh->mFaces[fidx];
            Eigen::Vector3i facet(
                    face.mIndices[0] + static_cast<int>(current_vidx),
                    face.mIndices[1] + static_cast<int>(current_vidx),
                    face.mIndices[2] + static_cast<int>(current_vidx));
            mesh.triangles_.push_back(facet);
        }

        if (assimp_mesh->mNormals) {
            for (size_t nidx = 0; nidx < assimp_mesh->mNumVertices; ++nidx) {
                auto& normal = assimp_mesh->mNormals[nidx];
                mesh.vertex_normals_.push_back({normal.x, normal.y, normal.z});
            }
        }

        // NOTE: only support a single UV channel
        if (assimp_mesh->HasTextureCoords(0)) {
            for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
                auto& face = assimp_mesh->mFaces[fidx];
                auto& uv1 = assimp_mesh->mTextureCoords[0][face.mIndices[0]];
                auto& uv2 = assimp_mesh->mTextureCoords[0][face.mIndices[1]];
                auto& uv3 = assimp_mesh->mTextureCoords[0][face.mIndices[2]];
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv1.x, uv1.y));
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv2.x, uv2.y));
                mesh.triangle_uvs_.push_back(Eigen::Vector2d(uv3.x, uv3.y));
            }
        }

        // NOTE: only support a single per-vertex color attribute
        if (assimp_mesh->HasVertexColors(0)) {
            for (size_t cidx = 0; cidx < assimp_mesh->mNumVertices; ++cidx) {
                auto& c = assimp_mesh->mColors[0][cidx];
                mesh.vertex_colors_.push_back({c.r, c.g, c.b});
            }
        }

        // Adjust face indices to index into combined mesh vertex array
        current_vidx += assimp_mesh->mNumVertices;
    }

    // Load material data
    auto* mat = scene->mMaterials[0];

    // NOTE: Developer debug printouts below. To be removed soon.
    // utility::LogWarning("MATERIAL: {}\n\tPROPS: {}\n",
    // mat->GetName().C_Str(), mat->mNumProperties); for(size_t i = 0; i <
    // mat->mNumProperties; ++i) {
    //     auto* prop = mat->mProperties[i];
    //     utility::LogWarning("\tPROPNAME: {}\n", prop->mKey.C_Str());
    // }

    if (scene->mNumMaterials > 1) {
        utility::LogWarning(
                "{} has {} materials but only a single material per object is "
                "currently supported",
                filename, scene->mNumMaterials);
    }

    // create material structure to match this name
    auto& mesh_material = mesh.materials_[std::string(mat->GetName().C_Str())];

    using MaterialParameter =
            geometry::TriangleMesh::Material::MaterialParameter;

    // Retrieve base material properties
    aiColor3D color(1.f, 1.f, 1.f);

    mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    mesh_material.baseColor =
            MaterialParameter::CreateRGB(color.r, color.g, color.b);
    mat->Get(AI_MATKEY_REFLECTIVITY, mesh_material.baseReflectance);
    mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
             mesh_material.baseMetallic);
    mat->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
             mesh_material.baseRoughness);

    // Retrieve textures
    std::string base_path =
            utility::filesystem::GetFileParentDirectory(filename);

    auto texture_loader = [&base_path, &mat](
                                  aiTextureType type,
                                  std::shared_ptr<geometry::Image>& img) {
        if (mat->GetTextureCount(type) > 0) {
            aiString path;
            mat->GetTexture(type, 0, &path);
            std::string strpath(path.C_Str());
            auto p_win = strpath.rfind("\\");
            auto p_unix = strpath.rfind("/");
            if (p_win != std::string::npos) {
                strpath = strpath.substr(p_win + 1);
            } else if (p_unix != std::string::npos) {
                strpath = strpath.substr(p_unix + 1);
            }
            utility::LogWarning("TEXTURE PATH CLEAN: {} for texture type {}",
                                base_path + strpath, type);
            auto image = io::CreateImageFromFile(base_path + strpath);
            if (image->HasData()) {
                img = image;
            }
        }
    };

    texture_loader(aiTextureType_DIFFUSE, mesh_material.albedo);
    texture_loader(aiTextureType_NORMALS, mesh_material.normalMap);
    // Prefer ASSIMP's PBR version of ambient occlusion if available
    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
        texture_loader(aiTextureType_AMBIENT_OCCLUSION,
                       mesh_material.ambientOcclusion);
    } else {
        // NOTE: According to ASSIMP's material.h the LIGHTMAP texture type can
        // be used type name(args) const;or ambient occlusion textures
        texture_loader(aiTextureType_LIGHTMAP, mesh_material.ambientOcclusion);
    }
    texture_loader(aiTextureType_METALNESS, mesh_material.metallic);
    texture_loader(aiTextureType_DIFFUSE_ROUGHNESS, mesh_material.roughness);
    texture_loader(aiTextureType_UNKNOWN, mesh_material.roughness);
    // if(mat->GetTextureCount(aiTextureType_UNKNOWN) > 0 ) {
    //     for(size_t i = 0; i < mat->GetTextureCount(aiTextureType_UNKNOWN);
    //     ++i) {
    //         aiString path;
    //         mat->GetTexture(aiTextureType_UNKNOWN, i, &path);
    //         utility::LogWarning("Unknown texture type: {}", path.C_Str());
    //     }
    // }

    // NOTE: ASSIMP doesn't appear to provide texture params for the following
    // std::shared_ptr<Image> reflectance;
    // std::shared_ptr<Image> clearCoat;
    // std::shared_ptr<Image> clearCoatRoughness;
    // std::shared_ptr<Image> anisotropy;

    return true;
}

}  // namespace io
}  // namespace open3d
