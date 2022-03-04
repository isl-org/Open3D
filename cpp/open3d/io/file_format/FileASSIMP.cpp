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

#include <fstream>
#include <numeric>
#include <vector>

#include "assimp/GltfMaterial.h"
#include "assimp/Importer.hpp"
#include "assimp/ProgressHandler.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/ModelIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Model.h"

#define AI_MATKEY_CLEARCOAT_THICKNESS "$mat.clearcoatthickness", 0, 0
#define AI_MATKEY_CLEARCOAT_ROUGHNESS "$mat.clearcoatroughness", 0, 0
#define AI_MATKEY_SHEEN "$mat.sheen", 0, 0
#define AI_MATKEY_ANISOTROPY "$mat.anisotropy", 0, 0

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeFBX(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

const unsigned int kPostProcessFlags_compulsory =
        aiProcess_JoinIdenticalVertices;

const unsigned int kPostProcessFlags_fast =
        aiProcessPreset_TargetRealtime_Fast |
        aiProcess_RemoveRedundantMaterials | aiProcess_OptimizeMeshes |
        aiProcess_PreTransformVertices;

struct TextureImages {
    std::shared_ptr<geometry::Image> albedo;
    std::shared_ptr<geometry::Image> normal;
    std::shared_ptr<geometry::Image> ao;
    std::shared_ptr<geometry::Image> roughness;
    std::shared_ptr<geometry::Image> metallic;
    std::shared_ptr<geometry::Image> reflectance;
    std::shared_ptr<geometry::Image> clearcoat;
    std::shared_ptr<geometry::Image> clearcoat_roughness;
    std::shared_ptr<geometry::Image> anisotropy;
    std::shared_ptr<geometry::Image> gltf_rough_metal;
};

void LoadTextures(const std::string& filename,
                  aiMaterial* mat,
                  TextureImages& maps) {
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
            // normalize path separators
            auto p_win = strpath.find("\\");
            while (p_win != std::string::npos) {
                strpath[p_win] = '/';
                p_win = strpath.find("\\", p_win + 1);
            }
            // if absolute path convert to relative to base path
            if (strpath.length() > 1 &&
                (strpath[0] == '/' || strpath[1] == ':')) {
                strpath = utility::filesystem::GetFileNameWithoutDirectory(
                        strpath);
            }
            auto image = io::CreateImageFromFile(base_path + strpath);
            if (image->HasData()) {
                img = image;
            }
        }
    };

    texture_loader(aiTextureType_DIFFUSE, maps.albedo);
    texture_loader(aiTextureType_NORMALS, maps.normal);
    // Assimp may place ambient occlusion texture in AMBIENT_OCCLUSION if
    // format has AO support. Prefer that texture if it is preset. Otherwise,
    // try AMBIENT where OBJ and FBX typically put AO textures.
    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
        texture_loader(aiTextureType_AMBIENT_OCCLUSION, maps.ao);
    } else {
        texture_loader(aiTextureType_AMBIENT, maps.ao);
    }
    texture_loader(aiTextureType_METALNESS, maps.metallic);
    if (mat->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
        texture_loader(aiTextureType_DIFFUSE_ROUGHNESS, maps.roughness);
    } else if (mat->GetTextureCount(aiTextureType_SHININESS) > 0) {
        // NOTE: In some FBX files assimp puts the roughness texture in
        // shininess slot
        texture_loader(aiTextureType_SHININESS, maps.roughness);
    }
    // NOTE: Assimp doesn't have a texture type for GLTF's combined
    // roughness/metallic texture so it puts it in the 'unknown' texture slot
    texture_loader(aiTextureType_UNKNOWN, maps.gltf_rough_metal);
    // NOTE: the following may be non-standard. We are using REFLECTION texture
    // type to store OBJ map_Ps 'sheen' PBR map
    texture_loader(aiTextureType_REFLECTION, maps.reflectance);

    // NOTE: ASSIMP doesn't appear to provide texture params for the following:
    // clearcoat
    // clearcoat_roughness
    // anisotropy
}

bool ReadTriangleMeshUsingASSIMP(
        const std::string& filename,
        geometry::TriangleMesh& mesh,
        const ReadTriangleMeshOptions& params /*={}*/) {
    Assimp::Importer importer;

    unsigned int post_process_flags = kPostProcessFlags_compulsory;

    if (params.enable_post_processing) {
        post_process_flags = kPostProcessFlags_fast;
    }

    const auto* scene = importer.ReadFile(filename.c_str(), post_process_flags);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP", filename);
        return false;
    }

    mesh.Clear();

    size_t current_vidx = 0;
    // Merge individual meshes in aiScene into a single TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // Only process triangle meshes
        if (assimp_mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            utility::LogInfo(
                    "Skipping non-triangle primitive geometry of type: "
                    "{}",
                    assimp_mesh->mPrimitiveTypes);
            continue;
        }

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
            mesh.triangle_material_ids_.push_back(assimp_mesh->mMaterialIndex);
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

    // Now load the materials
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        auto* mat = scene->mMaterials[i];

        // create material structure to match this name
        auto& mesh_material =
                mesh.materials_[std::string(mat->GetName().C_Str())];

        using MaterialParameter =
                geometry::TriangleMesh::Material::MaterialParameter;

        // Retrieve base material properties
        aiColor3D color(1.f, 1.f, 1.f);

        mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        mesh_material.baseColor =
                MaterialParameter::CreateRGB(color.r, color.g, color.b);
        mat->Get(AI_MATKEY_METALLIC_FACTOR, mesh_material.baseMetallic);
        mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, mesh_material.baseRoughness);
        // NOTE: We prefer sheen to reflectivity so the following code works
        // since if sheen is not present it won't modify baseReflectance
        mat->Get(AI_MATKEY_REFLECTIVITY, mesh_material.baseReflectance);
        mat->Get(AI_MATKEY_SHEEN, mesh_material.baseReflectance);

        mat->Get(AI_MATKEY_CLEARCOAT_THICKNESS, mesh_material.baseClearCoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS,
                 mesh_material.baseClearCoatRoughness);
        mat->Get(AI_MATKEY_ANISOTROPY, mesh_material.baseAnisotropy);

        // Retrieve textures
        TextureImages maps;
        LoadTextures(filename, mat, maps);
        mesh_material.albedo = maps.albedo;
        mesh_material.normalMap = maps.normal;
        mesh_material.ambientOcclusion = maps.ao;
        mesh_material.metallic = maps.metallic;
        mesh_material.roughness = maps.roughness;
        mesh_material.reflectance = maps.reflectance;

        // For legacy visualization support
        if (mesh_material.albedo) {
            mesh.textures_.push_back(*mesh_material.albedo->FlipVertical());
        } else {
            mesh.textures_.push_back(geometry::Image());
        }
    }

    return true;
}

bool ReadModelUsingAssimp(const std::string& filename,
                          visualization::rendering::TriangleMeshModel& model,
                          const ReadTriangleModelOptions& params /*={}*/) {
    int64_t progress_total = 100;  // 70: ReadFile(), 10: mesh, 20: textures
    float readfile_total = 70.0f;
    float mesh_total = 10.0f;
    float textures_total = 20.0f;
    int64_t progress = 0;
    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(progress_total);
    class AssimpProgress : public Assimp::ProgressHandler {
    public:
        AssimpProgress(const ReadTriangleModelOptions& params, float scaling)
            : params_(params), scaling_(scaling) {}

        bool Update(float percentage = -1.0f) override {
            if (params_.update_progress) {
                params_.update_progress(
                        std::max(0.0f, 100.0f * scaling_ * percentage));
            }
            return true;
        }

    private:
        const ReadTriangleModelOptions& params_;
        float scaling_;
    };

    Assimp::Importer importer;
    // The importer takes ownership of the pointer (the documentation
    // is silent on this salient point).
    importer.SetProgressHandler(
            new AssimpProgress(params, readfile_total / progress_total));
    const auto* scene =
            importer.ReadFile(filename.c_str(), kPostProcessFlags_fast);
    if (!scene) {
        utility::LogWarning("Unable to load file {} with ASSIMP", filename);
        return false;
    }

    progress = int64_t(readfile_total);
    reporter.Update(progress);

    // Process each Assimp mesh into a geometry::TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // Only process triangle meshes
        if (assimp_mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            utility::LogInfo(
                    "Skipping non-triangle primitive geometry of type: "
                    "{}",
                    assimp_mesh->mPrimitiveTypes);
            continue;
        }

        std::shared_ptr<geometry::TriangleMesh> mesh =
                std::make_shared<geometry::TriangleMesh>();

        // copy vertex data
        for (size_t vidx = 0; vidx < assimp_mesh->mNumVertices; ++vidx) {
            auto& vertex = assimp_mesh->mVertices[vidx];
            mesh->vertices_.push_back(
                    Eigen::Vector3d(vertex.x, vertex.y, vertex.z));
        }

        // copy face indices data
        for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
            auto& face = assimp_mesh->mFaces[fidx];
            Eigen::Vector3i facet(face.mIndices[0], face.mIndices[1],
                                  face.mIndices[2]);
            mesh->triangles_.push_back(facet);
        }

        if (assimp_mesh->mNormals) {
            for (size_t nidx = 0; nidx < assimp_mesh->mNumVertices; ++nidx) {
                auto& normal = assimp_mesh->mNormals[nidx];
                mesh->vertex_normals_.push_back({normal.x, normal.y, normal.z});
            }
        }

        // NOTE: only use the first UV channel
        if (assimp_mesh->HasTextureCoords(0)) {
            for (size_t fidx = 0; fidx < assimp_mesh->mNumFaces; ++fidx) {
                auto& face = assimp_mesh->mFaces[fidx];
                auto& uv1 = assimp_mesh->mTextureCoords[0][face.mIndices[0]];
                auto& uv2 = assimp_mesh->mTextureCoords[0][face.mIndices[1]];
                auto& uv3 = assimp_mesh->mTextureCoords[0][face.mIndices[2]];
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv1.x, uv1.y));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv2.x, uv2.y));
                mesh->triangle_uvs_.push_back(Eigen::Vector2d(uv3.x, uv3.y));
            }
        }

        // NOTE: only use the first color attribute
        if (assimp_mesh->HasVertexColors(0)) {
            for (size_t cidx = 0; cidx < assimp_mesh->mNumVertices; ++cidx) {
                auto& c = assimp_mesh->mColors[0][cidx];
                mesh->vertex_colors_.push_back({c.r, c.g, c.b});
            }
        }

        // Add the mesh to the model
        model.meshes_.push_back({mesh, std::string(assimp_mesh->mName.C_Str()),
                                 assimp_mesh->mMaterialIndex});
    }

    progress = int64_t(readfile_total + mesh_total);
    reporter.Update(progress);

    // Load materials
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        auto* mat = scene->mMaterials[i];

        visualization::rendering::MaterialRecord o3d_mat;

        o3d_mat.name = mat->GetName().C_Str();

        // Retrieve base material properties
        aiColor3D color(1.f, 1.f, 1.f);

        mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        o3d_mat.base_color = Eigen::Vector4f(color.r, color.g, color.b, 1.f);
        mat->Get(AI_MATKEY_METALLIC_FACTOR, o3d_mat.base_metallic);
        mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, o3d_mat.base_roughness);
        mat->Get(AI_MATKEY_REFLECTIVITY, o3d_mat.base_reflectance);
        mat->Get(AI_MATKEY_SHEEN, o3d_mat.base_reflectance);

        mat->Get(AI_MATKEY_CLEARCOAT_THICKNESS, o3d_mat.base_clearcoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS,
                 o3d_mat.base_clearcoat_roughness);
        mat->Get(AI_MATKEY_ANISOTROPY, o3d_mat.base_anisotropy);
        aiString alpha_mode;
        mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alpha_mode);
        std::string alpha_mode_str(alpha_mode.C_Str());
        if (alpha_mode_str == "BLEND" || alpha_mode_str == "MASK") {
            o3d_mat.has_alpha = true;
        }

        // Retrieve textures
        TextureImages maps;
        LoadTextures(filename, mat, maps);
        o3d_mat.albedo_img = maps.albedo;
        o3d_mat.normal_img = maps.normal;
        o3d_mat.ao_img = maps.ao;
        o3d_mat.metallic_img = maps.metallic;
        o3d_mat.roughness_img = maps.roughness;
        o3d_mat.reflectance_img = maps.reflectance;
        o3d_mat.ao_rough_metal_img = maps.gltf_rough_metal;

        if (o3d_mat.has_alpha) {
            o3d_mat.shader = "defaultLitTransparency";
        } else {
            o3d_mat.shader = "defaultLit";
        }

        model.materials_.push_back(o3d_mat);

        progress = int64_t(readfile_total + mesh_total +
                           textures_total * float(i + 1) /
                                   float(scene->mNumMaterials));
        reporter.Update(progress);
    }

    reporter.Update(progress_total);

    return true;
}

}  // namespace io
}  // namespace open3d
