// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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

namespace open3d {
namespace io {

FileGeometry ReadFileGeometryTypeFBX(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

FileGeometry ReadFileGeometryTypeUSD(const std::string& path) {
    return FileGeometry(CONTAINS_TRIANGLES | CONTAINS_POINTS);
}

// Ref:
// https://github.com/assimp/assimp/blob/master/include/assimp/postprocess.h
const unsigned int kPostProcessFlags_compulsory =
        aiProcess_JoinIdenticalVertices | aiProcess_SortByPType |
        aiProcess_PreTransformVertices;

const unsigned int kPostProcessFlags_fast =
        kPostProcessFlags_compulsory | aiProcess_GenNormals |
        aiProcess_Triangulate | aiProcess_GenUVCoords |
        aiProcess_RemoveRedundantMaterials | aiProcess_OptimizeMeshes;

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
                  const aiScene* scene,
                  const aiMaterial* mat,
                  TextureImages& maps) {
    // Retrieve textures
    std::string base_path =
            utility::filesystem::GetFileParentDirectory(filename);

    // Load texture at slot (type, index) into img. index defaults to 0 for
    // all standard single-slot types; multi-slot types (e.g. CLEARCOAT)
    // use explicit indices.
    auto texture_loader = [&base_path, &scene, &mat, &filename](
                                  aiTextureType type,
                                  std::shared_ptr<geometry::Image>& img,
                                  unsigned int index = 0) {
        if (mat->GetTextureCount(type) > index) {
            aiString path;
            mat->GetTexture(type, index, &path);

            // If the texture is an embedded texture, use `GetEmbeddedTexture`.
            if (auto texture = scene->GetEmbeddedTexture(path.C_Str())) {
                if (texture->mHeight == 0) {
                    // Compressed image: mWidth is the size in bytes. Detect
                    // PNG/JPEG from magic bytes (same as ImageIO::ReadImage).
                    auto image = io::CreateImageFromMemory(
                            "",
                            reinterpret_cast<const unsigned char*>(
                                    texture->pcData),
                            texture->mWidth);
                    if (image->HasData()) {
                        img = image;
                    } else {
                        utility::LogWarning(
                                "Unsupported or undecodable embedded texture "
                                "{} in file {}: only jpg and png are "
                                "supported.",
                                path.C_Str(), filename);
                    }
                } else {
                    // Uncompressed texels. The USD importer fills aiTexel as
                    // .b=R, .g=G, .r=B, .a=A, so the in-memory byte order is
                    // already [R, G, B, A]. Copy straight into the RGBA image.
                    auto image = std::make_shared<geometry::Image>();
                    image->Prepare(static_cast<int>(texture->mWidth),
                                   static_cast<int>(texture->mHeight), 4, 1);
                    const auto* texels = reinterpret_cast<const unsigned char*>(
                            texture->pcData);
                    const size_t num_bytes =
                            static_cast<size_t>(texture->mWidth) *
                            static_cast<size_t>(texture->mHeight) * 4;
                    for (size_t i = 0; i < num_bytes; ++i) {
                        image->data_[i] = texels[i];
                    }
                    img = image;
                }
            }
            // Else, build the path to it.
            else {
                std::string strpath(path.C_Str());
                // Normalize path separators.
                auto p_win = strpath.find("\\");
                while (p_win != std::string::npos) {
                    strpath[p_win] = '/';
                    p_win = strpath.find("\\", p_win + 1);
                }
                // If absolute path convert to relative to base path.
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
        }
    };

    // Prefer BASE_COLOR texture as assimp now uses it for PBR workflows
    if (mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
        texture_loader(aiTextureType_BASE_COLOR, maps.albedo);
    } else {
        texture_loader(aiTextureType_DIFFUSE, maps.albedo);
    }
    texture_loader(aiTextureType_NORMALS, maps.normal);
    // Ambient occlusion: Assimp uses different slots per format.
    // glTF 2.0 and USD importers use LIGHTMAP for occlusionTexture /
    // surfaceShader.occlusion; OBJ/FBX often use AMBIENT; some paths use
    // AMBIENT_OCCLUSION.
    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
        texture_loader(aiTextureType_AMBIENT_OCCLUSION, maps.ao);
    } else if (mat->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
        texture_loader(aiTextureType_LIGHTMAP, maps.ao);
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

    // Clearcoat: slot 0 = clearcoat factor, slot 1 = clearcoat roughness.
    // Anisotropy: single slot 0 (Assimp 6 aiTextureType_CLEARCOAT/ANISOTROPY).
    texture_loader(aiTextureType_CLEARCOAT, maps.clearcoat, 0);
    texture_loader(aiTextureType_CLEARCOAT, maps.clearcoat_roughness, 1);
    texture_loader(aiTextureType_ANISOTROPY, maps.anisotropy, 0);
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
        utility::LogWarning("Unable to load file {} with ASSIMP: {}", filename,
                            importer.GetErrorString());
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
    mesh.materials_.resize(scene->mNumMaterials);
    for (size_t i = 0; i < scene->mNumMaterials; ++i) {
        auto* mat = scene->mMaterials[i];

        // Set the material structure to match this name
        auto& mesh_material = mesh.materials_[i].second;
        mesh.materials_[i].first = mat->GetName().C_Str();

        using MaterialParameter =
                geometry::TriangleMesh::Material::MaterialParameter;

        // Retrieve base material properties.
        // Prefer PBR RGBA base color; fall back to legacy RGB diffuse.
        aiColor4D base_color4(1.f, 1.f, 1.f, 1.f);
        if (mat->Get(AI_MATKEY_BASE_COLOR, base_color4) != AI_SUCCESS) {
            aiColor3D c(1.f, 1.f, 1.f);
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, c);
            base_color4 = aiColor4D(c.r, c.g, c.b, 1.f);
        }
        mesh_material.baseColor = MaterialParameter(
                base_color4.r, base_color4.g, base_color4.b, base_color4.a);

        mat->Get(AI_MATKEY_METALLIC_FACTOR, mesh_material.baseMetallic);
        mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, mesh_material.baseRoughness);
        mat->Get(AI_MATKEY_REFLECTIVITY, mesh_material.baseReflectance);
        // Clearcoat and anisotropy — Assimp 6 key names
        mat->Get(AI_MATKEY_CLEARCOAT_FACTOR, mesh_material.baseClearCoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR,
                 mesh_material.baseClearCoatRoughness);
        mat->Get(AI_MATKEY_ANISOTROPY_FACTOR, mesh_material.baseAnisotropy);

        // Opacity for non-glTF formats (OBJ d / FBX Opacity)
        {
            float opacity = 1.f;
            mat->Get(AI_MATKEY_OPACITY, opacity);
            if (opacity < 1.f) {
                mesh_material.baseColor.f4[3] = opacity;
            }
        }

        // Retrieve textures
        TextureImages maps;
        LoadTextures(filename, scene, mat, maps);
        mesh_material.albedo = maps.albedo;
        mesh_material.normalMap = maps.normal;
        mesh_material.ambientOcclusion = maps.ao;
        mesh_material.metallic = maps.metallic;
        mesh_material.roughness = maps.roughness;
        mesh_material.reflectance = maps.reflectance;

        // For legacy visualization support
        if (mesh_material.albedo) {
            mesh.textures_.emplace_back(*mesh_material.albedo->FlipVertical());
        } else {
            mesh.textures_.emplace_back();
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
        utility::LogWarning("Unable to load file {} with ASSIMP: {}", filename,
                            importer.GetErrorString());
        return false;
    }

    if (scene->mNumMeshes == 0) {
        const std::string usd_expt =
                (filename.find(".usd") != std::string::npos)
                        ? " (USD import is experimental.)"
                        : "";
        utility::LogWarning("File {} loaded but produced no meshes.{}",
                            filename, usd_expt);
        return false;
    }

    progress = int64_t(readfile_total);
    reporter.Update(progress);

    // Process each Assimp mesh into a geometry::TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];
        // Only process triangle meshes
        if (!(assimp_mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)) {
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

        // Prefer PBR RGBA base color (glTF/USD); fall back to legacy RGB
        // diffuse
        aiColor4D base_color4(1.f, 1.f, 1.f, 1.f);
        if (mat->Get(AI_MATKEY_BASE_COLOR, base_color4) != AI_SUCCESS) {
            aiColor3D c(1.f, 1.f, 1.f);
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, c);
            base_color4 = aiColor4D(c.r, c.g, c.b, 1.f);
        }
        o3d_mat.base_color = Eigen::Vector4f(base_color4.r, base_color4.g,
                                             base_color4.b, base_color4.a);

        mat->Get(AI_MATKEY_METALLIC_FACTOR, o3d_mat.base_metallic);
        mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, o3d_mat.base_roughness);
        mat->Get(AI_MATKEY_REFLECTIVITY, o3d_mat.base_reflectance);
        // Clearcoat and anisotropy — Assimp 6 key names
        mat->Get(AI_MATKEY_CLEARCOAT_FACTOR, o3d_mat.base_clearcoat);
        mat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR,
                 o3d_mat.base_clearcoat_roughness);
        mat->Get(AI_MATKEY_ANISOTROPY_FACTOR, o3d_mat.base_anisotropy);

        // Emissive color scaled by optional emissive intensity factor
        {
            aiColor3D e(0.f, 0.f, 0.f);
            mat->Get(AI_MATKEY_COLOR_EMISSIVE, e);
            float intensity = 1.f;
            mat->Get(AI_MATKEY_EMISSIVE_INTENSITY, intensity);
            o3d_mat.emissive_color = Eigen::Vector4f(
                    e.r * intensity, e.g * intensity, e.b * intensity, 1.f);
        }

        // Transmission / volume (glTF KHR_materials_transmission/volume)
        bool has_transmission =
                (mat->Get(AI_MATKEY_TRANSMISSION_FACTOR,
                          o3d_mat.transmission) == AI_SUCCESS) &&
                o3d_mat.transmission > 0.f;
        if (!has_transmission) {
            o3d_mat.transmission = 0.f;
        }
        mat->Get(AI_MATKEY_VOLUME_THICKNESS_FACTOR, o3d_mat.thickness);
        {
            aiColor3D att(1.f, 1.f, 1.f);
            mat->Get(AI_MATKEY_VOLUME_ATTENUATION_COLOR, att);
            o3d_mat.absorption_color = Eigen::Vector3f(att.r, att.g, att.b);
        }
        mat->Get(AI_MATKEY_VOLUME_ATTENUATION_DISTANCE,
                 o3d_mat.absorption_distance);

        // Alpha/transparency: glTF alpha mode takes priority; for non-glTF
        // formats (OBJ, FBX) fall back to base_color alpha or
        // AI_MATKEY_OPACITY. KHR_materials_transmission is approximated via
        // alpha blending: base_color.w = 1 - transmission_factor. This avoids
        // screen-space refraction (defaultLitSSR) which produces ghosting and
        // is hidden in offscreen render_to_image calls.
        aiString alpha_mode;
        mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alpha_mode);
        std::string alpha_mode_str(alpha_mode.C_Str());
        if (alpha_mode_str == "BLEND" || alpha_mode_str == "MASK") {
            o3d_mat.has_alpha = true;
        } else if (o3d_mat.base_color.w() < 1.f) {
            o3d_mat.has_alpha = true;
        } else if (has_transmission) {
            // Approximate transmission as alpha blending: fully transmissive
            // (transmission=1) maps to fully transparent (alpha=0).
            o3d_mat.base_color.w() = 1.f - o3d_mat.transmission;
            o3d_mat.has_alpha = true;
        } else {
            float opacity = 1.f;
            mat->Get(AI_MATKEY_OPACITY, opacity);
            if (opacity < 1.f) {
                o3d_mat.base_color.w() = opacity;
                o3d_mat.has_alpha = true;
            }
        }

        // Retrieve textures
        TextureImages maps;
        LoadTextures(filename, scene, mat, maps);
        o3d_mat.albedo_img = maps.albedo;
        o3d_mat.normal_img = maps.normal;
        o3d_mat.ao_img = maps.ao;
        o3d_mat.metallic_img = maps.metallic;
        o3d_mat.roughness_img = maps.roughness;
        o3d_mat.reflectance_img = maps.reflectance;
        o3d_mat.ao_rough_metal_img = maps.gltf_rough_metal;
        o3d_mat.clearcoat_img = maps.clearcoat;
        o3d_mat.clearcoat_roughness_img = maps.clearcoat_roughness;
        o3d_mat.anisotropy_img = maps.anisotropy;

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
