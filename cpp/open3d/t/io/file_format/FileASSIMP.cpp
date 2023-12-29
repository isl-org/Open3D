// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <assimp/GltfMaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/ProgressHandler.hpp>
#include <fstream>
#include <numeric>
#include <vector>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

#define AI_MATKEY_CLEARCOAT_THICKNESS "$mat.clearcoatthickness", 0, 0
#define AI_MATKEY_CLEARCOAT_ROUGHNESS "$mat.clearcoatroughness", 0, 0
#define AI_MATKEY_SHEEN "$mat.sheen", 0, 0
#define AI_MATKEY_ANISOTROPY "$mat.anisotropy", 0, 0

namespace open3d {
namespace t {
namespace io {

// Split all polygons with more than 3 edges into triangles.
const unsigned int kPostProcessFlags_compulsory =
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
        aiProcess_SortByPType;

const unsigned int kPostProcessFlags_fast =
        aiProcessPreset_TargetRealtime_Fast |
        aiProcess_RemoveRedundantMaterials | aiProcess_OptimizeMeshes |
        aiProcess_PreTransformVertices;

bool ReadTriangleMeshUsingASSIMP(
        const std::string& filename,
        geometry::TriangleMesh& mesh,
        const open3d::io::ReadTriangleMeshOptions& params /*={}*/) {
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

    std::vector<core::Tensor> mesh_vertices;
    std::vector<core::Tensor> mesh_vertex_normals;
    std::vector<core::Tensor> mesh_faces;
    std::vector<core::Tensor> mesh_vertex_colors;
    std::vector<core::Tensor> mesh_uvs;

    size_t current_vidx = 0;
    size_t count_mesh_with_normals = 0;
    size_t count_mesh_with_colors = 0;
    size_t count_mesh_with_uvs = 0;

    // Merge individual meshes in aiScene into a single TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];

        core::Tensor vertices = core::Tensor::Empty(
                {assimp_mesh->mNumVertices, 3}, core::Dtype::Float32);
        auto vertices_ptr = vertices.GetDataPtr<float>();
        std::memcpy(vertices_ptr, assimp_mesh->mVertices,
                    3 * assimp_mesh->mNumVertices * sizeof(float));
        mesh_vertices.push_back(vertices);

        core::Tensor vertex_normals;
        core::Tensor vertex_colors;
        core::Tensor triangle_uvs;
        if (assimp_mesh->mNormals) {
            // Loop fusion for performance optimization.
            vertex_normals = core::Tensor::Empty({assimp_mesh->mNumVertices, 3},
                                                 core::Dtype::Float32);
            auto vertex_normals_ptr = vertex_normals.GetDataPtr<float>();
            std::memcpy(vertex_normals_ptr, assimp_mesh->mNormals,
                        3 * assimp_mesh->mNumVertices * sizeof(float));
            mesh_vertex_normals.push_back(vertex_normals);
            count_mesh_with_normals++;
        }

        if (assimp_mesh->HasVertexColors(0)) {
            vertex_colors = core::Tensor::Empty({assimp_mesh->mNumVertices, 3},
                                                core::Dtype::Float32);
            auto vertex_colors_ptr = vertex_colors.GetDataPtr<float>();
            for (unsigned int i = 0; i < assimp_mesh->mNumVertices; ++i) {
                *vertex_colors_ptr++ = assimp_mesh->mColors[0][i].r;
                *vertex_colors_ptr++ = assimp_mesh->mColors[0][i].g;
                *vertex_colors_ptr++ = assimp_mesh->mColors[0][i].b;
            }
            mesh_vertex_colors.push_back(vertex_colors);
            count_mesh_with_colors++;
        }

        core::Tensor faces = core::Tensor::Empty({assimp_mesh->mNumFaces, 3},
                                                 core::Dtype::Int64);
        auto faces_ptr = faces.GetDataPtr<int64_t>();
        core::ParallelFor(
                core::Device("CPU:0"), assimp_mesh->mNumFaces,
                [&](size_t fidx) {
                    const auto& face = assimp_mesh->mFaces[fidx];
                    faces_ptr[3 * fidx] = face.mIndices[0] + current_vidx;
                    faces_ptr[3 * fidx + 1] = face.mIndices[1] + current_vidx;
                    faces_ptr[3 * fidx + 2] = face.mIndices[2] + current_vidx;
                });

        mesh_faces.push_back(faces);

        if (assimp_mesh->HasTextureCoords(0)) {
            auto vertex_uvs = core::Tensor::Empty(
                    {assimp_mesh->mNumVertices, 2}, core::Dtype::Float32);
            auto uvs_ptr = vertex_uvs.GetDataPtr<float>();
            // NOTE: Can't just memcpy because ASSIMP UVs are 3 element and
            // TriangleMesh wants 2 element UVs.
            for (int i = 0; i < (int)assimp_mesh->mNumVertices; ++i) {
                *uvs_ptr++ = assimp_mesh->mTextureCoords[0][i].x;
                *uvs_ptr++ = assimp_mesh->mTextureCoords[0][i].y;
            }
            triangle_uvs = vertex_uvs.IndexGet({faces});
            mesh_uvs.push_back(triangle_uvs);
            count_mesh_with_uvs++;
        }
        // Adjust face indices to index into combined mesh vertex array
        current_vidx += static_cast<int>(assimp_mesh->mNumVertices);
    }

    mesh.Clear();
    if (scene->mNumMeshes > 1) {
        mesh.SetVertexPositions(core::Concatenate(mesh_vertices));
        mesh.SetTriangleIndices(core::Concatenate(mesh_faces));
        // NOTE: For objects with multiple meshes we only store normals, colors,
        // and uvs if every mesh in the object had them. Mesh class does not
        // support some vertices having normals/colors/uvs and some not having
        // them.
        if (count_mesh_with_normals == scene->mNumMeshes) {
            mesh.SetVertexNormals(core::Concatenate(mesh_vertex_normals));
        }
        if (count_mesh_with_colors == scene->mNumMeshes) {
            mesh.SetVertexColors(core::Concatenate(mesh_vertex_colors));
        }
        if (count_mesh_with_uvs == scene->mNumMeshes) {
            mesh.SetTriangleAttr("texture_uvs", core::Concatenate(mesh_uvs));
        }
    } else {
        mesh.SetVertexPositions(mesh_vertices[0]);
        mesh.SetTriangleIndices(mesh_faces[0]);
        if (count_mesh_with_normals > 0) {
            mesh.SetVertexNormals(mesh_vertex_normals[0]);
        }
        if (count_mesh_with_colors > 0) {
            mesh.SetVertexColors(mesh_vertex_colors[0]);
        }
        if (count_mesh_with_uvs > 0) {
            mesh.SetTriangleAttr("texture_uvs", mesh_uvs[0]);
        }
    }

    return true;
}

static void SetTextureMaterialProperty(aiMaterial* mat,
                                       aiScene* scene,
                                       int texture_idx,
                                       aiTextureType tt,
                                       t::geometry::Image& img) {
    // Encode image as PNG
    std::vector<uint8_t> img_buffer;
    WriteImageToPNGInMemory(img_buffer, img, 6);

    // Fill in Assimp's texture class and add to its material
    auto tex = scene->mTextures[texture_idx];
    std::string tex_id("*");
    tex_id += std::to_string(texture_idx);
    tex->mFilename = tex_id.c_str();
    tex->mHeight = 0;
    tex->mWidth = img_buffer.size();
    // NOTE: Assimp takes ownership of the data so we need to copy it
    // into a separate buffer that Assimp can take care of delete []-ing
    uint8_t* img_data = new uint8_t[img_buffer.size()];
    memcpy(img_data, img_buffer.data(), img_buffer.size());
    tex->pcData = reinterpret_cast<aiTexel*>(img_data);
    strcpy(tex->achFormatHint, "png");
    aiString uri(tex_id);
    const int uv_index = 0;
    const aiTextureMapMode mode = aiTextureMapMode_Wrap;
    mat->AddProperty(&uri, AI_MATKEY_TEXTURE(tt, 0));
    mat->AddProperty(&uv_index, 1, AI_MATKEY_UVWSRC(tt, 0));
    mat->AddProperty(&mode, 1, AI_MATKEY_MAPPINGMODE_U(tt, 0));
    mat->AddProperty(&mode, 1, AI_MATKEY_MAPPINGMODE_V(tt, 0));
}

bool WriteTriangleMeshUsingASSIMP(const std::string& filename,
                                  const geometry::TriangleMesh& mesh,
                                  const bool write_ascii,
                                  const bool compressed,
                                  const bool write_vertex_normals,
                                  const bool write_vertex_colors,
                                  const bool write_triangle_uvs,
                                  const bool print_progress) {
    // Sanity checks...
    if (write_ascii) {
        utility::LogWarning(
                "TriangleMesh can't be saved in ASCII fromat as .glb");
        return false;
    }
    if (compressed) {
        utility::LogWarning(
                "TriangleMesh can't be saved in compressed format as .glb");
        return false;
    }
    if (!mesh.HasVertexPositions()) {
        utility::LogWarning(
                "TriangleMesh has no vertex positions and can't be saved as "
                ".glb");
        return false;
    }
    // Check for unsupported features
    if (mesh.HasTriangleNormals()) {
        utility::LogWarning(
                "Exporting triangle normals is not supported. Please convert "
                "to vertex normals or export to a format that supports it.");
    }
    if (mesh.HasTriangleColors()) {
        utility::LogWarning(
                "Exporting triangle colors is not supported. Please convert to "
                "vertex colors or export to a format that supporst it.");
    }

    Assimp::Exporter exporter;
    auto ai_scene = std::unique_ptr<aiScene>(new aiScene);

    // Fill mesh data...
    ai_scene->mNumMeshes = 1;
    ai_scene->mMeshes = new aiMesh*[1];
    auto ai_mesh = new aiMesh;
    ai_mesh->mName.Set("Object1");
    ai_mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
    // Guaranteed to have both vertex positions and triangle indices
    auto vertices = mesh.GetVertexPositions().Contiguous();
    auto indices =
            mesh.GetTriangleIndices().To(core::Dtype::UInt32).Contiguous();
    ai_mesh->mNumVertices = vertices.GetShape(0);
    ai_mesh->mVertices = new aiVector3D[ai_mesh->mNumVertices];
    memcpy(&ai_mesh->mVertices->x, vertices.GetDataPtr(),
           sizeof(float) * ai_mesh->mNumVertices * 3);
    ai_mesh->mNumFaces = indices.GetShape(0);
    ai_mesh->mFaces = new aiFace[ai_mesh->mNumFaces];
    for (unsigned int i = 0; i < ai_mesh->mNumFaces; ++i) {
        ai_mesh->mFaces[i].mNumIndices = 3;
        // NOTE: Yes, dynamically allocating 3 ints for each face is inefficient
        // but this is what Assimp seems to require as it deletes each mIndices
        // on destruction. We could block allocate space for all the faces,
        // assign pointers here then zero out the pointers before destruction so
        // the delete becomes a no-op, but that seems error prone. Could revisit
        // if this becomes an IO bottleneck.
        ai_mesh->mFaces[i].mIndices = new unsigned int[3];  // triangles
        ai_mesh->mFaces[i].mIndices[0] = indices[i][0].Item<unsigned int>();
        ai_mesh->mFaces[i].mIndices[1] = indices[i][1].Item<unsigned int>();
        ai_mesh->mFaces[i].mIndices[2] = indices[i][2].Item<unsigned int>();
    }

    if (write_vertex_normals && mesh.HasVertexNormals()) {
        auto normals = mesh.GetVertexNormals().Contiguous();
        auto m_normals = normals.GetShape(0);
        ai_mesh->mNormals = new aiVector3D[m_normals];
        memcpy(&ai_mesh->mNormals->x, normals.GetDataPtr(),
               sizeof(float) * m_normals * 3);
    }

    if (write_vertex_colors && mesh.HasVertexColors()) {
        auto colors = mesh.GetVertexColors().Contiguous();
        auto m_colors = colors.GetShape(0);
        ai_mesh->mColors[0] = new aiColor4D[m_colors];
        if (colors.GetShape(1) == 4) {
            memcpy(&ai_mesh->mColors[0][0].r, colors.GetDataPtr(),
                   sizeof(float) * m_colors * 4);
        } else {  // must be 3 components
            auto color_ptr = reinterpret_cast<float*>(colors.GetDataPtr());
            for (unsigned int i = 0; i < m_colors; ++i) {
                ai_mesh->mColors[0][i].r = *color_ptr++;
                ai_mesh->mColors[0][i].g = *color_ptr++;
                ai_mesh->mColors[0][i].b = *color_ptr++;
                ai_mesh->mColors[0][i].a = 1.0f;
            }
        }
    }

    if (write_triangle_uvs && mesh.HasTriangleAttr("texture_uvs")) {
        auto triangle_uvs = mesh.GetTriangleAttr("texture_uvs").Contiguous();
        auto vertex_uvs = core::Tensor::Empty({ai_mesh->mNumVertices, 2},
                                              core::Dtype::Float32);
        auto n_uvs = ai_mesh->mNumVertices;
        for (int64_t i = 0; i < indices.GetShape(0); i++) {
            vertex_uvs[indices[i][0].Item<uint32_t>()] = triangle_uvs[i][0];
            vertex_uvs[indices[i][1].Item<uint32_t>()] = triangle_uvs[i][1];
            vertex_uvs[indices[i][2].Item<uint32_t>()] = triangle_uvs[i][2];
        }
        ai_mesh->mTextureCoords[0] = new aiVector3D[n_uvs];
        auto uv_ptr = reinterpret_cast<float*>(vertex_uvs.GetDataPtr());
        for (unsigned int i = 0; i < n_uvs; ++i) {
            ai_mesh->mTextureCoords[0][i].x = *uv_ptr++;
            ai_mesh->mTextureCoords[0][i].y = *uv_ptr++;
        }
        ai_mesh->mNumUVComponents[0] = 2;
    }
    ai_scene->mMeshes[0] = ai_mesh;

    // Fill material data...
    ai_scene->mNumMaterials = 1;
    ai_scene->mMaterials = new aiMaterial*[ai_scene->mNumMaterials];
    auto ai_mat = new aiMaterial;
    if (mesh.HasMaterial()) {
        ai_mat->GetName().Set("mat1");
        auto shading_mode = aiShadingMode_PBR_BRDF;
        ai_mat->AddProperty(&shading_mode, 1, AI_MATKEY_SHADING_MODEL);

        // Set base material properties
        // NOTE: not all properties supported by Open3D are supported by Assimp.
        // Those properties (reflectivity, anisotropy) are not exported
        if (mesh.GetMaterial().HasBaseColor()) {
            auto c = mesh.GetMaterial().GetBaseColor();
            auto ac = aiColor4D(c.x(), c.y(), c.z(), c.w());
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_COLOR_DIFFUSE);
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_BASE_COLOR);
        }
        if (mesh.GetMaterial().HasBaseRoughness()) {
            auto r = mesh.GetMaterial().GetBaseRoughness();
            ai_mat->AddProperty(&r, 1, AI_MATKEY_ROUGHNESS_FACTOR);
        }
        if (mesh.GetMaterial().HasBaseMetallic()) {
            auto m = mesh.GetMaterial().GetBaseMetallic();
            ai_mat->AddProperty(&m, 1, AI_MATKEY_METALLIC_FACTOR);
        }
        if (mesh.GetMaterial().HasBaseClearcoat()) {
            auto c = mesh.GetMaterial().GetBaseClearcoat();
            ai_mat->AddProperty(&c, 1, AI_MATKEY_CLEARCOAT_FACTOR);
        }
        if (mesh.GetMaterial().HasBaseClearcoatRoughness()) {
            auto r = mesh.GetMaterial().GetBaseClearcoatRoughness();
            ai_mat->AddProperty(&r, 1, AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR);
        }

        // Count texture maps...
        // NOTE: GLTF2 expects a single combined roughness/metal map. If the
        // model has one we just export it, otherwise if both roughness and
        // metal maps are avaialbe we combine them, otherwise if only one or the
        // other is available we just export the one map.
        int n_textures = 0;
        if (mesh.GetMaterial().HasAlbedoMap()) ++n_textures;
        if (mesh.GetMaterial().HasNormalMap()) ++n_textures;
        if (mesh.GetMaterial().HasAORoughnessMetalMap()) {
            ++n_textures;
        } else if (mesh.GetMaterial().HasRoughnessMap() &&
                   mesh.GetMaterial().HasMetallicMap()) {
            ++n_textures;
        } else {
            if (mesh.GetMaterial().HasRoughnessMap()) ++n_textures;
            if (mesh.GetMaterial().HasMetallicMap()) ++n_textures;
        }
        if (n_textures > 0) {
            ai_scene->mTextures = new aiTexture*[n_textures];
            for (int i = 0; i < n_textures; ++i) {
                ai_scene->mTextures[i] = new aiTexture();
            }
            ai_scene->mNumTextures = n_textures;
        }

        // Now embed the textures that are available...
        int current_idx = 0;
        if (mesh.GetMaterial().HasAlbedoMap()) {
            auto img = mesh.GetMaterial().GetAlbedoMap();
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_DIFFUSE, img);
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_BASE_COLOR, img);
            ++current_idx;
        }
        if (mesh.GetMaterial().HasAORoughnessMetalMap()) {
            auto img = mesh.GetMaterial().GetAORoughnessMetalMap();
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_UNKNOWN, img);
            ++current_idx;
        } else if (mesh.GetMaterial().HasRoughnessMap() &&
                   mesh.GetMaterial().HasMetallicMap()) {
            auto rough = mesh.GetMaterial().GetRoughnessMap();
            auto metal = mesh.GetMaterial().GetMetallicMap();
            auto rows = rough.GetRows();
            auto cols = rough.GetCols();
            auto rough_metal =
                    geometry::Image(rows, cols, 4, core::Dtype::UInt8);
            rough_metal.AsTensor() =
                    core::Tensor::Ones(rough_metal.AsTensor().GetShape(),
                                       core::Dtype::UInt8) *
                    255;
            auto metal_channel = metal.AsTensor().GetItem(
                    {core::TensorKey::Slice(0, rows + 1, core::None),
                     core::TensorKey::Slice(0, cols + 1, core::None),
                     core::TensorKey::Index(0)});
            auto rough_channel = rough.AsTensor().GetItem(
                    {core::TensorKey::Slice(0, rows + 1, core::None),
                     core::TensorKey::Slice(0, cols + 1, core::None),
                     core::TensorKey::Index(0)});
            rough_metal.AsTensor().SetItem(
                    {core::TensorKey::Slice(0, rows + 1, core::None),
                     core::TensorKey::Slice(0, cols + 1, core::None),
                     core::TensorKey::Index(2)},
                    metal_channel);
            rough_metal.AsTensor().SetItem(
                    {core::TensorKey::Slice(0, rows + 1, core::None),
                     core::TensorKey::Slice(0, cols + 1, core::None),
                     core::TensorKey::Index(1)},
                    rough_channel);
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_UNKNOWN, rough_metal);
            ++current_idx;
        } else {
            if (mesh.GetMaterial().HasRoughnessMap()) {
                auto img = mesh.GetMaterial().GetRoughnessMap();
                SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                           aiTextureType_UNKNOWN, img);
                ++current_idx;
            }
            if (mesh.GetMaterial().HasMetallicMap()) {
                auto img = mesh.GetMaterial().GetMetallicMap();
                SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                           aiTextureType_UNKNOWN, img);
                ++current_idx;
            }
        }
        if (mesh.GetMaterial().HasNormalMap()) {
            auto img = mesh.GetMaterial().GetNormalMap();
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_NORMALS, img);
            ++current_idx;
        }
    }
    ai_scene->mMaterials[0] = ai_mat;

    auto root_node = new aiNode;
    root_node->mName.Set("root");
    root_node->mNumMeshes = 1;
    root_node->mMeshes = new unsigned int[root_node->mNumMeshes];
    root_node->mMeshes[0] = 0;
    ai_scene->mRootNode = root_node;

    // Export
    if (exporter.Export(ai_scene.get(), "glb2", filename.c_str()) ==
        AI_FAILURE) {
        utility::LogWarning("Got error: {}", exporter.GetErrorString());
        return false;
    }

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
