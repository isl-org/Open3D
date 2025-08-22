// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <assimp/GltfMaterial.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/ProgressHandler.hpp>
#include <unordered_map>
#include <vector>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/TensorFunction.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "open3d/utility/Logging.h"

#define AI_MATKEY_CLEARCOAT_THICKNESS "$mat.clearcoatthickness", 0, 0
#define AI_MATKEY_CLEARCOAT_ROUGHNESS "$mat.clearcoatroughness", 0, 0
#define AI_MATKEY_SHEEN "$mat.sheen", 0, 0
#define AI_MATKEY_ANISOTROPY "$mat.anisotropy", 0, 0

namespace open3d {
namespace t {
namespace io {

// Split all polygons with more than 3 edges into triangles.
// Ref:
// https://github.com/assimp/assimp/blob/master/include/assimp/postprocess.h
const unsigned int kPostProcessFlags_compulsory =
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
        aiProcess_SortByPType | aiProcess_PreTransformVertices;

const unsigned int kPostProcessFlags_fast =
        kPostProcessFlags_compulsory | aiProcess_GenNormals |
        aiProcess_GenUVCoords | aiProcess_RemoveRedundantMaterials |
        aiProcess_OptimizeMeshes;

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
        utility::LogWarning("Unable to load file {} with ASSIMP: {}", filename,
                            importer.GetErrorString());
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

namespace {
// Add hash function for tuple key
struct TupleHash {
    size_t operator()(const std::tuple<int64_t, float, float>& t) const {
        auto h1 = std::hash<int64_t>{}(std::get<0>(t));
        auto h2 = std::hash<float>{}(std::get<1>(t));
        auto h3 = std::hash<float>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

/// @brief Given a triangle mesh with per triangle UV coordinates, convert to
/// per vertex UVs. This is useful for meshes that have non-unique UVs per
/// vertex, which can happen when the same vertex is used in multiple triangles
/// with different UVs. The function will create a new mesh with unique vertex
/// UVs and update the triangle UVs accordingly.
/// @param mesh Input triangle mesh with per triangle UVs.
/// @param update_triangle_uvs Should the triangle UVs be updated to match the
/// new vertex UVs. Else triangle_uvs will be dropped
/// @return Updated triangle mesh with unique vertex UVs.
geometry::TriangleMesh MakeVertexUVsUnique(const geometry::TriangleMesh& mesh,
                                           bool update_triangle_uvs) {
    if (!mesh.HasTriangleAttr("texture_uvs")) {
        return mesh;
    }
    auto vertices = mesh.GetVertexPositions().Contiguous();
    auto indices = mesh.GetTriangleIndices().Contiguous();
    auto triangle_uvs =
            mesh.GetTriangleAttr("texture_uvs").To(core::Float32).Contiguous();

    bool has_normals = mesh.HasVertexNormals();
    bool has_colors = mesh.HasVertexColors();
    core::Tensor normals, colors;
    if (has_normals) {
        normals = mesh.GetVertexNormals().Contiguous();
    }
    if (has_colors) {
        colors = mesh.GetVertexColors().Contiguous();
    }
    geometry::TriangleMesh new_mesh;
    core::Tensor new_vertices, new_faces, new_normals, new_colors, vertex_uvs;
    bool need_updates = true;

    DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(indices.GetDtype(), int, [&]() {
        scalar_int_t next_vertex_idx = 0;
        // Map to track unique (vertex_idx, uv) combinations
        std::unordered_map<std::tuple<scalar_int_t, float, float>, scalar_int_t,
                           TupleHash>
                vertex_uv_to_new_idx;
        // First pass: collect all unique vertex-UV combinations
        auto p_indices = indices.GetDataPtr<scalar_int_t>();
        auto p_uvs = triangle_uvs.GetDataPtr<float>();
        for (int64_t i = 0; i < indices.GetShape(0); i++) {
            for (int j = 0; j < 3; j++) {
                auto orig_vertex_idx = *p_indices++;
                float u = *p_uvs++;
                float v = *p_uvs++;
                auto key = std::make_tuple(orig_vertex_idx, u, v);
                if (vertex_uv_to_new_idx.find(key) ==
                    vertex_uv_to_new_idx.end()) {
                    vertex_uv_to_new_idx[key] = next_vertex_idx++;
                }
            }
        }
        // Create new tensors with the correct size
        int64_t num_new_vertices = next_vertex_idx;
        if (num_new_vertices == vertices.GetShape(0)) {
            need_updates = false;
            return;  // No duplicate UVs found return the original mesh.
        }
        new_vertices =
                core::Tensor::Empty({num_new_vertices, 3}, vertices.GetDtype());

        if (has_normals) {
            new_normals = core::Tensor::Empty({num_new_vertices, 3},
                                              normals.GetDtype());
        }
        if (has_colors) {
            int color_dims = colors.GetShape(1);
            new_colors = core::Tensor::Empty({num_new_vertices, color_dims},
                                             colors.GetDtype());
        }
        vertex_uvs = core::Tensor::Empty({num_new_vertices, 2}, core::Float32);

        // Fill the new vertex data
        for (const auto& entry : vertex_uv_to_new_idx) {
            auto [orig_vertex_idx, u, v] = entry.first;
            auto new_vertex_idx = entry.second;
            // Copy vertex position
            new_vertices[new_vertex_idx] = vertices[orig_vertex_idx];
            if (has_normals) {  // Copy vertex normal if available
                new_normals[new_vertex_idx] = normals[orig_vertex_idx];
            }
            if (has_colors) {  // Copy vertex color if available
                new_colors[new_vertex_idx] = colors[orig_vertex_idx];
            }
            // Store UV coordinates
            vertex_uvs[new_vertex_idx][0] = u;
            vertex_uvs[new_vertex_idx][1] = v;
        }

        // Second pass: build face indices with new vertex indices
        new_faces = core::Tensor::Empty({indices.GetShape(0), 3},
                                        indices.GetDtype());
        auto faces_ptr = indices.GetDataPtr<scalar_int_t>();
        auto new_faces_ptr = new_faces.GetDataPtr<scalar_int_t>();  // {F, 3}
        auto triangle_uvs_ptr = triangle_uvs.GetDataPtr<float>();   // {F, 3, 2}
        for (int64_t i = 0; i < indices.GetShape(0); i++) {
            for (int j = 0; j < 3; j++) {
                auto idx = *faces_ptr++;
                auto u = *triangle_uvs_ptr++;
                auto v = *triangle_uvs_ptr++;
                *new_faces_ptr++ = vertex_uv_to_new_idx[{idx, u, v}];
            }
        }
    });
    if (!need_updates) {
        return mesh;
    }

    new_mesh.SetVertexPositions(new_vertices);
    new_mesh.SetTriangleIndices(new_faces);
    if (has_normals) {
        new_mesh.SetVertexNormals(new_normals);
    }
    if (has_colors) {
        new_mesh.SetVertexColors(new_colors);
    }
    new_mesh.SetVertexAttr("texture_uvs", vertex_uvs);

    // Convert vertex UVs back to triangle UVs for the new mesh. Not used by
    // ASSIMP.
    if (update_triangle_uvs) {
        auto new_triangle_uvs =
                core::Tensor::Empty({indices.GetShape(0), 3, 2}, core::Float32);
        DISPATCH_INT_DTYPE_PREFIX_TO_TEMPLATE(indices.GetDtype(), int, [&]() {
            scalar_int_t vertex_idx;
            auto new_faces_ptr = new_faces.GetDataPtr<scalar_int_t>();
            auto new_triangle_uvs_ptr = new_triangle_uvs.GetDataPtr<float>();
            auto vertex_uvs_ptr = vertex_uvs.GetDataPtr<float>();
            for (int64_t i = 0; i < indices.GetShape(0); i++) {
                for (int j = 0; j < 3; j++) {
                    vertex_idx = *new_faces_ptr++;
                    *new_triangle_uvs_ptr++ = vertex_uvs_ptr[2 * vertex_idx];
                    *new_triangle_uvs_ptr++ =
                            vertex_uvs_ptr[2 * vertex_idx + 1];
                }
            }
        });
        new_mesh.SetTriangleAttr("texture_uvs", new_triangle_uvs);
    }
    // Copy material if present
    if (mesh.HasMaterial()) {
        new_mesh.SetMaterial(mesh.GetMaterial());
    }

    return new_mesh;
}
}  // namespace

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
                "TriangleMesh can't be saved in ASCII format as .glb");
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
                "vertex colors or export to a format that supports it.");
    }

    geometry::TriangleMesh w_mesh = mesh;  // writeable mesh copy
    if (write_triangle_uvs && mesh.HasTriangleAttr("texture_uvs")) {
        if (!write_vertex_normals && w_mesh.HasVertexNormals()) {
            w_mesh.RemoveVertexAttr("normals");
        }
        if (!write_vertex_colors && w_mesh.HasVertexColors()) {
            w_mesh.RemoveVertexAttr("colors");
        }
        w_mesh = MakeVertexUVsUnique(w_mesh, /*update_triangle_uvs=*/false);
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
    auto vertices = w_mesh.GetVertexPositions().Contiguous();
    auto indices =
            w_mesh.GetTriangleIndices().To(core::Dtype::UInt32).Contiguous();
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

    if (write_vertex_normals && w_mesh.HasVertexNormals()) {
        auto normals = w_mesh.GetVertexNormals().Contiguous();
        auto m_normals = normals.GetShape(0);
        ai_mesh->mNormals = new aiVector3D[m_normals];
        memcpy(&ai_mesh->mNormals->x, normals.GetDataPtr(),
               sizeof(float) * m_normals * 3);
    }

    if (write_vertex_colors && w_mesh.HasVertexColors()) {
        auto colors = w_mesh.GetVertexColors().Contiguous();
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

    if (write_triangle_uvs &&
        w_mesh.HasVertexAttr(
                "texture_uvs")) {  // Save vertex UVs converted earlier
        auto vertex_uvs = w_mesh.GetVertexAttr("texture_uvs").Contiguous();
        auto uv_ptr = vertex_uvs.GetDataPtr<float>();
        auto n_uvs = vertex_uvs.GetShape(0);
        ai_mesh->mNumUVComponents[0] = 2;
        ai_mesh->mTextureCoords[0] = new aiVector3D[n_uvs];
        for (unsigned int i = 0; i < n_uvs; ++i) {
            ai_mesh->mTextureCoords[0][i].x = *uv_ptr++;
            ai_mesh->mTextureCoords[0][i].y = *uv_ptr++;
        }
    } else if (write_triangle_uvs && w_mesh.HasTriangleAttr("texture_uvs")) {
        auto triangle_uvs = w_mesh.GetTriangleAttr("texture_uvs").Contiguous();
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
    if (w_mesh.HasMaterial()) {
        ai_mat->GetName().Set("mat1");
        auto shading_mode = aiShadingMode_PBR_BRDF;
        ai_mat->AddProperty(&shading_mode, 1, AI_MATKEY_SHADING_MODEL);

        // Set base material properties
        // NOTE: not all properties supported by Open3D are supported by Assimp.
        // Those properties (reflectivity, anisotropy) are not exported
        if (w_mesh.GetMaterial().HasBaseColor()) {
            auto c = w_mesh.GetMaterial().GetBaseColor();
            auto ac = aiColor4D(c.x(), c.y(), c.z(), c.w());
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_COLOR_DIFFUSE);
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_BASE_COLOR);
        }
        if (w_mesh.GetMaterial().HasBaseRoughness()) {
            auto r = w_mesh.GetMaterial().GetBaseRoughness();
            ai_mat->AddProperty(&r, 1, AI_MATKEY_ROUGHNESS_FACTOR);
        }
        if (w_mesh.GetMaterial().HasBaseMetallic()) {
            auto m = w_mesh.GetMaterial().GetBaseMetallic();
            ai_mat->AddProperty(&m, 1, AI_MATKEY_METALLIC_FACTOR);
        }
        if (w_mesh.GetMaterial().HasBaseClearcoat()) {
            auto c = w_mesh.GetMaterial().GetBaseClearcoat();
            ai_mat->AddProperty(&c, 1, AI_MATKEY_CLEARCOAT_FACTOR);
        }
        if (w_mesh.GetMaterial().HasBaseClearcoatRoughness()) {
            auto r = w_mesh.GetMaterial().GetBaseClearcoatRoughness();
            ai_mat->AddProperty(&r, 1, AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR);
        }
        if (w_mesh.GetMaterial().HasEmissiveColor()) {
            auto c = w_mesh.GetMaterial().GetEmissiveColor();
            auto ac = aiColor4D(c.x(), c.y(), c.z(), c.w());
            ai_mat->AddProperty(&ac, 1, AI_MATKEY_COLOR_EMISSIVE);
        }

        // Count texture maps...
        // NOTE: GLTF2 expects a single combined roughness/metal map. If the
        // model has one we just export it, otherwise if both roughness and
        // metal maps are available we combine them, otherwise if only one or
        // the other is available we just export the one map.
        int n_textures = 0;
        if (w_mesh.GetMaterial().HasAlbedoMap()) ++n_textures;
        if (w_mesh.GetMaterial().HasNormalMap()) ++n_textures;
        if (w_mesh.GetMaterial().HasAORoughnessMetalMap()) {
            ++n_textures;
        } else if (w_mesh.GetMaterial().HasRoughnessMap() &&
                   w_mesh.GetMaterial().HasMetallicMap()) {
            ++n_textures;
        } else {
            if (w_mesh.GetMaterial().HasRoughnessMap()) ++n_textures;
            if (w_mesh.GetMaterial().HasMetallicMap()) ++n_textures;
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
        if (w_mesh.GetMaterial().HasAlbedoMap()) {
            auto img = w_mesh.GetMaterial().GetAlbedoMap();
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_DIFFUSE, img);
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_BASE_COLOR, img);
            ++current_idx;
        }
        if (w_mesh.GetMaterial().HasAORoughnessMetalMap()) {
            auto img = w_mesh.GetMaterial().GetAORoughnessMetalMap();
            SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                       aiTextureType_UNKNOWN, img);
            ++current_idx;
        } else if (w_mesh.GetMaterial().HasRoughnessMap() &&
                   w_mesh.GetMaterial().HasMetallicMap()) {
            auto rough = w_mesh.GetMaterial().GetRoughnessMap();
            auto metal = w_mesh.GetMaterial().GetMetallicMap();
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
            if (w_mesh.GetMaterial().HasRoughnessMap()) {
                auto img = w_mesh.GetMaterial().GetRoughnessMap();
                SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                           aiTextureType_UNKNOWN, img);
                ++current_idx;
            }
            if (w_mesh.GetMaterial().HasMetallicMap()) {
                auto img = w_mesh.GetMaterial().GetMetallicMap();
                SetTextureMaterialProperty(ai_mat, ai_scene.get(), current_idx,
                                           aiTextureType_UNKNOWN, img);
                ++current_idx;
            }
        }
        if (w_mesh.GetMaterial().HasNormalMap()) {
            auto img = w_mesh.GetMaterial().GetNormalMap();
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
        utility::LogWarning(
                "Got error: ({}) while writing TriangleMesh to file {}.",
                exporter.GetErrorString(), filename);
        return false;
    }

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
