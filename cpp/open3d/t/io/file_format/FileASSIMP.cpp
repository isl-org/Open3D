// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <assimp/GltfMaterial.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <assimp/ProgressHandler.hpp>
#include <fstream>
#include <numeric>
#include <vector>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/TensorFunction.h"
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

}  // namespace io
}  // namespace t
}  // namespace open3d
