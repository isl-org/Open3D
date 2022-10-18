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

    size_t current_vidx = 0;
    size_t count_mesh_with_normals = 0;

    // Merge individual meshes in aiScene into a single TriangleMesh
    for (size_t midx = 0; midx < scene->mNumMeshes; ++midx) {
        const auto* assimp_mesh = scene->mMeshes[midx];

        core::Tensor vertices = core::Tensor::Empty(
                {assimp_mesh->mNumVertices, 3}, core::Dtype::Float32);
        auto vertices_ptr = vertices.GetDataPtr<float>();

        core::Tensor vertex_normals;
        if (assimp_mesh->mNormals) {
            // Loop fusion for performance optimization.
            vertex_normals = core::Tensor::Empty({assimp_mesh->mNumFaces, 3},
                                                 core::Dtype::Float32);
            auto vertex_normals_ptr = vertices.GetDataPtr<float>();
            std::memcpy(vertices_ptr, assimp_mesh->mVertices,
                        3 * assimp_mesh->mNumVertices * sizeof(float));
            std::memcpy(vertex_normals_ptr, assimp_mesh->mNormals,
                        3 * assimp_mesh->mNumVertices * sizeof(float));
            mesh_vertex_normals.push_back(vertex_normals);
            count_mesh_with_normals++;
        } else {
            std::memcpy(vertices_ptr, assimp_mesh->mVertices,
                        3 * assimp_mesh->mNumVertices * sizeof(float));
        }
        mesh_vertices.push_back(vertices);

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

        // Adjust face indices to index into combined mesh vertex array
        current_vidx += static_cast<int>(assimp_mesh->mNumVertices);
    }

    mesh.Clear();
    if (scene->mNumMeshes > 1) {
        mesh.SetVertexPositions(core::Concatenate(mesh_vertices));
        mesh.SetTriangleIndices(core::Concatenate(mesh_faces));
        if (count_mesh_with_normals == scene->mNumMeshes) {
            mesh.SetVertexNormals(core::Concatenate(mesh_vertex_normals));
        }
    } else {
        mesh.SetVertexPositions(mesh_vertices[0]);
        mesh.SetTriangleIndices(mesh_faces[0]);
        if (count_mesh_with_normals) {
            mesh.SetVertexNormals(mesh_vertex_normals[0]);
        }
    }

    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
