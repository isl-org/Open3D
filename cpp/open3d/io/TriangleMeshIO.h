// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/geometry/TriangleMesh.h"

namespace open3d {
namespace io {

/// Factory function to create a mesh from a file (TriangleMeshFactory.cpp)
/// Return an empty mesh if fail to read the file.
std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename, bool print_progress = false);

struct ReadTriangleMeshOptions {
    /// Enables post-processing on the mesh.
    /// Post-processing will
    ///   - triangulate meshes with polygonal faces
    ///   - remove redundant materials
    ///   - pretransform vertices
    ///   - generate face normals if needed
    ///
    /// For more information see ASSIMPs documentation on the flags
    /// `aiProcessPreset_TargetRealtime_Fast,
    /// aiProcess_RemoveRedundantMaterials, aiProcess_OptimizeMeshes,
    /// aiProcess_PreTransformVertices`.
    ///
    /// Note that identical vertices will always be joined regardless of whether
    /// post-processing is enabled or not, which changes the number of vertices
    /// in the mesh.
    ///
    /// The ply-format is not affected by the post-processing.
    bool enable_post_processing = false;
    /// Print progress to stdout about loading progress.
    /// Also see \p update_progress if you want to have your own progress
    /// indicators or to be able to cancel loading.
    bool print_progress = false;
    /// Callback to invoke as reading is progressing, parameter is percentage
    /// completion (0.-100.) return true indicates to continue loading, false
    /// means to try to stop loading and cleanup
    std::function<bool(double)> update_progress;
};

/// The general entrance for reading a TriangleMesh from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      ReadTriangleMeshOptions params = {});

/// The general entrance for writing a TriangleMesh to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// At current only .obj format supports uv coordinates (triangle_uvs) and
/// textures.
/// \return return true if the write function is successful, false otherwise.
bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii = false,
                       bool compressed = false,
                       bool write_vertex_normals = true,
                       bool write_vertex_colors = true,
                       bool write_triangle_uvs = true,
                       bool print_progress = false);

// Currently enable_post_processing not used in ReadTriangleMeshFromPLY.
bool ReadTriangleMeshFromPLY(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             const ReadTriangleMeshOptions &params);

bool WriteTriangleMeshToPLY(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);

bool WriteTriangleMeshToSTL(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);

// Currently enable_post_processing not used in ReadTriangleMeshFromOBJ.
bool ReadTriangleMeshFromOBJ(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             const ReadTriangleMeshOptions &params);

bool WriteTriangleMeshToOBJ(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);

bool ReadTriangleMeshUsingASSIMP(const std::string &filename,
                                 geometry::TriangleMesh &mesh,
                                 const ReadTriangleMeshOptions &params);

// Currently enable_post_processing not used in ReadTriangleMeshFromOFF.
bool ReadTriangleMeshFromOFF(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             const ReadTriangleMeshOptions &params);

bool WriteTriangleMeshToOFF(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);

// Currently enable_post_processing not used in ReadTriangleMeshFromGLTF.
bool ReadTriangleMeshFromGLTF(const std::string &filename,
                              geometry::TriangleMesh &mesh,
                              const ReadTriangleMeshOptions &params);

bool WriteTriangleMeshToGLTF(const std::string &filename,
                             const geometry::TriangleMesh &mesh,
                             bool write_ascii,
                             bool compressed,
                             bool write_vertex_normals,
                             bool write_vertex_colors,
                             bool write_triangle_uvs,
                             bool print_progress);

/// Function to convert a polygon into a collection of
/// triangles whose vertices are only those of the polygon.
/// Assume that the vertices are connected by edges based on their order, and
/// the final vertex connected to the first.
/// The triangles are added to the mesh that is passed as reference. The mesh
/// should contain all vertices prior to calling this function.
/// \return return true if triangulation is successful, false otherwise.
bool AddTrianglesByEarClipping(geometry::TriangleMesh &mesh,
                               std::vector<unsigned int> &indices);

}  // namespace io
}  // namespace open3d
