// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/io/TriangleMeshIO.h"
#include "open3d/t/geometry/TriangleMesh.h"

namespace open3d {
namespace t {
namespace io {

/// Factory function to create a mesh from a file (TriangleMeshFactory.cpp)
/// Return an empty mesh if fail to read the file.
std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename, bool print_progress = false);

/// The general entrance for reading a TriangleMesh from a file.
/// The function calls read functions based on the extension name of filename.
/// Supported formats are \c obj,ply,stl,off,gltf,glb,fbx .
/// \param filename Path to the mesh file.
/// \param mesh Output parameter for the mesh.
/// \param params Additional read options to enable post-processing or progress
/// reporting. \return return true if the read function is successful, false
/// otherwise.
bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      open3d::io::ReadTriangleMeshOptions params = {});

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

bool ReadTriangleMeshUsingASSIMP(
        const std::string &filename,
        geometry::TriangleMesh &mesh,
        const open3d::io::ReadTriangleMeshOptions &params);

bool ReadTriangleMeshFromNPZ(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             const open3d::io::ReadTriangleMeshOptions &params);

bool WriteTriangleMeshToNPZ(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            const bool write_ascii,
                            const bool compressed,
                            const bool write_vertex_normals,
                            const bool write_vertex_colors,
                            const bool write_triangle_uvs,
                            const bool print_progress);

}  // namespace io
}  // namespace t
}  // namespace open3d
