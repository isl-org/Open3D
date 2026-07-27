// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
/// \par Supported read formats:
///   - \c ply  -- native Open3D reader (geometry + colors/normals)
///   - \c npz  -- Open3D NPZ format (full round-trip incl. materials)
///   - \c obj, stl, off, gltf, glb, fbx -- via ASSIMP (geometry; optional
///     vertex normals/colors, UVs, and material/PBR data depending on format,
///     e.g. STL is geometry-only)
/// \param filename Path to the mesh file.
/// \param mesh Output parameter for the mesh.
/// \param params Additional read options to enable post-processing or progress
///              reporting.
/// \return true if the read function is successful, false otherwise.
bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      open3d::io::ReadTriangleMeshOptions params = {});

/// The general entrance for writing a TriangleMesh to a file.
/// The function calls write functions based on the extension name of filename.
/// \par Supported write formats and material/texture export:
///   - \c npz  -- full round-trip (geometry + material + all texture maps)
///   - \c glb  -- via ASSIMP; full PBR single material with embedded textures
///   - \c gltf -- via ASSIMP; full PBR single material with external textures
///   - \c obj  -- via ASSIMP; single material with external PNG sidecar
///   textures
///               (albedo, normal, AO, roughness, metallic)
///   - \c fbx  -- via ASSIMP; geometry + best-effort material (textures not
///               guaranteed by the ASSIMP FBX exporter)
///   - \c stl  -- via ASSIMP; geometry only (positions, faces, normals)
///   - \c ply, off -- via legacy Open3D writer; geometry + colors/normals only
/// \note For ASSIMP writers, \p write_ascii is format-specific: it selects
///       ASCII vs binary STL and is ignored for \c glb / \c gltf (encoding
///       follows the file extension), \c obj, and \c fbx as documented in logs.
/// \note Only a single material per mesh is supported. Multiple materials,
///       triangle_material_ids, per-triangle normals, and animation are not
///       supported.
/// \return true if the write function is successful, false otherwise.
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

bool WriteTriangleMeshUsingASSIMP(const std::string &filename,
                                  const geometry::TriangleMesh &mesh,
                                  const bool write_ascii,
                                  const bool compressed,
                                  const bool write_vertex_normals,
                                  const bool write_vertex_colors,
                                  const bool write_triangle_uvs,
                                  const bool print_progress);

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
