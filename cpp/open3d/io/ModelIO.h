// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <functional>
#include <string>

namespace open3d {
namespace geometry {
class TriangleMesh;
}  // namespace geometry
namespace visualization {
namespace rendering {
struct TriangleMeshModel;
}
}  // namespace visualization

namespace io {

namespace detail {

/**
 * Creates a mesh with a texture coordinate per vertex for IO
 * @note TriangleMesh's with adjacency lists are not supported by this function
 * and the output mesh will not have any adjacency information transferred
 * @param mesh The mesh with texture coordinaets to be converted
 * @return A pair containing a new mesh, and a vector texture coordinates
 * of the same length as the vector of vertices. The per face texture
 * in the returned mesh have also been updated.
 */
std::pair<geometry::TriangleMesh, std::vector<Eigen::Vector2d>>
MeshWithPerVertexUVs(const geometry::TriangleMesh& mesh);

}  // namespace detail

struct ReadTriangleModelOptions {
    /// Print progress to stdout about loading progress.
    /// Also see \p update_progress if you want to have your own progress
    /// indicators or to be able to cancel loading.
    bool print_progress = false;
    /// Callback to invoke as reading is progressing, parameter is percentage
    /// completion (0.-100.) return true indicates to continue loading, false
    /// means to try to stop loading and cleanup
    std::function<bool(double)> update_progress;
};

bool ReadTriangleModel(const std::string& filename,
                       visualization::rendering::TriangleMeshModel& model,
                       ReadTriangleModelOptions params = {});

bool WriteTriangleModel(
        const std::string& filename,
        const visualization::rendering::TriangleMeshModel& model);

// Implemented in FileGLTF.cpp
bool WriteTriangleModelToGLTF(
        const std::string& filename,
        const visualization::rendering::TriangleMeshModel& mesh_model);

}  // namespace io
}  // namespace open3d
