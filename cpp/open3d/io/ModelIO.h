// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>

#include <Eigen/Core>

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

std::pair<geometry::TriangleMesh, std::vector<Eigen::Vector2d>>
MeshWithPerVertexUVs(const geometry::TriangleMesh& mesh);

bool WriteValidatedTriangleMeshModelToGLTF(
        const std::string& filename,
        const visualization::rendering::TriangleMeshModel& model);

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

bool WriteTriangleMeshModel(const std::string& filename,
        const visualization::rendering::TriangleMeshModel& model);

bool WriteTriangleMeshModelToGLTF(const std::string& filename,
        const visualization::rendering::TriangleMeshModel& mesh_model);

}  // namespace io
}  // namespace open3d
