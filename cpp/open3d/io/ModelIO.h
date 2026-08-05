// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>

namespace open3d {
namespace visualization {
namespace rendering {
struct TriangleMeshModel;
}
}  // namespace visualization

namespace io {

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

/// Read a TriangleMeshModel (multi-mesh, multi-material) from a file.
/// Supported formats (via Assimp): fbx, gltf, glb, obj, stl, off,
/// usd, usda, usdc, usdz.
///
/// USD import is experimental: only mesh geometry and PBR material import
/// (base color, normal, metallic, roughness, occlusion textures, and related
/// scalars) are supported. USD export, animations, lights, cameras, and other
/// scene features are not supported.
/// \param filename Path to the model file.
/// \param model Output TriangleMeshModel.
/// \param params Optional progress/cancel options.
/// \return true on success.
bool ReadTriangleModel(const std::string& filename,
                       visualization::rendering::TriangleMeshModel& model,
                       ReadTriangleModelOptions params = {});

}  // namespace io
}  // namespace open3d
