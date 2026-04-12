// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

#if defined(__APPLE__)

#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;
class FilamentView;

/// Creates shared MTLTextures and Filament imports (zero-copy path on Metal).
/// @param needs_scene_depth  When false, skip allocating the scene-depth
///                           texture (no mesh occluders in the scene).
bool PrepareGaussianImportedRenderTargetsApple(
        FilamentView& view,
        FilamentResourceManager& resource_mgr,
        std::uint32_t width,
        std::uint32_t height,
        bool needs_scene_depth,
        GaussianSplatRenderer::OutputTargets& targets);

void ReleaseGaussianImportedMTLTexturesApple(
        GaussianSplatRenderer::OutputTargets& targets);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
