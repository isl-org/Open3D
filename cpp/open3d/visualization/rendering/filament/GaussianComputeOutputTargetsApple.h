// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

#if defined(__APPLE__)

#include "open3d/visualization/rendering/filament/GaussianComputeRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;
class FilamentView;

/// Creates shared MTLTextures and Filament imports (zero-copy path on Metal).
bool PrepareGaussianImportedRenderTargetsApple(
        FilamentView& view,
        FilamentResourceManager& resource_mgr,
        std::uint32_t width,
        std::uint32_t height,
        GaussianComputeRenderer::OutputTargets& targets);

void ReleaseGaussianImportedMTLTexturesApple(
        GaussianComputeRenderer::OutputTargets& targets);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
