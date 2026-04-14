// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// OpenGL compute backend for Gaussian splatting (Linux and Windows).
// Extracted from GaussianSplatRenderer.cpp so the GL-specific code has its
// own translation unit and does not pollute the platform-agnostic renderer.

#pragma once

#if !defined(__APPLE__)

#include <memory>

#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;

/// Create the OpenGL Gaussian splat backend.
/// Must be called after GaussianSplatOpenGLContext::InitializeStandalone().
[[nodiscard]] std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatOpenGLBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
