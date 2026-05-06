// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan compute backend for GaussianSplatRenderer.
// Implements GaussianSplatRenderer::Backend using ComputeGPUVulkan.
// Only compiled on non-Apple platforms.

#pragma once

#if !defined(__APPLE__)

#include <memory>

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;

/// Create the Vulkan compute Gaussian splat backend.
/// Returns nullptr if GaussianSplatVulkanInteropContext is not initialized.
[[nodiscard]] std::unique_ptr<GaussianSplatRenderer::Backend>
CreateGaussianSplatVulkanBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
