// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan compute backend for the Gaussian splatting pipeline.
// Compiled on non-Apple platforms.
//
// Key design:
//   - Implements GaussianSplatGpuContext backed by Vulkan compute pipelines
//     loaded from SPIR-V assets at resources/gaussian_splat/.
//   - Uses VK_KHR_push_descriptor for efficient per-dispatch binding (no
//     descriptor pool allocation per frame).
//   - Uses VMA for internal buffer/image allocation.  Images shared with
//     Filament (color, depth) are owned by GaussianSplatVulkanContext and
//     registered here via RegisterVkImageInComputeContext().
//   - Synchronisation: each EndXxxPass() submits and waits (fence-based) so
//     the rest of the pipeline sees a completed GPU result.

#pragma once

#if !defined(__APPLE__)

#include <cstdint>
#include <memory>

#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"

namespace open3d {
namespace visualization {
namespace rendering {

/// Register a Vulkan image with the compute context so that subsequent
/// BindImage() / BindSamplerTexture() calls resolve to the backing VkImage.
///
/// @param ctx               Created by CreateComputeGpuContextVulkan().
/// @param vk_image_opaque   VkImage handle cast to uintptr_t.
/// @param vk_format_opaque  VkFormat value cast to uint32_t.
/// @param width, height     Image dimensions in pixels.
void RegisterVkImageInComputeContext(GaussianSplatGpuContext& ctx,
                                     std::uintptr_t vk_image_opaque,
                                     std::uint32_t vk_format_opaque,
                                     std::uint32_t width,
                                     std::uint32_t height);

/// Unregister a previously registered image (called on resize/teardown).
void UnregisterVkImageFromComputeContext(GaussianSplatGpuContext& ctx,
                                         std::uintptr_t vk_image_opaque);

/// Factory: create a Vulkan-backed GaussianSplatGpuContext.
/// Uses device / queue from GaussianSplatVulkanContext::GetInstance().
/// Returns nullptr if the context is not initialized.
[[nodiscard]] std::unique_ptr<GaussianSplatGpuContext>
CreateComputeGpuContextVulkan();

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
