// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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
//   - Uses VMA for general buffer/image allocation.  Shared cross-API images
//     (color, depth) are registered via RegisterSharedImageInVulkanContext()
//     rather than allocated here; VulkanInteropContext owns those.
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

/// Register a GL-Vulkan shared image with a Vulkan compute context so that
/// subsequent BindImage() / BindSamplerTexture() calls for that GL texture
/// name resolve to the backing VkImage.
///
/// @param ctx               Created by CreateComputeGpuContextVulkan().
/// @param gl_name           GL texture name from SharedImageDesc.gl_texture.
/// @param vk_image_opaque   SharedImageDesc.vk_image cast to uintptr_t.
/// @param vk_format_opaque  VkFormat value cast to uint32_t.
/// @param width, height     Image dimensions in pixels.
void RegisterSharedImageInVulkanContext(GaussianSplatGpuContext& ctx,
                                        std::uint32_t gl_name,
                                        std::uintptr_t vk_image_opaque,
                                        std::uint32_t vk_format_opaque,
                                        std::uint32_t width,
                                        std::uint32_t height);

/// Unregister a previously registered shared image (called on resize / teardown
/// before VulkanInteropContext::DestroySharedImage).
void UnregisterSharedImageFromVulkanContext(GaussianSplatGpuContext& ctx,
                                            std::uint32_t gl_name);

/// Factory: create a Vulkan-backed GaussianSplatGpuContext.
/// Uses device / queue from GaussianSplatVulkanInteropContext::GetInstance().
/// Returns nullptr if the interop context is not initialized.
[[nodiscard]] std::unique_ptr<GaussianSplatGpuContext>
CreateComputeGpuContextVulkan();

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
