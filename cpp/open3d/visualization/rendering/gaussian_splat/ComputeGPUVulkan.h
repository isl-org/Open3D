// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan compute backend for the Gaussian splatting pipeline.
// Compiled on non-Apple platforms alongside ComputeGPUGL.cpp.
//
// Key design:
//   - Implements GaussianSplatGpuContext backed by Vulkan compute pipelines
//     loaded from the same SPIR-V assets used by the GL backend.
//   - Uses VK_KHR_push_descriptor for efficient per-dispatch binding (no
//     descriptor pool allocation per frame).
//   - Uses VMA for general buffer/image allocation.  Shared cross-API images
//     (color, depth) are registered via RegisterSharedImageInVulkanContext()
//     rather than allocated here; VulkanInteropContext owns those.
//   - Synchronisation: each EndXxxPass() submits and waits (fence-based) so
//     the rest of the pipeline sees a completed GPU result.  Milestone E will
//     replace the fence wait with the explicit GL<->VK semaphore pair created
//     in PrepareOutputTextures.

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
///
/// @param use_subgroups  If true, load the _subgroup SPIR-V variants for
///                       prefix-sum and radix-sort (faster on Vulkan).
[[nodiscard]] std::unique_ptr<GaussianSplatGpuContext>
CreateComputeGpuContextVulkan(bool use_subgroups);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
