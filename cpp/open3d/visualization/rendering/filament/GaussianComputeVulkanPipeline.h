// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Vulkan compute pipeline for dispatching Gaussian splat rendering passes.
// Uses the same Vulkan device that Filament is running on.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace open3d {
namespace visualization {
namespace rendering {

/// Opaque handle to an allocated Vulkan buffer/memory pair.
struct VulkanBufferHandle {
    std::uintptr_t buffer = 0;
    std::uintptr_t memory = 0;
    std::size_t size = 0;
    void* mapped = nullptr;  ///< Non-null when host-visible.
    bool valid = false;
};

/// Opaque handle to a Vulkan image + image view + memory.
struct VulkanImageHandle {
    std::uintptr_t image = 0;
    std::uintptr_t image_view = 0;
    std::uintptr_t memory = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    bool valid = false;
};

/// Opaque handle to a compiled Vulkan compute pipeline.
struct VulkanComputePipelineHandle {
    std::uintptr_t shader_module = 0;
    std::uintptr_t pipeline_layout = 0;
    std::uintptr_t pipeline = 0;
    std::uintptr_t descriptor_set_layout = 0;
    bool valid = false;
};

/// Describes one buffer binding within a descriptor set.
struct VulkanBufferBinding {
    std::uint32_t binding = 0;
    VulkanBufferHandle buffer;
    std::size_t offset = 0;
    std::size_t range = 0;  ///< 0 = VK_WHOLE_SIZE.
    bool is_uniform = false;  ///< True for std140 uniform, false for SSBO.
};

/// Describes one image binding within a descriptor set.
struct VulkanImageBinding {
    std::uint32_t binding = 0;
    VulkanImageHandle image;
};

/// All Vulkan objects held for a single compute pipeline context.
struct VulkanComputeContext {
    std::uintptr_t device = 0;
    std::uintptr_t physical_device = 0;
    std::uintptr_t queue = 0;
    std::uint32_t queue_family_index = 0;
    std::uintptr_t command_pool = 0;
    std::uintptr_t descriptor_pool = 0;
    bool valid = false;
};

// Initialize a Vulkan compute context using the provided device/queue.
VulkanComputeContext CreateVulkanComputeContext(
        std::uintptr_t physical_device,
        std::uintptr_t device,
        std::uintptr_t queue,
        std::uint32_t queue_family_index,
        std::string* error_message);

void DestroyVulkanComputeContext(VulkanComputeContext& ctx);

// Create a compute pipeline from precompiled SPIR-V.
VulkanComputePipelineHandle CreateVulkanComputePipeline(
        const VulkanComputeContext& ctx,
        const std::vector<char>& spirv,
        const std::string& label,
        std::string* error_message);

void DestroyVulkanComputePipeline(const VulkanComputeContext& ctx,
                                  VulkanComputePipelineHandle& handle);

// Create a host-visible buffer (for uploading data from CPU).
VulkanBufferHandle CreateVulkanHostBuffer(const VulkanComputeContext& ctx,
                                          std::size_t size,
                                          bool uniform,
                                          const std::string& label,
                                          std::string* error_message);

// Create a device-local storage buffer (GPU-only).
VulkanBufferHandle CreateVulkanDeviceBuffer(const VulkanComputeContext& ctx,
                                            std::size_t size,
                                            const std::string& label,
                                            std::string* error_message);

void DestroyVulkanBuffer(const VulkanComputeContext& ctx,
                         VulkanBufferHandle& handle);

// Upload data to a host-visible buffer.
bool UploadVulkanBuffer(const VulkanBufferHandle& handle,
                        const void* data,
                        std::size_t size,
                        std::size_t offset);

// Download data from a host-visible buffer.
bool DownloadVulkanBuffer(const VulkanBufferHandle& handle,
                          void* data,
                          std::size_t size,
                          std::size_t offset);

// Create a storage image (for compute shader writes).
VulkanImageHandle CreateVulkanStorageImage(const VulkanComputeContext& ctx,
                                           std::uint32_t width,
                                           std::uint32_t height,
                                           std::uint32_t format,
                                           const std::string& label,
                                           std::string* error_message);

void DestroyVulkanImage(const VulkanComputeContext& ctx,
                        VulkanImageHandle& handle);

// Record and submit a compute dispatch with the given bindings.
// Blocks until completion (fence wait).
bool DispatchVulkanCompute(
        const VulkanComputeContext& ctx,
        const VulkanComputePipelineHandle& pipeline,
        const std::vector<VulkanBufferBinding>& buffer_bindings,
        const std::vector<VulkanImageBinding>& image_bindings,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y,
        std::uint32_t group_count_z,
        std::string* error_message);

// Dispatch with push constants.  \p push_data / \p push_size describe the raw
// bytes to push via vkCmdPushConstants at offset 0.
bool DispatchVulkanComputeWithPushConstants(
        const VulkanComputeContext& ctx,
        const VulkanComputePipelineHandle& pipeline,
        const std::vector<VulkanBufferBinding>& buffer_bindings,
        const void* push_data,
        std::uint32_t push_size,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y,
        std::uint32_t group_count_z,
        std::string* error_message);

// Create a compute pipeline from SPIR-V with push constants and a custom
// SSBO-only descriptor set layout with \p num_ssbo_bindings storage buffers
// at binding 0, 1, ..., num_ssbo_bindings-1.
VulkanComputePipelineHandle CreateVulkanComputePipelineWithPushConstants(
        const VulkanComputeContext& ctx,
        const std::vector<char>& spirv,
        std::uint32_t push_constant_size,
        std::uint32_t num_ssbo_bindings,
        const std::string& label,
        std::string* error_message);

// Transition an image from GENERAL to SHADER_READ_ONLY_OPTIMAL (for sampling).
bool TransitionImageForSampling(const VulkanComputeContext& ctx,
                                const VulkanImageHandle& image,
                                std::string* error_message);

// Download an image's pixel data to CPU.
bool DownloadVulkanImage(const VulkanComputeContext& ctx,
                         const VulkanImageHandle& image,
                         void* data,
                         std::size_t data_size,
                         std::string* error_message);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
