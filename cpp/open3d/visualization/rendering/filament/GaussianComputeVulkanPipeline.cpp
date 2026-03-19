// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Vulkan compute pipeline implementation for Gaussian splat rendering.
// Uses Filament's BlueVK dynamically-loaded Vulkan functions.

#include "open3d/visualization/rendering/filament/GaussianComputeVulkanPipeline.h"

#include <functional>  // std::function used by VulkanPlatform.h

// Include BlueVK for Vulkan function pointers (same as Filament uses).
#if __has_include(<bluevk/BlueVK.h>)
#include <bluevk/BlueVK.h>
#define OPEN3D_HAS_BLUEVK 1
#endif

#include <algorithm>
#include <array>
#include <cstring>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

#if defined(OPEN3D_HAS_BLUEVK)

using namespace bluevk;

namespace {

// Find a memory type index that satisfies the requirements.
uint32_t FindMemoryType(VkPhysicalDevice physical_device,
                        uint32_t type_filter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) ==
                    properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Submit a one-shot command buffer and wait for completion.
bool SubmitAndWait(VkDevice device,
                   VkQueue queue,
                   VkCommandPool pool,
                   const std::function<void(VkCommandBuffer)>& record,
                   std::string* error) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &alloc_info, &cmd) != VK_SUCCESS) {
        if (error) *error = "Failed to allocate command buffer.";
        return false;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);
    record(cmd);
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        if (error) *error = "Failed to create fence.";
        return false;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkResult result = vkQueueSubmit(queue, 1, &submit_info, fence);
    if (result != VK_SUCCESS) {
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        if (error)
            *error = "vkQueueSubmit failed (VkResult=" +
                     std::to_string(static_cast<int>(result)) + ").";
        return false;
    }

    VkResult wait_result = vkWaitForFences(device, 1, &fence, VK_TRUE,
                                             UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
    if (wait_result != VK_SUCCESS) {
        if (error)
            *error = "vkWaitForFences failed (VkResult=" +
                     std::to_string(static_cast<int>(wait_result)) + ").";
        return false;
    }
    return true;
}

}  // namespace

VulkanComputeContext CreateVulkanComputeContext(
        std::uintptr_t physical_device_handle,
        std::uintptr_t device_handle,
        std::uintptr_t queue_handle,
        std::uint32_t queue_family_index,
        std::string* error_message) {
    VulkanComputeContext ctx;
    auto device = reinterpret_cast<VkDevice>(device_handle);
    auto physical_device = reinterpret_cast<VkPhysicalDevice>(
            physical_device_handle);
    auto queue = reinterpret_cast<VkQueue>(queue_handle);
    if (!device || !physical_device || !queue) {
        if (error_message) *error_message = "Null Vulkan handle.";
        return ctx;
    }

    ctx.device = device_handle;
    ctx.physical_device = physical_device_handle;
    ctx.queue = queue_handle;
    ctx.queue_family_index = queue_family_index;

    // Create command pool.
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmd_pool;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool) !=
        VK_SUCCESS) {
        if (error_message) *error_message = "Failed to create command pool.";
        return ctx;
    }
    ctx.command_pool = reinterpret_cast<std::uintptr_t>(cmd_pool);

    // Create descriptor pool with enough room for the Gaussian passes
    // (5 base + up to 9 radix sort dispatches per frame).
    std::array<VkDescriptorPoolSize, 3> pool_sizes{};
    pool_sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 20};
    pool_sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 200};
    pool_sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 20};

    VkDescriptorPoolCreateInfo desc_pool_info{};
    desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    desc_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    desc_pool_info.maxSets = 80;
    desc_pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    desc_pool_info.pPoolSizes = pool_sizes.data();
    VkDescriptorPool desc_pool;
    if (vkCreateDescriptorPool(device, &desc_pool_info, nullptr, &desc_pool) !=
        VK_SUCCESS) {
        vkDestroyCommandPool(device, cmd_pool, nullptr);
        if (error_message)
            *error_message = "Failed to create descriptor pool.";
        return ctx;
    }
    ctx.descriptor_pool = reinterpret_cast<std::uintptr_t>(desc_pool);
    ctx.valid = true;
    return ctx;
}

void DestroyVulkanComputeContext(VulkanComputeContext& ctx) {
    if (!ctx.valid) return;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    if (ctx.descriptor_pool) {
        vkDestroyDescriptorPool(
                device,
                reinterpret_cast<VkDescriptorPool>(ctx.descriptor_pool),
                nullptr);
    }
    if (ctx.command_pool) {
        vkDestroyCommandPool(
                device, reinterpret_cast<VkCommandPool>(ctx.command_pool),
                nullptr);
    }
    ctx.valid = false;
}

VulkanComputePipelineHandle CreateVulkanComputePipeline(
        const VulkanComputeContext& ctx,
        const std::vector<char>& spirv,
        const std::string& label,
        std::string* error_message) {
    VulkanComputePipelineHandle handle;
    if (!ctx.valid || spirv.empty()) {
        if (error_message) *error_message = "Invalid context or empty SPIR-V.";
        return handle;
    }
    auto device = reinterpret_cast<VkDevice>(ctx.device);

    // Create shader module.
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size();
    module_info.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &module_info, nullptr, &shader_module) !=
        VK_SUCCESS) {
        if (error_message)
            *error_message = "Failed to create shader module for " + label;
        return handle;
    }
    handle.shader_module = reinterpret_cast<std::uintptr_t>(shader_module);

    // Create descriptor set layout.
    // Maximum bindings: 0..13 covers all Gaussian passes.
    static constexpr uint32_t kMaxBindings = 14;
    std::array<VkDescriptorSetLayoutBinding, kMaxBindings> bindings{};
    for (uint32_t i = 0; i < kMaxBindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        if (i == 0) {
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (i >= 12) {
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else {
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = kMaxBindings;
    layout_info.pBindings = bindings.data();
    VkDescriptorSetLayout desc_layout;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                    &desc_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create descriptor set layout for " + label;
        return handle;
    }
    handle.descriptor_set_layout =
            reinterpret_cast<std::uintptr_t>(desc_layout);

    // Create pipeline layout.
    VkPipelineLayoutCreateInfo pipe_layout_info{};
    pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.setLayoutCount = 1;
    pipe_layout_info.pSetLayouts = &desc_layout;
    VkPipelineLayout pipe_layout;
    if (vkCreatePipelineLayout(device, &pipe_layout_info, nullptr,
                               &pipe_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create pipeline layout for " + label;
        return handle;
    }
    handle.pipeline_layout = reinterpret_cast<std::uintptr_t>(pipe_layout);

    // Create compute pipeline.
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipe_layout;

    VkPipeline vk_pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &vk_pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipe_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create compute pipeline for " + label;
        return handle;
    }
    handle.pipeline = reinterpret_cast<std::uintptr_t>(vk_pipeline);
    handle.valid = true;
    return handle;
}

void DestroyVulkanComputePipeline(const VulkanComputeContext& ctx,
                                  VulkanComputePipelineHandle& handle) {
    if (!ctx.valid || !handle.valid) return;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    if (handle.pipeline) {
        vkDestroyPipeline(device,
                          reinterpret_cast<VkPipeline>(handle.pipeline),
                          nullptr);
    }
    if (handle.pipeline_layout) {
        vkDestroyPipelineLayout(
                device,
                reinterpret_cast<VkPipelineLayout>(handle.pipeline_layout),
                nullptr);
    }
    if (handle.descriptor_set_layout) {
        vkDestroyDescriptorSetLayout(
                device,
                reinterpret_cast<VkDescriptorSetLayout>(
                        handle.descriptor_set_layout),
                nullptr);
    }
    if (handle.shader_module) {
        vkDestroyShaderModule(
                device,
                reinterpret_cast<VkShaderModule>(handle.shader_module),
                nullptr);
    }
    handle = {};
}

VulkanBufferHandle CreateVulkanHostBuffer(const VulkanComputeContext& ctx,
                                          std::size_t size,
                                          bool uniform,
                                          const std::string& label,
                                          std::string* error_message) {
    VulkanBufferHandle handle;
    if (!ctx.valid || size == 0) return handle;

    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto phys = reinterpret_cast<VkPhysicalDevice>(ctx.physical_device);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (uniform) {
        usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    } else {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = usage;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    if (vkCreateBuffer(device, &buf_info, nullptr, &buffer) != VK_SUCCESS) {
        if (error_message) *error_message = "vkCreateBuffer failed: " + label;
        return handle;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

    uint32_t mem_type = FindMemoryType(
            phys, mem_reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(device, buffer, nullptr);
        if (error_message)
            *error_message = "No host-visible memory type for " + label;
        return handle;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory memory;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) !=
        VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        if (error_message) *error_message = "vkAllocateMemory failed: " + label;
        return handle;
    }

    vkBindBufferMemory(device, buffer, memory, 0);

    void* mapped = nullptr;
    vkMapMemory(device, memory, 0, size, 0, &mapped);

    handle.buffer = reinterpret_cast<std::uintptr_t>(buffer);
    handle.memory = reinterpret_cast<std::uintptr_t>(memory);
    handle.size = size;
    handle.mapped = mapped;
    handle.valid = true;
    return handle;
}

VulkanBufferHandle CreateVulkanDeviceBuffer(const VulkanComputeContext& ctx,
                                            std::size_t size,
                                            const std::string& label,
                                            std::string* error_message) {
    // For simplicity, use host-visible buffers so we can readback.
    // A production implementation would use staging buffers.
    return CreateVulkanHostBuffer(ctx, size, false, label, error_message);
}

void DestroyVulkanBuffer(const VulkanComputeContext& ctx,
                         VulkanBufferHandle& handle) {
    if (!ctx.valid || !handle.valid) return;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    if (handle.mapped) {
        vkUnmapMemory(device,
                      reinterpret_cast<VkDeviceMemory>(handle.memory));
        handle.mapped = nullptr;
    }
    if (handle.buffer) {
        vkDestroyBuffer(device, reinterpret_cast<VkBuffer>(handle.buffer),
                        nullptr);
    }
    if (handle.memory) {
        vkFreeMemory(device, reinterpret_cast<VkDeviceMemory>(handle.memory),
                     nullptr);
    }
    handle = {};
}

bool UploadVulkanBuffer(const VulkanBufferHandle& handle,
                        const void* data,
                        std::size_t size,
                        std::size_t offset) {
    if (!handle.valid || !handle.mapped || !data ||
        offset + size > handle.size) {
        return false;
    }
    std::memcpy(static_cast<char*>(handle.mapped) + offset, data, size);
    return true;
}

bool DownloadVulkanBuffer(const VulkanBufferHandle& handle,
                          void* data,
                          std::size_t size,
                          std::size_t offset) {
    if (!handle.valid || !handle.mapped || !data ||
        offset + size > handle.size) {
        return false;
    }
    std::memcpy(data, static_cast<const char*>(handle.mapped) + offset, size);
    return true;
}

VulkanImageHandle CreateVulkanStorageImage(const VulkanComputeContext& ctx,
                                           std::uint32_t width,
                                           std::uint32_t height,
                                           std::uint32_t format,
                                           const std::string& label,
                                           std::string* error_message) {
    VulkanImageHandle handle;
    if (!ctx.valid) return handle;

    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto phys = reinterpret_cast<VkPhysicalDevice>(ctx.physical_device);

    VkImageCreateInfo img_info{};
    img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_info.imageType = VK_IMAGE_TYPE_2D;
    img_info.format = static_cast<VkFormat>(format);
    img_info.extent = {width, height, 1};
    img_info.mipLevels = 1;
    img_info.arrayLayers = 1;
    img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_info.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    img_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    if (vkCreateImage(device, &img_info, nullptr, &image) != VK_SUCCESS) {
        if (error_message) *error_message = "vkCreateImage failed: " + label;
        return handle;
    }

    VkMemoryRequirements mem_reqs;
    vkGetImageMemoryRequirements(device, image, &mem_reqs);

    uint32_t mem_type = FindMemoryType(phys, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        vkDestroyImage(device, image, nullptr);
        if (error_message)
            *error_message = "No device-local memory for image: " + label;
        return handle;
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = mem_reqs.size;
    alloc.memoryTypeIndex = mem_type;
    VkDeviceMemory memory;
    if (vkAllocateMemory(device, &alloc, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyImage(device, image, nullptr);
        if (error_message)
            *error_message = "vkAllocateMemory failed for image: " + label;
        return handle;
    }
    vkBindImageMemory(device, image, memory, 0);

    // Create image view.
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = static_cast<VkFormat>(format);
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    VkImageView image_view;
    if (vkCreateImageView(device, &view_info, nullptr, &image_view) !=
        VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        if (error_message)
            *error_message = "vkCreateImageView failed: " + label;
        return handle;
    }

    // Transition to GENERAL layout for compute writes.
    std::string barrier_error;
    bool ok = SubmitAndWait(
            device, reinterpret_cast<VkQueue>(ctx.queue),
            reinterpret_cast<VkCommandPool>(ctx.command_pool),
            [&](VkCommandBuffer cmd) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &barrier);
            },
            &barrier_error);
    if (!ok) {
        vkDestroyImageView(device, image_view, nullptr);
        vkFreeMemory(device, memory, nullptr);
        vkDestroyImage(device, image, nullptr);
        if (error_message)
            *error_message = "Image layout transition failed: " + barrier_error;
        return handle;
    }

    handle.image = reinterpret_cast<std::uintptr_t>(image);
    handle.image_view = reinterpret_cast<std::uintptr_t>(image_view);
    handle.memory = reinterpret_cast<std::uintptr_t>(memory);
    handle.width = width;
    handle.height = height;
    handle.format = format;
    handle.valid = true;
    return handle;
}

void DestroyVulkanImage(const VulkanComputeContext& ctx,
                        VulkanImageHandle& handle) {
    if (!ctx.valid || !handle.valid) return;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    if (handle.image_view) {
        vkDestroyImageView(device,
                           reinterpret_cast<VkImageView>(handle.image_view),
                           nullptr);
    }
    if (handle.image) {
        vkDestroyImage(device, reinterpret_cast<VkImage>(handle.image),
                       nullptr);
    }
    if (handle.memory) {
        vkFreeMemory(device, reinterpret_cast<VkDeviceMemory>(handle.memory),
                     nullptr);
    }
    handle = {};
}

bool DispatchVulkanCompute(
        const VulkanComputeContext& ctx,
        const VulkanComputePipelineHandle& pipeline,
        const std::vector<VulkanBufferBinding>& buffer_bindings,
        const std::vector<VulkanImageBinding>& image_bindings,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y,
        std::uint32_t group_count_z,
        std::string* error_message) {
    if (!ctx.valid || !pipeline.valid) {
        if (error_message) *error_message = "Invalid context or pipeline.";
        return false;
    }

    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto desc_pool = reinterpret_cast<VkDescriptorPool>(ctx.descriptor_pool);
    auto desc_layout = reinterpret_cast<VkDescriptorSetLayout>(
            pipeline.descriptor_set_layout);

    // Allocate descriptor set.
    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool = desc_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts = &desc_layout;
    VkDescriptorSet desc_set;
    if (vkAllocateDescriptorSets(device, &ds_alloc, &desc_set) !=
        VK_SUCCESS) {
        if (error_message)
            *error_message = "Failed to allocate descriptor set.";
        return false;
    }

    // Write buffer descriptors.
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buf_infos(buffer_bindings.size());
    std::vector<VkDescriptorImageInfo> img_infos(image_bindings.size());

    for (size_t i = 0; i < buffer_bindings.size(); ++i) {
        const auto& bb = buffer_bindings[i];
        buf_infos[i].buffer = reinterpret_cast<VkBuffer>(bb.buffer.buffer);
        buf_infos[i].offset = bb.offset;
        buf_infos[i].range = bb.range == 0 ? VK_WHOLE_SIZE : bb.range;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = desc_set;
        write.dstBinding = bb.binding;
        write.descriptorCount = 1;
        write.descriptorType = bb.is_uniform
                                       ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buf_infos[i];
        writes.push_back(write);
    }

    for (size_t i = 0; i < image_bindings.size(); ++i) {
        const auto& ib = image_bindings[i];
        img_infos[i].imageView =
                reinterpret_cast<VkImageView>(ib.image.image_view);
        img_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img_infos[i].sampler = VK_NULL_HANDLE;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = desc_set;
        write.dstBinding = ib.binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &img_infos[i];
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    // Record and submit compute dispatch.
    auto queue = reinterpret_cast<VkQueue>(ctx.queue);
    auto cmd_pool = reinterpret_cast<VkCommandPool>(ctx.command_pool);
    auto pipe = reinterpret_cast<VkPipeline>(pipeline.pipeline);
    auto pipe_layout =
            reinterpret_cast<VkPipelineLayout>(pipeline.pipeline_layout);

    bool ok = SubmitAndWait(
            device, queue, cmd_pool,
            [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipe_layout, 0, 1, &desc_set, 0,
                                        nullptr);
                vkCmdDispatch(cmd, std::max(group_count_x, 1u),
                              std::max(group_count_y, 1u),
                              std::max(group_count_z, 1u));

                // Memory barrier between compute passes.
                VkMemoryBarrier mem_barrier{};
                mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mem_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 1, &mem_barrier, 0, nullptr, 0,
                                     nullptr);
            },
            error_message);

    // Free the descriptor set.
    vkFreeDescriptorSets(device, desc_pool, 1, &desc_set);
    return ok;
}

VulkanComputePipelineHandle CreateVulkanComputePipelineWithPushConstants(
        const VulkanComputeContext& ctx,
        const std::vector<char>& spirv,
        std::uint32_t push_constant_size,
        std::uint32_t num_ssbo_bindings,
        const std::string& label,
        std::string* error_message) {
    VulkanComputePipelineHandle handle;
    if (!ctx.valid || spirv.empty()) {
        if (error_message) *error_message = "Invalid context or empty SPIR-V.";
        return handle;
    }
    auto device = reinterpret_cast<VkDevice>(ctx.device);

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv.size();
    module_info.pCode = reinterpret_cast<const uint32_t*>(spirv.data());
    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &module_info, nullptr, &shader_module) !=
        VK_SUCCESS) {
        if (error_message)
            *error_message = "Failed to create shader module for " + label;
        return handle;
    }
    handle.shader_module = reinterpret_cast<std::uintptr_t>(shader_module);

    // SSBO-only descriptor set layout.
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_ssbo_bindings);
    for (uint32_t i = 0; i < num_ssbo_bindings; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = num_ssbo_bindings;
    layout_info.pBindings = bindings.data();
    VkDescriptorSetLayout desc_layout;
    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                    &desc_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create descriptor set layout for " + label;
        return handle;
    }
    handle.descriptor_set_layout =
            reinterpret_cast<std::uintptr_t>(desc_layout);

    // Pipeline layout with push constants.
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;

    VkPipelineLayoutCreateInfo pipe_layout_info{};
    pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_info.setLayoutCount = 1;
    pipe_layout_info.pSetLayouts = &desc_layout;
    pipe_layout_info.pushConstantRangeCount = 1;
    pipe_layout_info.pPushConstantRanges = &push_range;
    VkPipelineLayout pipe_layout;
    if (vkCreatePipelineLayout(device, &pipe_layout_info, nullptr,
                               &pipe_layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create pipeline layout for " + label;
        return handle;
    }
    handle.pipeline_layout = reinterpret_cast<std::uintptr_t>(pipe_layout);

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipe_layout;

    VkPipeline vk_pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &vk_pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, pipe_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);
        if (error_message)
            *error_message =
                    "Failed to create compute pipeline for " + label;
        return handle;
    }
    handle.pipeline = reinterpret_cast<std::uintptr_t>(vk_pipeline);
    handle.valid = true;
    return handle;
}

bool DispatchVulkanComputeWithPushConstants(
        const VulkanComputeContext& ctx,
        const VulkanComputePipelineHandle& pipeline,
        const std::vector<VulkanBufferBinding>& buffer_bindings,
        const void* push_data,
        std::uint32_t push_size,
        std::uint32_t group_count_x,
        std::uint32_t group_count_y,
        std::uint32_t group_count_z,
        std::string* error_message) {
    if (!ctx.valid || !pipeline.valid) {
        if (error_message) *error_message = "Invalid context or pipeline.";
        return false;
    }

    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto desc_pool = reinterpret_cast<VkDescriptorPool>(ctx.descriptor_pool);
    auto desc_layout = reinterpret_cast<VkDescriptorSetLayout>(
            pipeline.descriptor_set_layout);

    VkDescriptorSetAllocateInfo ds_alloc{};
    ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_alloc.descriptorPool = desc_pool;
    ds_alloc.descriptorSetCount = 1;
    ds_alloc.pSetLayouts = &desc_layout;
    VkDescriptorSet desc_set;
    if (vkAllocateDescriptorSets(device, &ds_alloc, &desc_set) !=
        VK_SUCCESS) {
        if (error_message)
            *error_message = "Failed to allocate descriptor set.";
        return false;
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buf_infos(buffer_bindings.size());
    for (size_t i = 0; i < buffer_bindings.size(); ++i) {
        const auto& bb = buffer_bindings[i];
        buf_infos[i].buffer = reinterpret_cast<VkBuffer>(bb.buffer.buffer);
        buf_infos[i].offset = bb.offset;
        buf_infos[i].range = bb.range == 0 ? VK_WHOLE_SIZE : bb.range;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = desc_set;
        write.dstBinding = bb.binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buf_infos[i];
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    auto queue = reinterpret_cast<VkQueue>(ctx.queue);
    auto cmd_pool = reinterpret_cast<VkCommandPool>(ctx.command_pool);
    auto pipe = reinterpret_cast<VkPipeline>(pipeline.pipeline);
    auto pipe_layout =
            reinterpret_cast<VkPipelineLayout>(pipeline.pipeline_layout);

    bool ok = SubmitAndWait(
            device, queue, cmd_pool,
            [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipe_layout, 0, 1, &desc_set, 0,
                                        nullptr);
                if (push_data && push_size > 0) {
                    vkCmdPushConstants(cmd, pipe_layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       push_size, push_data);
                }
                vkCmdDispatch(cmd, std::max(group_count_x, 1u),
                              std::max(group_count_y, 1u),
                              std::max(group_count_z, 1u));

                VkMemoryBarrier mem_barrier{};
                mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mem_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                            VK_ACCESS_SHADER_WRITE_BIT |
                                            VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     0, 1, &mem_barrier, 0, nullptr, 0,
                                     nullptr);
            },
            error_message);

    vkFreeDescriptorSets(device, desc_pool, 1, &desc_set);
    return ok;
}

bool TransitionImageForSampling(const VulkanComputeContext& ctx,
                                const VulkanImageHandle& image,
                                std::string* error_message) {
    if (!ctx.valid || !image.valid) return false;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto queue = reinterpret_cast<VkQueue>(ctx.queue);
    auto pool = reinterpret_cast<VkCommandPool>(ctx.command_pool);

    return SubmitAndWait(
            device, queue, pool,
            [&](VkCommandBuffer cmd) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = reinterpret_cast<VkImage>(image.image);
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &barrier);
            },
            error_message);
}

bool DownloadVulkanImage(const VulkanComputeContext& ctx,
                         const VulkanImageHandle& image,
                         void* data,
                         std::size_t data_size,
                         std::string* error_message) {
    if (!ctx.valid || !image.valid || !data) return false;
    auto device = reinterpret_cast<VkDevice>(ctx.device);
    auto queue = reinterpret_cast<VkQueue>(ctx.queue);
    auto pool = reinterpret_cast<VkCommandPool>(ctx.command_pool);

    // Create staging buffer.
    VulkanBufferHandle staging =
            CreateVulkanHostBuffer(ctx, data_size, false, "image_staging",
                                   error_message);
    if (!staging.valid) return false;

    bool ok = SubmitAndWait(
            device, queue, pool,
            [&](VkCommandBuffer cmd) {
                // Transition to transfer src.
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = reinterpret_cast<VkImage>(image.image);
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &barrier);

                // Copy image to buffer.
                VkBufferImageCopy region{};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = {image.width, image.height, 1};
                vkCmdCopyImageToBuffer(
                        cmd, reinterpret_cast<VkImage>(image.image),
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        reinterpret_cast<VkBuffer>(staging.buffer), 1,
                        &region);

                // Transition back to GENERAL.
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &barrier);
            },
            error_message);

    if (ok && staging.mapped) {
        std::memcpy(data, staging.mapped, data_size);
    }

    DestroyVulkanBuffer(ctx, staging);
    return ok;
}

#else  // !OPEN3D_HAS_BLUEVK

// Stubs when Vulkan headers are not available.

VulkanComputeContext CreateVulkanComputeContext(
        std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uint32_t,
        std::string* e) {
    if (e) *e = "Vulkan compute not available (BlueVK not found).";
    return {};
}

void DestroyVulkanComputeContext(VulkanComputeContext&) {}

VulkanComputePipelineHandle CreateVulkanComputePipeline(
        const VulkanComputeContext&, const std::vector<char>&,
        const std::string&, std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return {};
}

void DestroyVulkanComputePipeline(const VulkanComputeContext&,
                                  VulkanComputePipelineHandle&) {}

VulkanBufferHandle CreateVulkanHostBuffer(const VulkanComputeContext&,
                                          std::size_t, bool,
                                          const std::string&, std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return {};
}

VulkanBufferHandle CreateVulkanDeviceBuffer(const VulkanComputeContext&,
                                            std::size_t,
                                            const std::string&,
                                            std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return {};
}

void DestroyVulkanBuffer(const VulkanComputeContext&, VulkanBufferHandle&) {}

bool UploadVulkanBuffer(const VulkanBufferHandle&, const void*, std::size_t,
                        std::size_t) {
    return false;
}
bool DownloadVulkanBuffer(const VulkanBufferHandle&, void*, std::size_t,
                          std::size_t) {
    return false;
}

VulkanImageHandle CreateVulkanStorageImage(const VulkanComputeContext&,
                                           std::uint32_t, std::uint32_t,
                                           std::uint32_t, const std::string&,
                                           std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return {};
}

void DestroyVulkanImage(const VulkanComputeContext&, VulkanImageHandle&) {}

bool DispatchVulkanCompute(const VulkanComputeContext&,
                           const VulkanComputePipelineHandle&,
                           const std::vector<VulkanBufferBinding>&,
                           const std::vector<VulkanImageBinding>&,
                           std::uint32_t, std::uint32_t, std::uint32_t,
                           std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return false;
}

bool TransitionImageForSampling(const VulkanComputeContext&,
                                const VulkanImageHandle&, std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return false;
}

bool DownloadVulkanImage(const VulkanComputeContext&, const VulkanImageHandle&,
                         void*, std::size_t, std::string* e) {
    if (e) *e = "Vulkan compute not available.";
    return false;
}

#endif  // OPEN3D_HAS_BLUEVK

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
