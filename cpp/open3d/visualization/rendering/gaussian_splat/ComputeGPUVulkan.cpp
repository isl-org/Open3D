// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan 1.3 compute backend for GaussianSplatGpuContext.
// One VkPipeline per ComputeProgramId; per-dispatch bindings pushed via
// VK_KHR_push_descriptor (no descriptor pool management per frame).
// VMA handles general buffer and image allocation.

#if !defined(__APPLE__)

#include "open3d/visualization/rendering/gaussian_splat/ComputeGPUVulkan.h"

#include <cassert>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// vulkan-hpp RAII: dynamic dispatch through the per-object DeviceDispatcher;
// VK_NO_PROTOTYPES is defined transitively via
// GaussianSplatVulkanContext.h. The global
// VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE is defined in
// GaussianSplatVulkanContext.cpp (exactly once in the program).

// VMA: header-only allocator (implementation in
// GaussianSplatVulkanContext.cpp).
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanContext.h"
#include "vk_mem_alloc.hpp"

namespace open3d {
namespace visualization {
namespace rendering {

// ---------------------------------------------------------------------------
// Per-pipeline (per-shader) descriptor set layout descriptor.
// Declares the exact bindings the shader uses; push descriptors require this.
// ---------------------------------------------------------------------------
struct ShaderBindingDesc {
    std::uint32_t binding;
    VkDescriptorType type;
    VkImageLayout
            image_layout;  // Only for STORAGE_IMAGE / COMBINED_IMAGE_SAMPLER
};

// Binding tables derived from SPIR-V analysis (spirv-dis) of each compiled
// shader.  image_layout is VK_IMAGE_LAYOUT_UNDEFINED for buffer bindings.
//
// NOTE: out_color in composite uses binding 16 (not 0) to avoid conflicting
//       with the GaussianViewParams UBO at B0.  depth_merge gs_depth uses 15
//       so Metal texture/sampler slots stay in 0..15.

static constexpr ShaderBindingDesc kBindingsProject[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsComposite[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL},
};
static constexpr ShaderBindingDesc kBindingsRadixHistograms[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsRadixScatter[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsDispatchArgs[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsDepthMerge[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL},
        {14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
};

// Table indexed by ComputeProgramId: binding descriptor + count.
struct ShaderBindingTable {
    const ShaderBindingDesc* descs;
    std::uint32_t count;
};

static constexpr ShaderBindingTable kShaderBindings[] = {
        {kBindingsProject, std::size(kBindingsProject)},
        {kBindingsComposite, std::size(kBindingsComposite)},
        {kBindingsRadixHistograms, std::size(kBindingsRadixHistograms)},
        {kBindingsRadixScatter, std::size(kBindingsRadixScatter)},
        {kBindingsDispatchArgs, std::size(kBindingsDispatchArgs)},
        {kBindingsDepthMerge, std::size(kBindingsDepthMerge)},
};
static_assert(std::size(kShaderBindings) ==
                      static_cast<std::size_t>(ComputeProgramId::kCount),
              "kShaderBindings must match ComputeProgramId::kCount");

namespace {

/// Synchronization scope for one side of an image layout transition.
struct ImageSyncScope {
    vk::PipelineStageFlags2 stage;
    vk::AccessFlags2 access;
};

/// Scope covering all prior accesses that the old layout implies.
ImageSyncScope SrcScopeForLayout(VkImageLayout layout) {
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return {Stage::eNone, {}};
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return {Stage::eColorAttachmentOutput,
                    Access::eColorAttachmentWrite};
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            return {Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                    Access::eDepthStencilAttachmentWrite};
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return {Stage::eTransfer, Access::eTransferRead};
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Read-only layout: declaring eShaderWrite here would be incorrect
            // and trips BestPractices-ImageBarrierAccessLayout.
            return {Stage::eComputeShader, Access::eShaderRead};
        default:
            return {Stage::eComputeShader,
                    Access::eShaderWrite | Access::eShaderRead};
    }
}

/// Scope covering all subsequent accesses that the new layout enables.
ImageSyncScope DstScopeForLayout(VkImageLayout layout) {
    using Stage = vk::PipelineStageFlagBits2;
    using Access = vk::AccessFlagBits2;
    switch (layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return {Stage::eTransfer, Access::eTransferRead};
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            return {Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                    Access::eDepthStencilAttachmentRead |
                            Access::eDepthStencilAttachmentWrite};
        case VK_IMAGE_LAYOUT_GENERAL:
            return {Stage::eComputeShader,
                    Access::eShaderWrite | Access::eShaderRead};
        default:
            return {Stage::eComputeShader, Access::eShaderRead};
    }
}

vk::ImageSubresourceRange FullSubresource(VkFormat format) {
    return {format == VK_FORMAT_D32_SFLOAT ? vk::ImageAspectFlagBits::eDepth
                                           : vk::ImageAspectFlagBits::eColor,
            0, 1, 0, 1};
}

}  // namespace

// ---------------------------------------------------------------------------
// Vulkan compute backend class
// ---------------------------------------------------------------------------
class GaussianSplatGpuContextVulkan final : public GaussianSplatGpuContext {
public:
    GaussianSplatGpuContextVulkan() = default;

    ~GaussianSplatGpuContextVulkan() override { Shutdown(); }

    // --- Program management -----------------------------------------------

    bool EnsureProgramsLoaded() override {
        if (programs_loaded_) return programs_valid_;
        programs_loaded_ = true;
        programs_valid_ = false;

        auto& vk_ctx = GaussianSplatVulkanContext::GetInstance();
        if (!vk_ctx.IsValid()) {
            utility::LogWarning(
                    "GaussianSplatVulkan: Vulkan context not initialized");
            return false;
        }

        device_ = vk_ctx.GetDevice();
        physical_device_ = vk_ctx.GetPhysicalDevice();
        compute_queue_ = vk_ctx.GetComputeQueue();
        queue_family_ = vk_ctx.GetComputeQueueFamily();
        debug_utils_enabled_ = vk_ctx.GetDebugUtilsEnabled();
        NameObject(vk::ObjectType::eQueue,
                   reinterpret_cast<std::uintptr_t>(compute_queue_),
                   "gs.queue.compute");

        if (!InitVma()) return false;
        if (!InitDeviceObjects()) return false;

        // Shaders are compiled with -V --target-env vulkan1.3 by
        // open3d_add_compute_shaders.
        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_splat/";

        for (int i = 0; i < static_cast<int>(ComputeProgramId::kCount); ++i) {
            if (!LoadShader(static_cast<ComputeProgramId>(i), shader_root)) {
                utility::LogWarning(
                        "GaussianSplatVulkan: failed to load shader {}",
                        kGsShaderNames[i]);
                return false;
            }
        }
        programs_valid_ = true;
        // Cache the device's maximum compute workgroup count along X so
        // PassRunner can split large dispatches without hardcoding 65535.
        // Use vkGetPhysicalDeviceProperties2 (Vulkan 1.1+) to avoid the
        // legacy-command validation warning triggered by the v1.0 variant.
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(
                physical_device_, &props2);
        max_wg_count_x_ = props2.properties.limits.maxComputeWorkGroupCount[0];
        utility::LogDebug("GaussianSplatVulkan: programs loaded");
        return true;
    }

    std::uint32_t GetMaxComputeWorkGroupCount() const override {
        // Returns the cached value queried at program-load time.
        // Falls back to the Vulkan-mandated minimum (65535) if not yet loaded.
        return max_wg_count_x_ > 0 ? max_wg_count_x_ : 65535u;
    }

    // --- Buffer management ------------------------------------------------

    std::uintptr_t CreateBuffer(std::size_t size,
                                const char* label = nullptr) override {
        return AllocBuf(size, false, label);
    }

    std::uintptr_t CreatePrivateBuffer(std::size_t size,
                                       const char* label = nullptr) override {
        return AllocBuf(size, true, label);
    }

    void DestroyBuffer(std::uintptr_t buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        auto& e = it->second;
        // Persistent-mapped allocations (CPU_TO_GPU +
        // VMA_ALLOCATION_CREATE_MAPPED_BIT) are managed by VMA internally; do
        // NOT call vmaUnmapMemory on them. vmaDestroyBuffer handles cleanup
        // including the persistent mapping.
        vmaDestroyBuffer(vma_, e.buffer, e.alloc);
        buffers_.erase(it);
    }

    std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                std::size_t new_size,
                                const char* label = nullptr) override {
        return ReallocBuf(buf, new_size, false, label);
    }

    std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                       std::size_t new_size,
                                       const char* label = nullptr) override {
        if (new_size == 0) {
            DestroyBuffer(buf);
            return 0;
        }
        return ReallocBuf(buf, new_size, true, label);
    }

    void UploadBuffer(std::uintptr_t buf,
                      const void* data,
                      std::size_t size,
                      std::size_t offset) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        auto& e = it->second;
        void* mapped = e.mapped;
        if (!mapped) {
            if (vmaMapMemory(vma_, e.alloc, &mapped) != VK_SUCCESS) return;
        }
        std::memcpy(static_cast<char*>(mapped) + offset, data, size);
        if (!e.mapped) {
            vmaFlushAllocation(vma_, e.alloc, offset, size);
            vmaUnmapMemory(vma_, e.alloc);
        }
    }

    bool DownloadBuffer(std::uintptr_t buf,
                        void* dst,
                        std::size_t size,
                        std::size_t offset) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return false;
        auto& e = it->second;
        if (e.is_private) return false;
        void* mapped = e.mapped;
        if (!mapped) {
            if (vmaMapMemory(vma_, e.alloc, &mapped) != VK_SUCCESS)
                return false;
        }
        vmaInvalidateAllocation(vma_, e.alloc, offset, size);
        std::memcpy(dst, static_cast<const char*>(mapped) + offset, size);
        if (!e.mapped) vmaUnmapMemory(vma_, e.alloc);
        return true;
    }

    void ClearBufferUInt32Zero(std::uintptr_t buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        cmd_.fillBuffer(vk::Buffer(it->second.buffer), 0, VK_WHOLE_SIZE, 0u);
    }

    // --- Bindings ---------------------------------------------------------

    void BindSSBO(std::uint32_t binding, std::uintptr_t buf) override {
        PushBufferWrite(binding, buf, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0,
                        VK_WHOLE_SIZE);
    }

    void BindUBO(std::uint32_t binding, std::uintptr_t buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        PushBufferWrite(binding, buf, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
                        it->second.size);
    }

    void BindUBORange(std::uint32_t binding,
                      std::uintptr_t buf,
                      std::size_t offset,
                      std::size_t range_size) override {
        PushBufferWrite(binding, buf, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, offset,
                        range_size);
    }

    void BindImage(std::uint32_t binding,
                   std::uintptr_t tex,
                   std::uint32_t /*width*/,
                   std::uint32_t /*height*/,
                   ImageFormat /*fmt*/) override {
        // Depth images (D32_SFLOAT) lack VK_IMAGE_USAGE_STORAGE_BIT and
        // cannot be bound as STORAGE_IMAGE.  This indicates a caller bug
        // (wrong handle); skip silently so we don't trigger VUID-00339.
        auto it = textures_.find(tex);
        if (it == textures_.end()) {
            return;
        }
        if (it->second.format == VK_FORMAT_D32_SFLOAT) {
            utility::LogWarning(
                    "GaussianSplatVulkan: BindImage(binding={}) skipped — "
                    "handle resolves to a depth image which cannot be a "
                    "STORAGE_IMAGE. Check handle/GL-name mapping.",
                    binding);
            return;
        }
        PushImageWrite(binding, tex, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                       VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE);
    }

    void BindSamplerTexture(std::uint32_t unit,
                            std::uintptr_t tex,
                            std::uint32_t /*width*/,
                            std::uint32_t /*height*/) override {
        PushImageWrite(unit, tex, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       static_cast<VkSampler>(*nearest_sampler_));
    }

    // --- Dispatch ---------------------------------------------------------

    void UseProgram(ComputeProgramId id) override {
        int i = static_cast<int>(id);
        if (i < 0 || i >= static_cast<int>(ComputeProgramId::kCount)) return;
        auto& p = pipelines_[i];
        if (!p.valid) return;
        active_id_ = i;
        cmd_.bindPipeline(vk::PipelineBindPoint::eCompute, *p.pipeline);
        pending_.clear();
    }

    void Dispatch(std::uint32_t gx,
                  std::uint32_t gy,
                  std::uint32_t gz) override {
        FlushPendingBindings();
        cmd_.dispatch(gx, gy, gz);
    }

    void DispatchIndirect(std::uintptr_t indirect_buf,
                          std::size_t byte_offset) override {
        auto it = buffers_.find(indirect_buf);
        if (it == buffers_.end()) return;
        FlushPendingBindings();
        cmd_.dispatchIndirect(vk::Buffer(it->second.buffer),
                              static_cast<vk::DeviceSize>(byte_offset));
    }

    void FullBarrier() override {
        // Full compute+transfer memory barrier using Vulkan 1.3
        // synchronization2.
        vk::MemoryBarrier2 mb{
                vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eMemoryWrite,
                vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eTransfer |
                        vk::PipelineStageFlagBits2::eDrawIndirect,
                vk::AccessFlagBits2::eMemoryRead |
                        vk::AccessFlagBits2::eMemoryWrite |
                        vk::AccessFlagBits2::eIndirectCommandRead,
        };
        cmd_.pipelineBarrier2(vk::DependencyInfo{{}, mb, {}, {}});
    }

    // --- Textures / images ------------------------------------------------

    std::uintptr_t CreateTexture2DR32F(std::uint32_t w,
                                       std::uint32_t h,
                                       const char* label) override {
        return AllocTex(w, h, VK_FORMAT_R32_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT, label);
    }

    void DestroyTexture(std::uintptr_t tex) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return;
        auto& e = it->second;
        // Explicitly destroy the view before the VMA image for correct order.
        e.view = vk::raii::ImageView{nullptr};
        if (e.alloc != VK_NULL_HANDLE) vmaDestroyImage(vma_, e.image, e.alloc);
        textures_.erase(it);
    }

    std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                       std::uint32_t w,
                                       std::uint32_t h,
                                       const char* label) override {
        if (KeepTextureOfSize(tex, w, h)) return tex;
        return CreateTexture2DR32F(w, h, label);
    }

    std::uintptr_t ResizeTexture2DR16UI(std::uintptr_t tex,
                                        std::uint32_t w,
                                        std::uint32_t h,
                                        const char* label) override {
        if (KeepTextureOfSize(tex, w, h)) return tex;
        return AllocTex(
                w, h, VK_FORMAT_R16_UINT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, label);
    }

    bool DownloadTextureR32F(std::uintptr_t tex,
                             std::uint32_t w,
                             std::uint32_t h,
                             std::vector<float>& out) override {
        return DownloadTexBytes(tex, w, h, sizeof(float), out);
    }

    bool DownloadTextureR16UI(std::uintptr_t tex,
                              std::uint32_t w,
                              std::uint32_t h,
                              std::vector<std::uint16_t>& out) override {
        return DownloadTexBytes(tex, w, h, sizeof(std::uint16_t), out);
    }

    bool DownloadTextureRGBA16F(std::uintptr_t tex,
                                std::uint32_t w,
                                std::uint32_t h,
                                std::vector<std::uint16_t>& out) override {
        return DownloadTexBytes(tex, w, h, 4 * sizeof(std::uint16_t), out);
    }

    // --- Frame boundary ---------------------------------------------------

    void BeginGeometryPass() override { BeginCmdBuf(); }
    // Fire-and-forget: submit and signal the fence but do NOT block.
    // Filament rendering proceeds in parallel; WaitForGeometryPass() (called
    // at the start of RunGaussianCompositePass) drains it only if needed.
    void EndGeometryPass() override { SubmitOnly(); }
    // Images shared with Filament live on the same VkDevice and queue family,
    // so no queue-family ownership transfer is needed. FilamentRenderer's
    // flushAndWait() bracketing provides the cross-API execution ordering, and
    // ResolveImageView() emits any required layout transition per binding.
    void BeginCompositePass() override {
        BeginCmdBuf();
        for (auto& item : textures_) {
            auto& entry = item.second;
            entry.used_in_composite = false;
        }
    }
    void EndCompositePass() override {
        ReleaseImportedImages();
        SubmitAndWait();
    }

    void WaitForGeometryPass() override { WaitForPendingSubmit(); }

    void FinishGpuWork() override {
        if (!cmd_active_) return;
        SubmitAndWait();
    }

    void PushDebugGroup(const char* label) override {
        if (!cmd_active_) return;
        if (!debug_utils_enabled_) return;
        auto fn = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT;
        if (!fn) return;
        VkDebugUtilsLabelEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pLabelName = label;
        fn(static_cast<VkCommandBuffer>(*cmd_), &info);
    }
    void PopDebugGroup() override {
        if (!cmd_active_) return;
        if (!debug_utils_enabled_) return;
        auto fn = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT;
        if (fn) fn(static_cast<VkCommandBuffer>(*cmd_));
    }

    // --- Shared-image registration (called by GaussianSplatVulkanBackend) -

    void RegisterVkImage(VkImage image,
                         VkFormat format,
                         std::uint32_t w,
                         std::uint32_t h) {
        VkImageAspectFlags aspect = (format == VK_FORMAT_D32_SFLOAT)
                                            ? VK_IMAGE_ASPECT_DEPTH_BIT
                                            : VK_IMAGE_ASPECT_COLOR_BIT;
        vk::raii::ImageView view = CreateImageView(image, format, aspect);
        if (static_cast<VkImageView>(*view) == VK_NULL_HANDLE) {
            utility::LogWarning(
                    "GaussianSplatVulkan: failed to create view for "
                    "VkImage={}",
                    reinterpret_cast<uintptr_t>(image));
            return;
        }
        TexEntry e{};
        e.image = image;
        e.view = std::move(view);
        e.alloc = VK_NULL_HANDLE;  // owned externally
        e.format = format;
        e.width = w;
        e.height = h;
        // Composite starts only after Filament has completed its render pass.
        // Filament keeps color attachments in GENERAL and depth attachments in
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        e.current_layout =
                (format == VK_FORMAT_D32_SFLOAT)
                        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        : VK_IMAGE_LAYOUT_GENERAL;
        e.imported = true;
        uintptr_t handle = reinterpret_cast<uintptr_t>(image);
        textures_[handle] = std::move(e);
    }

    void UnregisterVkImage(VkImage image) {
        auto it = textures_.find(reinterpret_cast<uintptr_t>(image));
        if (it == textures_.end()) return;
        // vk::raii::ImageView in TexEntry auto-destroyed on erase
        textures_.erase(it);
    }

private:
    // --- Internal state ---------------------------------------------------
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    vk::raii::CommandPool cmd_pool_{nullptr};
    vk::raii::CommandBuffer cmd_{nullptr};
    vk::raii::Fence fence_{nullptr};
    vk::raii::Sampler nearest_sampler_{nullptr};
    bool cmd_active_ = false;
    bool fence_submitted_ = false;  // true while fence has a pending submission

    struct Pipeline {
        vk::raii::DescriptorSetLayout dset_layout{nullptr};
        vk::raii::PipelineLayout layout{nullptr};
        vk::raii::Pipeline pipeline{nullptr};
        bool valid = false;
        // Bitmask of valid bindings so we can filter writes.
        std::uint64_t binding_mask = 0;  // bit i set ↔ binding i exists
    };
    Pipeline pipelines_[static_cast<int>(ComputeProgramId::kCount)];
    int active_id_ = -1;

    struct BufEntry {
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        std::size_t size = 0;
        void* mapped = nullptr;
        bool is_private = false;
    };
    std::unordered_map<uintptr_t, BufEntry> buffers_;

    struct TexEntry {
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        vk::raii::ImageView view{
                nullptr};  // RAII: auto-destroys vkDestroyImageView
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool imported = false;
        bool used_in_composite = false;
    };
    std::unordered_map<uintptr_t, TexEntry> textures_;
    // Start at a large value so internal handles never collide with VkImage
    // pointer values (externally-registered images key by VkImage pointer).
    std::uint64_t next_handle_ = 0x80000000ULL;

    struct PendingWrite {
        std::uint32_t binding = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorBufferInfo buf{};
        VkDescriptorImageInfo img{};
    };
    std::vector<PendingWrite> pending_;

    bool programs_loaded_ = false;
    bool programs_valid_ = false;
    std::uint32_t max_wg_count_x_ = 0;  // cached from VkPhysicalDeviceLimits
    bool debug_utils_enabled_ =
            false;  // VK_EXT_debug_utils enabled at instance

    // --- Init helpers -----------------------------------------------------

    bool InitVma() {
        // VMA uses dynamic Vulkan dispatch (configured in
        // GaussianSplatVulkanContext.cpp). This must match Filament's VMA
        // settings. The helper fills its loader callbacks and Vulkan functions
        // from the initialized Hpp dispatcher.
        VmaVulkanFunctions vk_fn =
                vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);

        VmaAllocatorCreateInfo ci{};
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        ci.physicalDevice = physical_device_;
        ci.device = device_;
        ci.instance = GaussianSplatVulkanContext::GetInstance().GetVkInstance();
        ci.pVulkanFunctions = &vk_fn;
        if (vmaCreateAllocator(&ci, &vma_) != VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vmaCreateAllocator failed");
            return false;
        }
        return true;
    }

    /// Create the command pool/buffer, submission fence and nearest sampler
    /// used by every pass. All handles are RAII, so a failure part-way through
    /// releases whatever was created before it.
    bool InitDeviceObjects() {
        auto& raii_dev =
                GaussianSplatVulkanContext::GetInstance().GetRaiiDevice();
        try {
            cmd_pool_ = raii_dev.createCommandPool({{}, queue_family_});
            NameObject(vk::ObjectType::eCommandPool,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkCommandPool>(*cmd_pool_)),
                       "gs.command_pool");

            cmd_ = std::move(
                    raii_dev.allocateCommandBuffers(
                                    {*cmd_pool_,
                                     vk::CommandBufferLevel::ePrimary, 1})
                            .front());
            NameObject(vk::ObjectType::eCommandBuffer,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkCommandBuffer>(*cmd_)),
                       "gs.command_buffer");

            fence_ = raii_dev.createFence({});

            nearest_sampler_ = raii_dev.createSampler(
                    {{},
                     vk::Filter::eNearest,
                     vk::Filter::eNearest,
                     vk::SamplerMipmapMode::eNearest,
                     vk::SamplerAddressMode::eClampToEdge,
                     vk::SamplerAddressMode::eClampToEdge,
                     vk::SamplerAddressMode::eClampToEdge});
            NameObject(vk::ObjectType::eSampler,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkSampler>(*nearest_sampler_)),
                       "gs.sampler.nearest");
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: device object creation failed: {}",
                    e.what());
            return false;
        }
        return true;
    }

    // --- Shader loading ---------------------------------------------------

    bool LoadShader(ComputeProgramId id, const std::string& shader_root) {
        const int i = static_cast<int>(id);
        const std::string name = kGsShaderNames[i];
        const std::string spv_path = shader_root + name + ".spv";
        std::vector<char> bytes;
        std::string err;
        if (!utility::filesystem::FReadToBuffer(spv_path, bytes, &err)) {
            utility::LogDebug("GaussianSplatVulkan: SPIR-V not found: {}",
                              spv_path);
            return false;
        }

        // Declare the exact bindings this shader uses; push descriptors
        // require a fully specified layout.
        const auto& bt = kShaderBindings[i];
        std::vector<vk::DescriptorSetLayoutBinding> layout_bindings(bt.count);
        std::uint64_t binding_mask = 0;
        for (std::uint32_t j = 0; j < bt.count; ++j) {
            layout_bindings[j] = {
                    bt.descs[j].binding,
                    static_cast<vk::DescriptorType>(bt.descs[j].type), 1,
                    vk::ShaderStageFlagBits::eCompute};
            if (bt.descs[j].binding < 64) {
                binding_mask |= std::uint64_t(1) << bt.descs[j].binding;
            }
        }

        // Every intermediate is a vk::raii handle, so an exception at any step
        // releases the objects created before it.
        auto& raii_dev =
                GaussianSplatVulkanContext::GetInstance().GetRaiiDevice();
        try {
            vk::raii::ShaderModule shader_module = raii_dev.createShaderModule(
                    {{},
                     bytes.size(),
                     reinterpret_cast<const std::uint32_t*>(bytes.data())});
            NameObject(vk::ObjectType::eShaderModule,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkShaderModule>(*shader_module)),
                       (name + ".shader").c_str());

            vk::raii::DescriptorSetLayout dset_layout =
                    raii_dev.createDescriptorSetLayout(
                            {vk::DescriptorSetLayoutCreateFlagBits::
                                     ePushDescriptorKHR,
                             layout_bindings});

            const vk::DescriptorSetLayout dsl_handle = *dset_layout;
            vk::raii::PipelineLayout pipeline_layout =
                    raii_dev.createPipelineLayout({{}, 1, &dsl_handle});
            NameObject(vk::ObjectType::ePipelineLayout,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkPipelineLayout>(*pipeline_layout)),
                       (name + ".layout").c_str());

            vk::raii::Pipeline pipeline = raii_dev.createComputePipeline(
                    nullptr, {{},
                              vk::PipelineShaderStageCreateInfo{
                                      {},
                                      vk::ShaderStageFlagBits::eCompute,
                                      *shader_module,
                                      "main"},
                              *pipeline_layout});
            NameObject(vk::ObjectType::ePipeline,
                       reinterpret_cast<std::uintptr_t>(
                               static_cast<VkPipeline>(*pipeline)),
                       name.c_str());

            // The shader module is released here; the pipeline owns its code.
            auto& p = pipelines_[i];
            p.dset_layout = std::move(dset_layout);
            p.layout = std::move(pipeline_layout);
            p.pipeline = std::move(pipeline);
            p.binding_mask = binding_mask;
            p.valid = true;
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: compute pipeline creation failed for "
                    "{}: {}",
                    name, e.what());
            return false;
        }
        utility::LogDebug("GaussianSplatVulkan: loaded {}", name);
        return true;
    }

    void NameObject(vk::ObjectType type,
                    std::uint64_t handle,
                    const char* name) {
        if (!debug_utils_enabled_ || !handle || name == nullptr) return;
        vk::DebugUtilsObjectNameInfoEXT info{};
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        try {
            (void)GaussianSplatVulkanContext::GetInstance()
                    .GetRaiiDevice()
                    .setDebugUtilsObjectNameEXT(info);
        } catch (const vk::SystemError&) {
            // Object naming is diagnostic metadata and must not affect
            // rendering.
        }
    }

    // --- Command buffer lifecycle -----------------------------------------

    void BeginCmdBuf() {
        if (cmd_active_) return;
        // If the geometry pass submitted a fence-only (fire-and-forget), drain
        // it now before resetting the command pool.  Also guards the case where
        // BeginCmdBuf is called again before a previous SubmitAndWait has had
        // a chance to drain (e.g. geometry pass on frame N+1 before composite
        // on frame N was submitted due to an early return).
        WaitForPendingSubmit();
        vk::Device(device_).resetCommandPool(*cmd_pool_, {});
        cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd_active_ = true;
        active_id_ = -1;
        pending_.clear();
    }

    // Submit the current command buffer and signal the fence, but do NOT wait.
    // Used by EndGeometryPass() so geometry overlaps with Filament rendering.
    // The fence is checked (and waited if necessary) in WaitForPendingSubmit(),
    // which BeginCmdBuf() and BeginCompositePass() call before starting work.
    void SubmitOnly() {
        if (!cmd_active_) return;
        cmd_.end();
        cmd_active_ = false;
        vk::Device(device_).resetFences({*fence_});
        vk::CommandBufferSubmitInfo cmd_info{*cmd_, 0};
        vk::SubmitInfo2 si{{}, {}, cmd_info, {}};
        vk::Queue(compute_queue_).submit2(si, *fence_);
        fence_submitted_ = true;
        active_id_ = -1;
        pending_.clear();
    }

    // Wait for the most recently submitted fence if one is outstanding.
    // No-op when no submission is pending (fence_submitted_ == false).
    void WaitForPendingSubmit() {
        if (!fence_submitted_) return;
        (void)vk::Device(device_).waitForFences({*fence_}, true, UINT64_MAX);
        vk::Device(device_).resetFences({*fence_});
        fence_submitted_ = false;
    }

    // Submit and immediately wait (used by EndCompositePass and FinishGpuWork).
    void SubmitAndWait() {
        SubmitOnly();
        WaitForPendingSubmit();
    }

    // --- Descriptor helpers -----------------------------------------------

    void PushBufferWrite(std::uint32_t binding,
                         std::uintptr_t buf,
                         VkDescriptorType type,
                         std::size_t offset,
                         VkDeviceSize range) {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = type;
        pw.buf = {it->second.buffer, offset, range};
        pending_.push_back(pw);
    }

    void PushImageWrite(std::uint32_t binding,
                        std::uintptr_t tex,
                        VkDescriptorType type,
                        VkImageLayout layout,
                        VkSampler sampler) {
        const VkImageView view = ResolveImageView(tex, layout);
        if (view == VK_NULL_HANDLE) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = type;
        pw.img = {sampler, view, layout};
        pending_.push_back(pw);
    }

    void FlushPendingBindings() {
        if (active_id_ < 0 || pending_.empty()) return;
        auto& p = pipelines_[active_id_];
        // Build push descriptor writes, filtering to bindings in the layout.
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(pending_.size());
        for (auto& pw : pending_) {
            if (pw.binding < 64 &&
                !(p.binding_mask & (uint64_t(1) << pw.binding)))
                continue;  // not in this pipeline's layout
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstBinding = pw.binding;
            w.descriptorCount = 1;
            w.descriptorType = pw.type;
            switch (pw.type) {
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    w.pBufferInfo = &pw.buf;
                    break;
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    w.pImageInfo = &pw.img;
                    break;
                default:
                    break;
            }
            writes.push_back(w);
        }
        if (!writes.empty()) {
            // vkCmdPushDescriptorSetKHR is a KHR extension function; call it
            // through the default dispatcher which was loaded with the device.
            VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPushDescriptorSetKHR(
                    static_cast<VkCommandBuffer>(*cmd_),
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    static_cast<VkPipelineLayout>(*p.layout), 0,
                    static_cast<std::uint32_t>(writes.size()), writes.data());
        }
        pending_.clear();
    }

    // --- Image / view helpers ---------------------------------------------

    void ReleaseImportedImages() {
        for (auto& item : textures_) {
            auto& entry = item.second;
            if (!entry.imported || !entry.used_in_composite) continue;
            if (entry.format == VK_FORMAT_D32_SFLOAT) {
                if (entry.current_layout ==
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    TransitionImageLayout(
                            entry.image, entry.format, entry.current_layout,
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    entry.current_layout =
                            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                }
                continue;
            }
            // Make composite writes visible to Filament's sampling of the
            // shared colour image in the next frame.
            MemoryBarrierInLayout(entry.image, VK_IMAGE_LAYOUT_GENERAL,
                                  vk::PipelineStageFlagBits2::eComputeShader,
                                  vk::AccessFlagBits2::eShaderWrite,
                                  vk::PipelineStageFlagBits2::eFragmentShader,
                                  vk::AccessFlagBits2::eShaderSampledRead);
        }
    }

    /// Resolve a texture handle to a VkImageView. Also ensures the image
    /// layout via a barrier.
    VkImageView ResolveImageView(std::uintptr_t handle,
                                 VkImageLayout needed_layout) {
        auto it = textures_.find(handle);
        if (it == textures_.end()) return VK_NULL_HANDLE;
        auto& e = it->second;
        const bool first_imported_use = e.imported && !e.used_in_composite;
        e.used_in_composite = e.imported;
        if (e.current_layout != needed_layout) {
            TransitionImageLayout(e.image, e.format, e.current_layout,
                                  needed_layout);
            e.current_layout = needed_layout;
        } else if (first_imported_use &&
                   needed_layout == VK_IMAGE_LAYOUT_GENERAL) {
            // Filament's colour-attachment writes must be visible before the
            // composite pass reads and blends over them.
            MemoryBarrierInLayout(
                    e.image, VK_IMAGE_LAYOUT_GENERAL,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentWrite,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead |
                            vk::AccessFlagBits2::eShaderWrite);
        }
        return static_cast<VkImageView>(*e.view);
    }

    // Layout transition using VK_QUEUE_FAMILY_IGNORED: every image (including
    // those shared with Filament) lives on this device's graphics/compute
    // family, so ownership never leaves this queue family.
    void TransitionImageLayout(VkImage image,
                               VkFormat format,
                               VkImageLayout old_layout,
                               VkImageLayout new_layout) {
        const ImageSyncScope src = SrcScopeForLayout(old_layout);
        const ImageSyncScope dst = DstScopeForLayout(new_layout);
        const vk::ImageMemoryBarrier2 barrier{
                src.stage,
                src.access,
                dst.stage,
                dst.access,
                static_cast<vk::ImageLayout>(old_layout),
                static_cast<vk::ImageLayout>(new_layout),
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                vk::Image(image),
                FullSubresource(format),
        };
        cmd_.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
    }

    /// Execution/memory barrier that keeps the layout unchanged. Used to order
    /// Filament's attachment writes against GS compute access to the same
    /// shared image, which stays in GENERAL for both.
    void MemoryBarrierInLayout(VkImage image,
                               VkImageLayout layout,
                               vk::PipelineStageFlags2 src_stage,
                               vk::AccessFlags2 src_access,
                               vk::PipelineStageFlags2 dst_stage,
                               vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{
                src_stage,
                src_access,
                dst_stage,
                dst_access,
                static_cast<vk::ImageLayout>(layout),
                static_cast<vk::ImageLayout>(layout),
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                vk::Image(image),
                FullSubresource(VK_FORMAT_UNDEFINED),
        };
        cmd_.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, barrier});
    }

    vk::raii::ImageView CreateImageView(VkImage image,
                                        VkFormat format,
                                        VkImageAspectFlags aspect) {
        vk::ImageViewCreateInfo vci{
                {},
                vk::Image(image),
                vk::ImageViewType::e2D,
                static_cast<vk::Format>(format),
                {},
                vk::ImageSubresourceRange{
                        static_cast<vk::ImageAspectFlags>(aspect), 0, 1, 0, 1},
        };
        try {
            return GaussianSplatVulkanContext::GetInstance()
                    .GetRaiiDevice()
                    .createImageView(vci);
        } catch (const vk::SystemError&) {
            return vk::raii::ImageView{nullptr};
        }
    }

    // --- Buffer allocation ------------------------------------------------

    std::uintptr_t AllocBuf(std::size_t size, bool priv, const char* label) {
        if (size == 0) return 0;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        if (priv) {
            aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        } else {
            aci.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VmaAllocation alloc = VK_NULL_HANDLE;
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(vma_, &bci, &aci, &buf, &alloc, &info) !=
            VK_SUCCESS)
            return 0;

        uintptr_t handle = next_handle_++;
        BufEntry& e = buffers_[handle];
        e.alloc = alloc;
        e.buffer = buf;
        e.size = size;
        e.mapped = priv ? nullptr : info.pMappedData;
        e.is_private = priv;
        NameObject(vk::ObjectType::eBuffer,
                   reinterpret_cast<std::uintptr_t>(buf), label);
        return handle;
    }

    /// Reuse the existing allocation when the size is unchanged, otherwise
    /// destroy and allocate a fresh buffer with a new handle.
    std::uintptr_t ReallocBuf(std::uintptr_t buf,
                              std::size_t new_size,
                              bool priv,
                              const char* label) {
        auto it = buffers_.find(buf);
        if (it != buffers_.end()) {
            if (it->second.size == new_size) return buf;
            DestroyBuffer(buf);
        }
        return AllocBuf(new_size, priv, label);
    }

    // --- Texture/image allocation -----------------------------------------

    std::uintptr_t AllocTex(std::uint32_t w,
                            std::uint32_t h,
                            VkFormat format,
                            VkImageUsageFlags usage,
                            VkImageAspectFlags aspect,
                            const char* label = nullptr) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImage image = VK_NULL_HANDLE;
        if (vmaCreateImage(vma_, &ici, &aci, &image, &alloc, nullptr) !=
            VK_SUCCESS)
            return 0;
        vk::raii::ImageView view = CreateImageView(image, format, aspect);
        if (static_cast<VkImageView>(*view) == VK_NULL_HANDLE) {
            vmaDestroyImage(vma_, image, alloc);
            return 0;
        }
        uintptr_t handle = next_handle_++;
        TexEntry& e = textures_[handle];
        e.alloc = alloc;
        e.image = image;
        e.view = std::move(view);
        e.format = format;
        e.width = w;
        e.height = h;
        e.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        NameObject(vk::ObjectType::eImage,
                   reinterpret_cast<std::uintptr_t>(image), label);
        NameObject(vk::ObjectType::eImageView,
                   reinterpret_cast<std::uintptr_t>(
                           static_cast<VkImageView>(*e.view)),
                   label);
        return handle;
    }

    /// True when \p tex already has the requested size; otherwise destroys it
    /// so the caller can allocate a replacement.
    bool KeepTextureOfSize(std::uintptr_t tex,
                           std::uint32_t w,
                           std::uint32_t h) {
        if (tex == 0) return false;
        auto it = textures_.find(tex);
        if (it != textures_.end() && it->second.width == w &&
            it->second.height == h) {
            return true;
        }
        DestroyTexture(tex);
        return false;
    }

    // --- Download helpers -------------------------------------------------

    template <typename T>
    bool DownloadTexBytes(std::uintptr_t tex,
                          std::uint32_t w,
                          std::uint32_t h,
                          std::size_t bytes_per_elem,
                          std::vector<T>& out) {
        auto it = textures_.find(tex);
        if (it == textures_.end() || w == 0 || h == 0) return false;
        const std::size_t row_size =
                static_cast<std::size_t>(w) * bytes_per_elem;
        const std::size_t total = row_size * h;
        out.resize(total / sizeof(T));

        // Create a staging readback buffer.
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = total;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(vma_, &bci, &aci, &staging, &alloc, &info) !=
            VK_SUCCESS)
            return false;

        // Record copy: transition image to TRANSFER_SRC, copy, transition to
        // GENERAL (a valid layout for the next compute or transfer use).
        BeginCmdBuf();
        auto& e = it->second;
        const auto aspect = (e.format == VK_FORMAT_D32_SFLOAT)
                                    ? vk::ImageAspectFlagBits::eDepth
                                    : vk::ImageAspectFlagBits::eColor;
        TransitionImageLayout(e.image, e.format, e.current_layout,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vk::BufferImageCopy region{
                0, 0, 0, {aspect, 0, 0, 1}, {0, 0, 0}, {w, h, 1},
        };
        cmd_.copyImageToBuffer(vk::Image(e.image),
                               vk::ImageLayout::eTransferSrcOptimal,
                               vk::Buffer(staging), region);
        TransitionImageLayout(e.image, e.format,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_IMAGE_LAYOUT_GENERAL);
        e.current_layout = VK_IMAGE_LAYOUT_GENERAL;
        SubmitAndWait();

        vmaInvalidateAllocation(vma_, alloc, 0, total);
        std::memcpy(out.data(), info.pMappedData, total);
        vmaDestroyBuffer(vma_, staging, alloc);
        return true;
    }

    // --- Shutdown ---------------------------------------------------------

    void Shutdown() {
        if (device_ == VK_NULL_HANDLE) return;
        vk::Device(device_).waitIdle();

        // Destroy image views (RAII) before their VMA images.
        for (auto& [h, e] : textures_) {
            e.view = vk::raii::ImageView{nullptr};  // vkDestroyImageView
            if (e.alloc != VK_NULL_HANDLE)
                vmaDestroyImage(vma_, e.image, e.alloc);
        }
        textures_.clear();

        // Destroy VMA buffers.
        for (auto& [h, e] : buffers_) {
            vmaDestroyBuffer(vma_, e.buffer, e.alloc);
        }
        buffers_.clear();

        // Reset pipeline RAII handles: destroys pipelines, layouts, dset
        // layouts.
        for (auto& p : pipelines_) {
            p.pipeline = vk::raii::Pipeline{nullptr};
            p.layout = vk::raii::PipelineLayout{nullptr};
            p.dset_layout = vk::raii::DescriptorSetLayout{nullptr};
            p.valid = false;
        }

        // Reset remaining RAII handles in reverse init order.
        nearest_sampler_ = vk::raii::Sampler{nullptr};
        fence_ = vk::raii::Fence{nullptr};
        cmd_ = vk::raii::CommandBuffer{nullptr};
        cmd_pool_ = vk::raii::CommandPool{nullptr};
        if (vma_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(vma_);
            vma_ = VK_NULL_HANDLE;
        }
        device_ = VK_NULL_HANDLE;
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void RegisterVkImageInComputeContext(GaussianSplatGpuContext& ctx,
                                     std::uintptr_t vk_image_opaque,
                                     std::uint32_t vk_format_opaque,
                                     std::uint32_t width,
                                     std::uint32_t height) {
    auto* vk_ctx = dynamic_cast<GaussianSplatGpuContextVulkan*>(&ctx);
    if (!vk_ctx) return;
    vk_ctx->RegisterVkImage(reinterpret_cast<VkImage>(vk_image_opaque),
                            static_cast<VkFormat>(vk_format_opaque), width,
                            height);
}

void UnregisterVkImageFromComputeContext(GaussianSplatGpuContext& ctx,
                                         std::uintptr_t vk_image_opaque) {
    auto* vk_ctx = dynamic_cast<GaussianSplatGpuContextVulkan*>(&ctx);
    if (!vk_ctx) return;
    vk_ctx->UnregisterVkImage(reinterpret_cast<VkImage>(vk_image_opaque));
}

std::unique_ptr<GaussianSplatGpuContext> CreateComputeGpuContextVulkan() {
    if (!GaussianSplatVulkanContext::GetInstance().IsValid()) {
        utility::LogWarning(
                "GaussianSplatVulkan: Vulkan context not initialized; "
                "compute context not created");
        return nullptr;
    }
    return std::make_unique<GaussianSplatGpuContextVulkan>();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
