// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

// BlueVK: dynamic Vulkan loader bundled with Filament.
#include "bluevk/BlueVK.h"
using namespace bluevk;

// VMA: header-only allocator (implementation in GaussianSplatVulkanInteropContext.cpp).
#include "vk_mem_alloc.h"

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"

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
    VkImageLayout image_layout;  // Only for STORAGE_IMAGE / COMBINED_IMAGE_SAMPLER
};

// Binding tables derived from SPIR-V analysis (spirv-dis) of each compiled
// shader.  image_layout is VK_IMAGE_LAYOUT_UNDEFINED for buffer bindings.
//
// NOTE: out_color in composite and gs_depth in depth_merge use binding 16
//       (not 0) to avoid conflicting with the GaussianViewParams UBO at B0.

static constexpr ShaderBindingDesc kBindingsProject[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsPrefixSum[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsScatter[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsComposite[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_LAYOUT_GENERAL},
};
static constexpr ShaderBindingDesc kBindingsRadixKeygen[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
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
static constexpr ShaderBindingDesc kBindingsRadixPayload[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
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
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
};
static constexpr ShaderBindingDesc kBindingsOneSweepGlobalHist[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};
static constexpr ShaderBindingDesc kBindingsOneSweepDigitPass[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
        {14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_IMAGE_LAYOUT_UNDEFINED},
};

// Table indexed by ComputeProgramId: binding descriptor + count.
struct ShaderBindingTable {
    const ShaderBindingDesc* descs;
    std::uint32_t count;
};

static constexpr ShaderBindingTable kShaderBindings[] = {
        {kBindingsProject, std::size(kBindingsProject)},
        {kBindingsPrefixSum, std::size(kBindingsPrefixSum)},
        {kBindingsScatter, std::size(kBindingsScatter)},
        {kBindingsComposite, std::size(kBindingsComposite)},
        {kBindingsRadixKeygen, std::size(kBindingsRadixKeygen)},
        {kBindingsRadixHistograms, std::size(kBindingsRadixHistograms)},
        {kBindingsRadixScatter, std::size(kBindingsRadixScatter)},
        {kBindingsRadixPayload, std::size(kBindingsRadixPayload)},
        {kBindingsDispatchArgs, std::size(kBindingsDispatchArgs)},
        {kBindingsDepthMerge, std::size(kBindingsDepthMerge)},
        {kBindingsOneSweepGlobalHist, std::size(kBindingsOneSweepGlobalHist)},
        {kBindingsOneSweepDigitPass, std::size(kBindingsOneSweepDigitPass)},
};
static_assert(std::size(kShaderBindings) ==
                      static_cast<std::size_t>(ComputeProgramId::kCount),
              "kShaderBindings must match ComputeProgramId::kCount");

// ---------------------------------------------------------------------------
// Vulkan compute backend class
// ---------------------------------------------------------------------------
class GaussianSplatGpuContextVulkan final : public GaussianSplatGpuContext {
public:
    explicit GaussianSplatGpuContextVulkan(bool use_subgroups)
        : use_subgroups_(use_subgroups) {}

    ~GaussianSplatGpuContextVulkan() override { Shutdown(); }

    // --- Program management -----------------------------------------------

    bool EnsureProgramsLoaded() override {
        if (programs_loaded_) return programs_valid_;
        programs_loaded_ = true;
        programs_valid_ = false;

        auto& interop = GaussianSplatVulkanInteropContext::GetInstance();
        if (!interop.IsValid()) {
            utility::LogWarning(
                    "GaussianSplatVulkan: interop context not initialized");
            return false;
        }

        device_ = interop.GetDevice();
        physical_device_ = interop.GetPhysicalDevice();
        compute_queue_ = interop.GetComputeQueue();
        queue_family_ = interop.GetComputeQueueFamily();

        if (!InitVma()) return false;
        if (!InitCommandPool()) return false;
        if (!InitFence()) return false;
        if (!InitSampler()) return false;

        // Vulkan-native SPIR-V lives in the same resource subdirectory as
        // the GL backend; open3d_add_compute_shaders compiles all shaders
        // with -V --target-env vulkan1.3, so the same .spv files work for
        // both VkCreateShaderModule and GL_ARB_gl_spirv.
        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_splat/";

        // Phase 1: required shaders [0, kGsFirstOneSweepProgram)
        for (int i = 0; i < kGsFirstOneSweepProgram; ++i) {
            if (!LoadShader(static_cast<ComputeProgramId>(i), shader_root)) {
                utility::LogWarning(
                        "GaussianSplatVulkan: failed to load shader {}",
                        kGsShaderNames[i]);
                return false;
            }
        }
        programs_valid_ = true;

        // Phase 2: optional OneSweep programs
        bool onesweep_ok = true;
        for (int i = kGsFirstOneSweepProgram;
             i < static_cast<int>(ComputeProgramId::kCount); ++i) {
            if (!LoadShader(static_cast<ComputeProgramId>(i), shader_root)) {
                onesweep_ok = false;
                break;
            }
        }
        onesweep_valid_ = onesweep_ok;
        utility::LogDebug("GaussianSplatVulkan: programs loaded (onesweep={})",
                          onesweep_ok ? "yes" : "no");
        return true;
    }

    bool AreOneSweepProgramsLoaded() const override { return onesweep_valid_; }

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
        if (e.mapped) vmaUnmapMemory(vma_, e.alloc);
        vmaDestroyBuffer(vma_, e.buffer, e.alloc);
        buffers_.erase(it);
    }

    std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                std::size_t new_size,
                                const char* label = nullptr) override {
        if (buf == 0) return CreateBuffer(new_size, label);
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return CreateBuffer(new_size, label);
        if (it->second.size == new_size) return buf;
        DestroyBuffer(buf);
        return CreateBuffer(new_size, label);
    }

    std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                       std::size_t new_size,
                                       const char* label = nullptr) override {
        if (new_size == 0) { DestroyBuffer(buf); return 0; }
        if (buf == 0) return CreatePrivateBuffer(new_size, label);
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return CreatePrivateBuffer(new_size, label);
        if (it->second.size == new_size) return buf;
        DestroyBuffer(buf);
        return CreatePrivateBuffer(new_size, label);
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
        vkCmdFillBuffer(cmd_, it->second.buffer, 0, VK_WHOLE_SIZE, 0u);
    }

    // --- Bindings ---------------------------------------------------------

    void BindSSBO(std::uint32_t binding, std::uintptr_t buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pw.buf = {it->second.buffer, 0, VK_WHOLE_SIZE};
        pending_.push_back(pw);
    }

    void BindUBO(std::uint32_t binding, std::uintptr_t buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pw.buf = {it->second.buffer, 0, it->second.size};
        pending_.push_back(pw);
    }

    void BindUBORange(std::uint32_t binding,
                      std::uintptr_t buf,
                      std::size_t offset,
                      std::size_t range_size) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pw.buf = {it->second.buffer, offset, range_size};
        pending_.push_back(pw);
    }

    void BindImage(std::uint32_t binding,
                   std::uintptr_t tex,
                   std::uint32_t /*width*/,
                   std::uint32_t /*height*/,
                   ImageFormat /*fmt*/) override {
        auto view = ResolveImageView(tex, VK_IMAGE_LAYOUT_GENERAL);
        if (view == VK_NULL_HANDLE) return;
        PendingWrite pw{};
        pw.binding = binding;
        pw.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pw.img = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        pending_.push_back(pw);
    }

    void BindSamplerTexture(std::uint32_t unit,
                            std::uintptr_t tex,
                            std::uint32_t /*width*/,
                            std::uint32_t /*height*/) override {
        auto view = ResolveImageView(
                tex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (view == VK_NULL_HANDLE) return;
        PendingWrite pw{};
        pw.binding = unit;
        pw.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pw.img = {nearest_sampler_, view,
                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        pending_.push_back(pw);
    }

    // --- Dispatch ---------------------------------------------------------

    void UseProgram(ComputeProgramId id) override {
        int i = static_cast<int>(id);
        if (i < 0 || i >= static_cast<int>(ComputeProgramId::kCount)) return;
        auto& p = pipelines_[i];
        if (!p.valid) return;
        active_id_ = i;
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        pending_.clear();
    }

    void Dispatch(std::uint32_t gx,
                  std::uint32_t gy,
                  std::uint32_t gz) override {
        FlushPendingBindings();
        vkCmdDispatch(cmd_, gx, gy, gz);
    }

    void DispatchIndirect(std::uintptr_t indirect_buf,
                          std::size_t byte_offset) override {
        auto it = buffers_.find(indirect_buf);
        if (it == buffers_.end()) return;
        FlushPendingBindings();
        vkCmdDispatchIndirect(cmd_, it->second.buffer,
                              static_cast<VkDeviceSize>(byte_offset));
    }

    void FullBarrier() override {
        // Memory barrier covering all compute accesses:
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                           VK_ACCESS_2_MEMORY_WRITE_BIT |
                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd_, &di);
    }

    // --- Textures / images ------------------------------------------------

    std::uintptr_t CreateTexture2DR32F(std::uint32_t w,
                                       std::uint32_t h,
                                       const char* /*label*/) override {
        return AllocTex(w, h, VK_FORMAT_R32_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void DestroyTexture(std::uintptr_t tex) override {
        auto it = textures_.find(tex);
        if (it == textures_.end()) return;
        auto& e = it->second;
        if (e.view != VK_NULL_HANDLE)
            vkDestroyImageView(device_, e.view, nullptr);
        if (!e.is_shared) vmaDestroyImage(vma_, e.image, e.alloc);
        textures_.erase(it);
    }

    std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                       std::uint32_t w,
                                       std::uint32_t h,
                                       const char* label) override {
        if (tex == 0) return CreateTexture2DR32F(w, h, label);
        auto it = textures_.find(tex);
        if (it != textures_.end() && it->second.width == w &&
            it->second.height == h)
            return tex;
        DestroyTexture(tex);
        return CreateTexture2DR32F(w, h, label);
    }

    std::uintptr_t ResizeTexture2DR16UI(std::uintptr_t tex,
                                        std::uint32_t w,
                                        std::uint32_t h,
                                        const char* /*label*/) override {
        if (tex != 0) {
            auto it = textures_.find(tex);
            if (it != textures_.end() && it->second.width == w &&
                it->second.height == h)
                return tex;
            DestroyTexture(tex);
        }
        return AllocTex(w, h, VK_FORMAT_R16_UINT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
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

    // --- Frame boundary ---------------------------------------------------

    void BeginGeometryPass() override { BeginCmdBuf(); }
    void EndGeometryPass() override { SubmitAndWait(); }
    void BeginCompositePass() override { BeginCmdBuf(); }
    void EndCompositePass() override { SubmitAndWait(); }

    void FinishGpuWork() override {
        if (!cmd_active_) return;
        SubmitAndWait();
    }

    void PushDebugGroup(const char* label) override {
        if (!vkCmdBeginDebugUtilsLabelEXT) return;
        VkDebugUtilsLabelEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pLabelName = label;
        vkCmdBeginDebugUtilsLabelEXT(cmd_, &info);
    }
    void PopDebugGroup() override {
        if (vkCmdEndDebugUtilsLabelEXT) vkCmdEndDebugUtilsLabelEXT(cmd_);
    }

    // --- Shared-image registration (called by GaussianSplatVulkanBackend) -

    void RegisterSharedImage(std::uint32_t gl_name,
                             VkImage image,
                             VkFormat format,
                             std::uint32_t w,
                             std::uint32_t h) {
        // If already registered, clean up old data first.
        UnregisterSharedImage(gl_name);
        VkImageAspectFlags aspect = (format == VK_FORMAT_D32_SFLOAT)
                                            ? VK_IMAGE_ASPECT_DEPTH_BIT
                                            : VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageView view = CreateImageView(image, format, aspect);
        if (view == VK_NULL_HANDLE) {
            utility::LogWarning(
                    "GaussianSplatVulkan: failed to create view for shared "
                    "gl_name={}",
                    gl_name);
            return;
        }
        TexEntry e{};
        e.image = image;
        e.view = view;
        e.alloc = VK_NULL_HANDLE;  // owned by VulkanInteropContext
        e.format = format;
        e.width = w;
        e.height = h;
        e.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        e.is_shared = true;
        uintptr_t handle = next_handle_++;
        textures_[handle] = e;
        gl_to_handle_[gl_name] = handle;
    }

    void UnregisterSharedImage(std::uint32_t gl_name) {
        auto it = gl_to_handle_.find(gl_name);
        if (it == gl_to_handle_.end()) return;
        uintptr_t h = it->second;
        auto te = textures_.find(h);
        if (te != textures_.end()) {
            if (te->second.view != VK_NULL_HANDLE)
                vkDestroyImageView(device_, te->second.view, nullptr);
            textures_.erase(te);
        }
        gl_to_handle_.erase(it);
    }

private:
    // --- Internal state ---------------------------------------------------
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VmaAllocator vma_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkSampler nearest_sampler_ = VK_NULL_HANDLE;
    bool cmd_active_ = false;

    struct Pipeline {
        VkDescriptorSetLayout dset_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool valid = false;
        // Bitmask of valid bindings so we can filter writes.
        std::uint64_t binding_mask = 0;  // bit i set ↔ binding i exists
    };
    Pipeline pipelines_[static_cast<int>(ComputeProgramId::kCount)] = {};
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
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool is_shared = false;
    };
    std::unordered_map<uintptr_t, TexEntry> textures_;
    std::unordered_map<std::uint32_t, uintptr_t> gl_to_handle_;
    std::uint64_t next_handle_ = 1;

    struct PendingWrite {
        std::uint32_t binding = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorBufferInfo buf{};
        VkDescriptorImageInfo img{};
    };
    std::vector<PendingWrite> pending_;

    bool use_subgroups_;
    bool programs_loaded_ = false;
    bool programs_valid_ = false;
    bool onesweep_valid_ = false;

    // --- Init helpers -----------------------------------------------------

    bool InitVma() {
        VmaVulkanFunctions vk_fn{};
        vk_fn.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vk_fn.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo ci{};
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        ci.physicalDevice = physical_device_;
        ci.device = device_;
        ci.instance = vkGetPhysicalDeviceProperties != nullptr
                              ? [&]() -> VkInstance {
            // Retrieve the instance from the BlueVK loader via the
            // physical device's parent. We can't get it directly, so
            // pass VK_NULL_HANDLE and let VMA use the function pointers.
            return VK_NULL_HANDLE;
        }()
                              : VK_NULL_HANDLE;
        ci.pVulkanFunctions = &vk_fn;
        // Note: instance is optional if all needed pfns are provided via
        // pVulkanFunctions; VMA works with global BlueVK dispatch.
        if (vmaCreateAllocator(&ci, &vma_) != VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vmaCreateAllocator failed");
            return false;
        }
        return true;
    }

    bool InitCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.queueFamilyIndex = queue_family_;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_) !=
            VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateCommandPool failed");
            return false;
        }
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmd_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &ai, &cmd_) != VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkAllocateCommandBuffers failed");
            return false;
        }
        return true;
    }

    bool InitFence() {
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        return vkCreateFence(device_, &fi, nullptr, &fence_) == VK_SUCCESS;
    }

    bool InitSampler() {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        return vkCreateSampler(device_, &si, nullptr, &nearest_sampler_) ==
               VK_SUCCESS;
    }

    // --- Shader loading ---------------------------------------------------

    bool LoadShader(ComputeProgramId id, const std::string& shader_root) {
        const int i = static_cast<int>(id);
        std::string name = kGsShaderNames[i];

        // Resolve subgroup variant name
        constexpr std::string_view kSubgroupSuffix = "_subgroup";
        const bool is_subgroup =
                name.size() > kSubgroupSuffix.size() &&
                name.compare(name.size() - kSubgroupSuffix.size(),
                             kSubgroupSuffix.size(), kSubgroupSuffix) == 0;

        // For non-subgroup names, optionally append _subgroup when enabled.
        // For already-subgroup names (_subgroup suffix), use as-is.
        std::string file_name;
        if (!is_subgroup && use_subgroups_) {
            // Check if a subgroup variant exists; fall back if not.
            const std::string candidate =
                    shader_root + name + "_subgroup.spv";
            std::vector<char> tmp;
            std::string err;
            if (utility::filesystem::FReadToBuffer(candidate, tmp, &err)) {
                file_name = name + "_subgroup";
            } else {
                file_name = name;
            }
        } else if (is_subgroup && !use_subgroups_) {
            // Strip _subgroup suffix
            file_name = name.substr(0, name.size() - kSubgroupSuffix.size());
        } else {
            file_name = name;
        }

        const std::string spv_path = shader_root + file_name + ".spv";
        std::vector<char> bytes;
        std::string err;
        if (!utility::filesystem::FReadToBuffer(spv_path, bytes, &err)) {
            utility::LogDebug("GaussianSplatVulkan: SPIR-V not found: {}",
                              spv_path);
            return false;
        }

        // Create shader module
        VkShaderModuleCreateInfo smi{};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = bytes.size();
        smi.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());
        VkShaderModule shader_module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &smi, nullptr, &shader_module) !=
            VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateShaderModule failed for {}",
                    file_name);
            return false;
        }

        // Build descriptor set layout for this pipeline
        const auto& bt = kShaderBindings[i];
        std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bt.count);
        std::uint64_t binding_mask = 0;
        for (std::uint32_t j = 0; j < bt.count; ++j) {
            auto& lb = layout_bindings[j];
            lb.binding = bt.descs[j].binding;
            lb.descriptorType = bt.descs[j].type;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            lb.pImmutableSamplers = nullptr;
            if (lb.binding < 64) binding_mask |= (uint64_t(1) << lb.binding);
        }

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        // Enable push descriptors for this layout.
        dslci.flags =
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dslci.bindingCount = bt.count;
        dslci.pBindings = layout_bindings.data();
        VkDescriptorSetLayout dset_layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr,
                                        &dset_layout) != VK_SUCCESS) {
            vkDestroyShaderModule(device_, shader_module, nullptr);
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateDescriptorSetLayout failed "
                    "for {}",
                    file_name);
            return false;
        }

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dset_layout;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(device_, &plci, nullptr,
                                   &pipeline_layout) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device_, dset_layout, nullptr);
            vkDestroyShaderModule(device_, shader_module, nullptr);
            return false;
        }

        VkComputePipelineCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pci.stage.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader_module;
        pci.stage.pName = "main";
        pci.layout = pipeline_layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult res = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1,
                                                &pci, nullptr, &pipeline);
        // Shader module can be destroyed after pipeline creation.
        vkDestroyShaderModule(device_, shader_module, nullptr);
        if (res != VK_SUCCESS) {
            vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
            vkDestroyDescriptorSetLayout(device_, dset_layout, nullptr);
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateComputePipelines failed "
                    "for {}",
                    file_name);
            return false;
        }

        auto& p = pipelines_[i];
        p.dset_layout = dset_layout;
        p.layout = pipeline_layout;
        p.pipeline = pipeline;
        p.binding_mask = binding_mask;
        p.valid = true;
        utility::LogDebug("GaussianSplatVulkan: loaded {}", file_name);
        return true;
    }

    // --- Command buffer lifecycle -----------------------------------------

    void BeginCmdBuf() {
        if (cmd_active_) return;
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &bi);
        cmd_active_ = true;
        active_id_ = -1;
        pending_.clear();
    }

    void SubmitAndWait() {
        if (!cmd_active_) return;
        vkEndCommandBuffer(cmd_);
        cmd_active_ = false;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        vkResetFences(device_, 1, &fence_);
        vkQueueSubmit(compute_queue_, 1, &si, fence_);
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
        active_id_ = -1;
        pending_.clear();
    }

    // --- Descriptor helpers -----------------------------------------------

    void FlushPendingBindings() {
        if (active_id_ < 0 || pending_.empty()) return;
        auto& p = pipelines_[active_id_];
        // Build push descriptor writes, filtering to bindings in the layout.
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(pending_.size());
        for (auto& pw : pending_) {
            if (pw.binding < 64 && !(p.binding_mask & (uint64_t(1) << pw.binding)))
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
            vkCmdPushDescriptorSetKHR(
                    cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                    static_cast<std::uint32_t>(writes.size()),
                    writes.data());
        }
        pending_.clear();
    }

    // --- Image / view helpers ---------------------------------------------

    /// Resolve a texture handle to a VkImageView: shared images are looked
    /// up by GL name (stored in gl_to_handle_); internal textures use their
    /// direct handle. Also ensures the image layout via a barrier.
    VkImageView ResolveImageView(std::uintptr_t handle,
                                 VkImageLayout needed_layout) {
        // First check if this handle directly maps to a known texture.
        auto it = textures_.find(handle);
        if (it == textures_.end()) {
            // Try interpreting the handle as a GL texture name.
            auto git = gl_to_handle_.find(
                    static_cast<std::uint32_t>(handle));
            if (git == gl_to_handle_.end()) return VK_NULL_HANDLE;
            it = textures_.find(git->second);
            if (it == textures_.end()) return VK_NULL_HANDLE;
        }
        auto& e = it->second;
        if (e.current_layout != needed_layout) {
            TransitionImageLayout(e.image, e.format, e.current_layout,
                                  needed_layout);
            e.current_layout = needed_layout;
        }
        return e.view;
    }

    void TransitionImageLayout(VkImage image,
                               VkFormat format,
                               VkImageLayout old_layout,
                               VkImageLayout new_layout) {
        const bool is_depth = (format == VK_FORMAT_D32_SFLOAT);
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT |
                          VK_ACCESS_2_MEMORY_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask =
                (new_layout == VK_IMAGE_LAYOUT_GENERAL)
                        ? (VK_ACCESS_2_SHADER_WRITE_BIT |
                           VK_ACCESS_2_SHADER_READ_BIT)
                        : VK_ACCESS_2_SHADER_READ_BIT;
        b.oldLayout = old_layout;
        b.newLayout = new_layout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange.aspectMask =
                is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT
                         : VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd_, &di);
    }

    VkImageView CreateImageView(VkImage image,
                                VkFormat format,
                                VkImageAspectFlags aspect) {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange.aspectMask = aspect;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(device_, &vci, nullptr, &view) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return view;
    }

    // --- Buffer allocation ------------------------------------------------

    std::uintptr_t AllocBuf(std::size_t size, bool priv, const char* /*lbl*/) {
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
        return handle;
    }

    // --- Texture/image allocation -----------------------------------------

    std::uintptr_t AllocTex(std::uint32_t w,
                            std::uint32_t h,
                            VkFormat format,
                            VkImageUsageFlags usage,
                            VkImageAspectFlags aspect) {
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
        VkImageView view = CreateImageView(image, format, aspect);
        if (view == VK_NULL_HANDLE) {
            vmaDestroyImage(vma_, image, alloc);
            return 0;
        }
        uintptr_t handle = next_handle_++;
        TexEntry& e = textures_[handle];
        e.alloc = alloc;
        e.image = image;
        e.view = view;
        e.format = format;
        e.width = w;
        e.height = h;
        e.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        e.is_shared = false;
        return handle;
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
        out.resize(static_cast<std::size_t>(w) * h);

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

        // Record copy: transition image to TRANSFER_SRC, copy, transition
        // back.
        BeginCmdBuf();
        auto& e = it->second;
        TransitionImageLayout(e.image, e.format, e.current_layout,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd_, e.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging,
                               1, &region);
        // Transition back to previous layout (or GENERAL).
        TransitionImageLayout(e.image, e.format,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              e.current_layout == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? VK_IMAGE_LAYOUT_GENERAL
                                      : e.current_layout);
        SubmitAndWait();

        std::memcpy(out.data(), info.pMappedData, total);
        vmaDestroyBuffer(vma_, staging, alloc);

        // Flip bottom-up → top-down (matches GL / Filament readPixels).
        for (std::uint32_t row = 0; row < h / 2; ++row) {
            std::swap_ranges(out.begin() + row * w,
                             out.begin() + row * w + w,
                             out.begin() + (h - 1 - row) * w);
        }
        return true;
    }

    // --- Shutdown ---------------------------------------------------------

    void Shutdown() {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);

        // Destroy textures
        for (auto& [h, e] : textures_) {
            if (e.view != VK_NULL_HANDLE)
                vkDestroyImageView(device_, e.view, nullptr);
            if (!e.is_shared && e.alloc != VK_NULL_HANDLE)
                vmaDestroyImage(vma_, e.image, e.alloc);
        }
        textures_.clear();

        // Destroy buffers
        for (auto& [h, e] : buffers_) {
            if (e.mapped) vmaUnmapMemory(vma_, e.alloc);
            vmaDestroyBuffer(vma_, e.buffer, e.alloc);
        }
        buffers_.clear();

        // Destroy pipelines
        for (auto& p : pipelines_) {
            if (!p.valid) continue;
            vkDestroyPipeline(device_, p.pipeline, nullptr);
            vkDestroyPipelineLayout(device_, p.layout, nullptr);
            vkDestroyDescriptorSetLayout(device_, p.dset_layout, nullptr);
            p.valid = false;
        }

        if (nearest_sampler_ != VK_NULL_HANDLE)
            vkDestroySampler(device_, nearest_sampler_, nullptr);
        if (fence_ != VK_NULL_HANDLE)
            vkDestroyFence(device_, fence_, nullptr);
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_);
            vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        }
        if (vma_ != VK_NULL_HANDLE) vmaDestroyAllocator(vma_);
        device_ = VK_NULL_HANDLE;
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void RegisterSharedImageInVulkanContext(GaussianSplatGpuContext& ctx,
                                        std::uint32_t gl_name,
                                        std::uintptr_t vk_image_opaque,
                                        std::uint32_t vk_format_opaque,
                                        std::uint32_t width,
                                        std::uint32_t height) {
    auto* vk_ctx = dynamic_cast<GaussianSplatGpuContextVulkan*>(&ctx);
    if (!vk_ctx) return;
    vk_ctx->RegisterSharedImage(gl_name,
                                reinterpret_cast<VkImage>(vk_image_opaque),
                                static_cast<VkFormat>(vk_format_opaque),
                                width, height);
}

void UnregisterSharedImageFromVulkanContext(GaussianSplatGpuContext& ctx,
                                            std::uint32_t gl_name) {
    auto* vk_ctx = dynamic_cast<GaussianSplatGpuContextVulkan*>(&ctx);
    if (!vk_ctx) return;
    vk_ctx->UnregisterSharedImage(gl_name);
}

std::unique_ptr<GaussianSplatGpuContext> CreateComputeGpuContextVulkan(
        bool use_subgroups) {
    auto& interop = GaussianSplatVulkanInteropContext::GetInstance();
    if (!interop.IsValid()) {
        utility::LogWarning(
                "GaussianSplatVulkan: interop context not initialized; "
                "Vulkan compute context not created");
        return nullptr;
    }
    return std::make_unique<GaussianSplatGpuContextVulkan>(use_subgroups);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
