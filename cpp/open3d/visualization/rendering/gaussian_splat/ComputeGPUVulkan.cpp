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

// vulkan-hpp RAII: dynamic dispatch through the per-object DeviceDispatcher;
// VK_NO_PROTOTYPES is defined transitively via GaussianSplatVulkanInteropContext.h.
// The global VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE is defined in
// GaussianSplatVulkanInteropContext.cpp (exactly once in the program).

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
    explicit GaussianSplatGpuContextVulkan(
            VulkanSubgroupOptions subgroup_options)
        : subgroup_options_(subgroup_options) {}

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
        debug_utils_enabled_ = interop.GetDebugUtilsEnabled();

        const std::uint32_t sg_size = interop.GetSubgroupSize();
        const auto subgroup_stages = static_cast<vk::ShaderStageFlags>(
            interop.GetSubgroupSupportedStages());
        const auto subgroup_ops = static_cast<vk::SubgroupFeatureFlags>(
            interop.GetSubgroupSupportedOperations());
        const bool compute_subgroups =
            static_cast<bool>(subgroup_stages &
                      vk::ShaderStageFlagBits::eCompute);
        const bool fixed_onesweep_subgroup_size =
                interop.SupportsRequiredComputeSubgroupSize(32u, 8u);

        if (subgroup_options_.enable_prefix_sum &&
            (!compute_subgroups || sg_size < 16 ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eBasic) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eArithmetic))) {
            utility::LogInfo(
                "GaussianSplatVulkan: disabling prefix_sum subgroup "
                "variant (compute_subgroups={}, subgroupSize={}, "
                "ops=0x{:x})",
                compute_subgroups, sg_size,
                interop.GetSubgroupSupportedOperations());
            subgroup_options_.enable_prefix_sum = false;
        }
        if (subgroup_options_.enable_onesweep &&
            (!compute_subgroups || !fixed_onesweep_subgroup_size ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eBasic) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eArithmetic) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eVote) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eBallot) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eShuffle))) {
            utility::LogInfo(
                "GaussianSplatVulkan: disabling OneSweep subgroup "
                "variants (compute_subgroups={}, subgroupSize={}, "
                "fixed32={}, ops=0x{:x})",
                compute_subgroups, sg_size, fixed_onesweep_subgroup_size,
                interop.GetSubgroupSupportedOperations());
            subgroup_options_.enable_onesweep = false;
        }
        if (subgroup_options_.enable_radix_sort &&
            (!compute_subgroups || sg_size == 0 ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eBasic) ||
             !(subgroup_ops & vk::SubgroupFeatureFlagBits::eArithmetic))) {
            utility::LogInfo(
                "GaussianSplatVulkan: disabling radix subgroup variant "
                "(compute_subgroups={}, subgroupSize={}, ops=0x{:x})",
                compute_subgroups, sg_size,
                interop.GetSubgroupSupportedOperations());
            subgroup_options_.enable_radix_sort = false;
        }

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
        // Persistent-mapped allocations (CPU_TO_GPU + VMA_ALLOCATION_CREATE_MAPPED_BIT)
        // are managed by VMA internally; do NOT call vmaUnmapMemory on them.
        // vmaDestroyBuffer handles cleanup including the persistent mapping.
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
        cmd_.fillBuffer(vk::Buffer(it->second.buffer), 0, VK_WHOLE_SIZE, 0u);
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
        pw.img = {static_cast<VkSampler>(*nearest_sampler_), view,
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
        // Full compute+transfer memory barrier using Vulkan 1.3 synchronization2.
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
        cmd_.pipelineBarrier2(
                vk::DependencyInfo{{}, mb, {}, {}});
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
        // Explicitly destroy the view before the VMA image for correct order.
        e.view = vk::raii::ImageView{nullptr};
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
        if (!debug_utils_enabled_) return;
        auto fn = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT;
        if (!fn) return;
        VkDebugUtilsLabelEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        info.pLabelName = label;
        fn(static_cast<VkCommandBuffer>(*cmd_), &info);
    }
    void PopDebugGroup() override {
        if (!debug_utils_enabled_) return;
        auto fn = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT;
        if (fn) fn(static_cast<VkCommandBuffer>(*cmd_));
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
        vk::raii::ImageView view = CreateImageView(image, format, aspect);
        if (static_cast<VkImageView>(*view) == VK_NULL_HANDLE) {
            utility::LogWarning(
                    "GaussianSplatVulkan: failed to create view for shared "
                    "gl_name={}",
                    gl_name);
            return;
        }
        TexEntry e{};
        e.image = image;
        e.view = std::move(view);
        e.alloc = VK_NULL_HANDLE;  // owned by VulkanInteropContext
        e.format = format;
        e.width = w;
        e.height = h;
        e.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        e.is_shared = true;
        uintptr_t handle = next_handle_++;
        textures_[handle] = std::move(e);
        gl_to_handle_[gl_name] = handle;
    }

    void UnregisterSharedImage(std::uint32_t gl_name) {
        auto it = gl_to_handle_.find(gl_name);
        if (it == gl_to_handle_.end()) return;
        uintptr_t h = it->second;
        auto te = textures_.find(h);
        if (te != textures_.end()) {
            // vk::raii::ImageView in TexEntry auto-destroyed on erase
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
        vk::raii::ImageView view{nullptr};  // RAII: auto-destroys vkDestroyImageView
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

    VulkanSubgroupOptions subgroup_options_{};
    bool programs_loaded_ = false;
    bool programs_valid_ = false;
    bool onesweep_valid_ = false;
    bool debug_utils_enabled_ = false;  // VK_EXT_debug_utils enabled at instance

    // --- Init helpers -----------------------------------------------------

    bool InitVma() {
        // VMA is compiled with VMA_STATIC_VULKAN_FUNCTIONS=0 and
        // VMA_DYNAMIC_VULKAN_FUNCTIONS=0 (set in GaussianSplatVulkanInteropContext.cpp
        // before VMA_IMPLEMENTATION), so all function pointers must be provided
        // manually. The VULKAN_HPP_DEFAULT_DISPATCHER holds all pointers after
        // GaussianSplatVulkanInteropContext::Initialize() calls
        // VULKAN_HPP_DEFAULT_DISPATCHER.init(instance) and .init(device).
        auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
        VmaVulkanFunctions vk_fn{};
        // Core dispatch loaders (required for VMA version detection)
        vk_fn.vkGetInstanceProcAddr = d.vkGetInstanceProcAddr;
        vk_fn.vkGetDeviceProcAddr = d.vkGetDeviceProcAddr;
        // Physical-device / memory queries
        vk_fn.vkGetPhysicalDeviceProperties = d.vkGetPhysicalDeviceProperties;
        vk_fn.vkGetPhysicalDeviceMemoryProperties = d.vkGetPhysicalDeviceMemoryProperties;
        // Memory allocation
        vk_fn.vkAllocateMemory = d.vkAllocateMemory;
        vk_fn.vkFreeMemory = d.vkFreeMemory;
        vk_fn.vkMapMemory = d.vkMapMemory;
        vk_fn.vkUnmapMemory = d.vkUnmapMemory;
        vk_fn.vkFlushMappedMemoryRanges = d.vkFlushMappedMemoryRanges;
        vk_fn.vkInvalidateMappedMemoryRanges = d.vkInvalidateMappedMemoryRanges;
        // Buffer / image lifecycle
        vk_fn.vkBindBufferMemory = d.vkBindBufferMemory;
        vk_fn.vkBindImageMemory = d.vkBindImageMemory;
        vk_fn.vkGetBufferMemoryRequirements = d.vkGetBufferMemoryRequirements;
        vk_fn.vkGetImageMemoryRequirements = d.vkGetImageMemoryRequirements;
        vk_fn.vkCreateBuffer = d.vkCreateBuffer;
        vk_fn.vkDestroyBuffer = d.vkDestroyBuffer;
        vk_fn.vkCreateImage = d.vkCreateImage;
        vk_fn.vkDestroyImage = d.vkDestroyImage;
        vk_fn.vkCmdCopyBuffer = d.vkCmdCopyBuffer;
        // Vulkan 1.1 / KHR equivalents required by VMA 3.x
        vk_fn.vkGetBufferMemoryRequirements2KHR = d.vkGetBufferMemoryRequirements2;
        vk_fn.vkGetImageMemoryRequirements2KHR = d.vkGetImageMemoryRequirements2;
        vk_fn.vkBindBufferMemory2KHR = d.vkBindBufferMemory2;
        vk_fn.vkBindImageMemory2KHR = d.vkBindImageMemory2;
        vk_fn.vkGetPhysicalDeviceMemoryProperties2KHR =
                d.vkGetPhysicalDeviceMemoryProperties2;
        // Vulkan 1.3 device memory query (lazy allocation support)
        vk_fn.vkGetDeviceBufferMemoryRequirements = d.vkGetDeviceBufferMemoryRequirements;
        vk_fn.vkGetDeviceImageMemoryRequirements = d.vkGetDeviceImageMemoryRequirements;

        VmaAllocatorCreateInfo ci{};
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        ci.physicalDevice = physical_device_;
        ci.device = device_;
        ci.instance = GaussianSplatVulkanInteropContext::GetInstance().GetVkInstance();
        ci.pVulkanFunctions = &vk_fn;
        if (vmaCreateAllocator(&ci, &vma_) != VK_SUCCESS) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vmaCreateAllocator failed");
            return false;
        }
        return true;
    }

    bool InitCommandPool() {
        auto& raii_dev =
                GaussianSplatVulkanInteropContext::GetInstance().GetRaiiDevice();
        vk::CommandPoolCreateInfo ci{
                vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                queue_family_,
        };
        try {
            cmd_pool_ = raii_dev.createCommandPool(ci);
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateCommandPool failed: {}",
                    e.what());
            return false;
        }
        vk::CommandBufferAllocateInfo ai{
                *cmd_pool_,
                vk::CommandBufferLevel::ePrimary,
                1,
        };
        try {
            auto cmds = raii_dev.allocateCommandBuffers(ai);
            cmd_ = std::move(cmds.front());
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkAllocateCommandBuffers failed: {}",
                    e.what());
            return false;
        }
        return true;
    }

    bool InitFence() {
        try {
            fence_ = GaussianSplatVulkanInteropContext::GetInstance()
                             .GetRaiiDevice()
                             .createFence({});
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateFence failed: {}", e.what());
            return false;
        }
        return true;
    }

    bool InitSampler() {
        vk::SamplerCreateInfo si{
                {},
                vk::Filter::eNearest,
                vk::Filter::eNearest,
                vk::SamplerMipmapMode::eNearest,
                vk::SamplerAddressMode::eClampToEdge,
                vk::SamplerAddressMode::eClampToEdge,
                vk::SamplerAddressMode::eClampToEdge,
        };
        try {
            nearest_sampler_ = GaussianSplatVulkanInteropContext::GetInstance()
                                        .GetRaiiDevice()
                                        .createSampler(si);
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateSampler failed: {}",
                    e.what());
            return false;
        }
        return true;
    }

    // --- Shader loading ---------------------------------------------------

    bool ShouldUseSubgroupVariant(ComputeProgramId id) const {
        switch (id) {
            case ComputeProgramId::kGsPrefixSum:
                return subgroup_options_.enable_prefix_sum;
            case ComputeProgramId::kGsRadixScatter:
                return subgroup_options_.enable_radix_sort;
            case ComputeProgramId::kGsOneSweepGlobalHist:
            case ComputeProgramId::kGsOneSweepDigitPass:
                return subgroup_options_.enable_onesweep;
            default:
                return false;
        }
    }

    bool LoadShader(ComputeProgramId id, const std::string& shader_root) {
        const int i = static_cast<int>(id);
        std::string name = kGsShaderNames[i];
        const bool want_subgroup = ShouldUseSubgroupVariant(id);
        const bool require_subgroup_size_32 =
            id == ComputeProgramId::kGsOneSweepGlobalHist ||
            id == ComputeProgramId::kGsOneSweepDigitPass;

        // Resolve subgroup variant name
        constexpr std::string_view kSubgroupSuffix = "_subgroup";
        const bool is_subgroup =
                name.size() > kSubgroupSuffix.size() &&
                name.compare(name.size() - kSubgroupSuffix.size(),
                             kSubgroupSuffix.size(), kSubgroupSuffix) == 0;

        // For non-subgroup names, optionally append _subgroup when enabled.
        // For already-subgroup names (_subgroup suffix), use as-is.
        std::string file_name;
        if (!is_subgroup && want_subgroup) {
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
        } else if (is_subgroup && !want_subgroup) {
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

        // RAII device: all vk::raii::Xxx handles auto-destroy on exception.
        auto& raii_dev =
                GaussianSplatVulkanInteropContext::GetInstance().GetRaiiDevice();
        vk::ShaderModuleCreateInfo smi{
                {},
                bytes.size(),
                reinterpret_cast<const std::uint32_t*>(bytes.data()),
        };
        vk::raii::ShaderModule shader_module{nullptr};
        try {
            shader_module = raii_dev.createShaderModule(smi);
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateShaderModule failed for {}: {}",
                    file_name, e.what());
            return false;
        }

        // Build descriptor set layout for this pipeline
        const auto& bt = kShaderBindings[i];
        std::vector<vk::DescriptorSetLayoutBinding> layout_bindings(bt.count);
        std::uint64_t binding_mask = 0;
        for (std::uint32_t j = 0; j < bt.count; ++j) {
            auto& lb = layout_bindings[j];
            lb.binding = bt.descs[j].binding;
            lb.descriptorType = static_cast<vk::DescriptorType>(bt.descs[j].type);
            lb.descriptorCount = 1;
            lb.stageFlags = vk::ShaderStageFlagBits::eCompute;
            if (lb.binding < 64) binding_mask |= (uint64_t(1) << lb.binding);
        }

        vk::DescriptorSetLayoutCreateInfo dslci{};
        // Enable push descriptors for this layout.
        dslci.flags = static_cast<vk::DescriptorSetLayoutCreateFlags>(
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
        dslci.bindingCount = bt.count;
        dslci.pBindings = layout_bindings.data();
        vk::raii::DescriptorSetLayout dset_layout{nullptr};
        try {
            dset_layout = raii_dev.createDescriptorSetLayout(dslci);
        } catch (const vk::SystemError& e) {
            // shader_module auto-destroyed by RAII
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateDescriptorSetLayout failed "
                    "for {}: {}",
                    file_name, e.what());
            return false;
        }

        const vk::DescriptorSetLayout dsl_handle = *dset_layout;
        vk::PipelineLayoutCreateInfo plci{{}, 1, &dsl_handle};
        vk::raii::PipelineLayout pipeline_layout{nullptr};
        try {
            pipeline_layout = raii_dev.createPipelineLayout(plci);
        } catch (const vk::SystemError& e) {
            // dset_layout, shader_module auto-destroyed by RAII
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreatePipelineLayout failed "
                    "for {}: {}",
                    file_name, e.what());
            return false;
        }

        vk::PipelineShaderStageCreateInfo stage_info{
            {},
            vk::ShaderStageFlagBits::eCompute,
            *shader_module,
            "main",
        };
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo
            required_subgroup_size_info{};
        if (want_subgroup && require_subgroup_size_32) {
            stage_info.flags =
                vk::PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize;
            required_subgroup_size_info.requiredSubgroupSize = 32u;
            stage_info.pNext = &required_subgroup_size_info;
        }

        vk::ComputePipelineCreateInfo pci{{}, stage_info, *pipeline_layout};
        vk::raii::Pipeline pipeline{nullptr};
        try {
            pipeline = raii_dev.createComputePipeline(nullptr, pci);
        } catch (const vk::SystemError& e) {
            // pipeline_layout, dset_layout, shader_module auto-destroyed by RAII
            utility::LogWarning(
                    "GaussianSplatVulkan: vkCreateComputePipelines failed "
                    "for {}: {}",
                    file_name, e.what());
            return false;
        }
        // shader_module auto-destroyed here (pipeline compiled; module no longer needed)

        auto& p = pipelines_[i];
        p.dset_layout = std::move(dset_layout);
        p.layout = std::move(pipeline_layout);
        p.pipeline = std::move(pipeline);
        p.binding_mask = binding_mask;
        p.valid = true;
        utility::LogDebug("GaussianSplatVulkan: loaded {}", file_name);
        return true;
    }

    // --- Command buffer lifecycle -----------------------------------------

    void BeginCmdBuf() {
        if (cmd_active_) return;
        // If the fence is still pending from a previous submission, wait for
        // it before resetting the command buffer.  This guards against the
        // case where BeginCmdBuf is called again before SubmitAndWait has had
        // a chance to drain (e.g. geometry pass on frame N+1 vs composite
        // pass on frame N that was never submitted due to an early return).
        if (fence_submitted_) {
            (void)vk::Device(device_).waitForFences(
                    {*fence_}, true, UINT64_MAX);
            vk::Device(device_).resetFences({*fence_});
            fence_submitted_ = false;
        }
        cmd_.reset({});
        cmd_.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd_active_ = true;
        active_id_ = -1;
        pending_.clear();
    }

    void SubmitAndWait() {
        if (!cmd_active_) return;
        cmd_.end();
        cmd_active_ = false;
        // Reset the fence only if it is not already pending.  BeginCmdBuf
        // above handles the case where we start a new buffer while pending.
        if (!fence_submitted_) {
            vk::Device(device_).resetFences({*fence_});
        }
        vk::CommandBufferSubmitInfo cmd_info{*cmd_, 0};
        vk::SubmitInfo2 si{{}, {}, cmd_info, {}};
        utility::LogDebug("GaussianSplatVulkan: submit");
        vk::Queue(compute_queue_).submit2(si, *fence_);
        fence_submitted_ = true;
        (void)vk::Device(device_).waitForFences({*fence_}, true, UINT64_MAX);
        vk::Device(device_).resetFences({*fence_});
        fence_submitted_ = false;
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
            // vkCmdPushDescriptorSetKHR is a KHR extension function; call it
            // through the default dispatcher which was loaded with the device.
            VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPushDescriptorSetKHR(
                    static_cast<VkCommandBuffer>(*cmd_),
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    static_cast<VkPipelineLayout>(*p.layout), 0,
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
        return static_cast<VkImageView>(*e.view);
    }

    void TransitionImageLayout(VkImage image,
                               VkFormat format,
                               VkImageLayout old_layout,
                               VkImageLayout new_layout) {
        const bool is_depth = (format == VK_FORMAT_D32_SFLOAT);
        const vk::AccessFlags2 dst_access =
                (new_layout == VK_IMAGE_LAYOUT_GENERAL)
                        ? (vk::AccessFlagBits2::eShaderWrite |
                           vk::AccessFlagBits2::eShaderRead)
                        : vk::AccessFlagBits2::eShaderRead;
        vk::ImageMemoryBarrier2 b{
                vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
                vk::PipelineStageFlagBits2::eComputeShader,
                dst_access,
                static_cast<vk::ImageLayout>(old_layout),
                static_cast<vk::ImageLayout>(new_layout),
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                vk::Image(image),
                vk::ImageSubresourceRange{
                        is_depth ? vk::ImageAspectFlagBits::eDepth
                                 : vk::ImageAspectFlagBits::eColor,
                        0, 1, 0, 1},
        };
        cmd_.pipelineBarrier2(
                vk::DependencyInfo{{}, {}, {}, b});
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
            return GaussianSplatVulkanInteropContext::GetInstance()
                    .GetRaiiDevice()
                    .createImageView(vci);
        } catch (const vk::SystemError&) {
            return vk::raii::ImageView{nullptr};
        }
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
        vk::BufferImageCopy region{
                0, 0, 0,
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                {0, 0, 0},
                {w, h, 1},
        };
        cmd_.copyImageToBuffer(
                vk::Image(e.image),
                vk::ImageLayout::eTransferSrcOptimal,
                vk::Buffer(staging), region);
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
        vk::Device(device_).waitIdle();

        // Destroy image views (RAII) before their VMA images.
        for (auto& [h, e] : textures_) {
            e.view = vk::raii::ImageView{nullptr};  // vkDestroyImageView
            if (!e.is_shared && e.alloc != VK_NULL_HANDLE)
                vmaDestroyImage(vma_, e.image, e.alloc);
        }
        textures_.clear();

        // Destroy VMA buffers.
        for (auto& [h, e] : buffers_) {
            vmaDestroyBuffer(vma_, e.buffer, e.alloc);
        }
        buffers_.clear();

        // Reset pipeline RAII handles: destroys pipelines, layouts, dset layouts.
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
    VulkanSubgroupOptions subgroup_options) {
    auto& interop = GaussianSplatVulkanInteropContext::GetInstance();
    if (!interop.IsValid()) {
        utility::LogWarning(
                "GaussianSplatVulkan: interop context not initialized; "
                "Vulkan compute context not created");
        return nullptr;
    }
    return std::make_unique<GaussianSplatGpuContextVulkan>(subgroup_options);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
