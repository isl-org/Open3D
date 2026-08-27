// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless Vulkan context for Gaussian splatting compute (Linux/Windows).
//
// Owns the VkInstance, VkPhysicalDevice, VkDevice and compute queue used both
// by the GS compute pipeline and by Filament's Vulkan backend. The same
// VkDevice is shared with Filament via VulkanPlatform::VulkanSharedContext
// (passed as Engine::create()'s sharedContext argument). This eliminates the
// old GL-Vulkan interop path entirely: GS output images are plain VkImages
// imported into Filament through VulkanDriver::importTextureR().
//
// Key relationships:
//   - FilamentEngine.cpp calls Initialize() before Engine::create() and
//     Shutdown() after Engine::destroy() so the shared device outlives
//     Filament.
//   - GaussianSplatVulkanBackend uses GetDevice() / GetComputeQueue() for
//     compute dispatch and creates plain VkImages for depth/colour sharing.
//   - GetVulkanSharedContext() returns a VulkanSharedContext struct for
//     Engine::create(). The struct must stay alive for the engine's lifetime.
//
// Uses vulkan-hpp (Vulkan-Headers) for Vulkan loading and RAII handle
// lifetime management, and VMA (vk_mem_alloc.h from the pinned 3rdparty
// download) for suballocated internal-only buffer allocations.
//
// Thread-safety: not thread-safe. All calls must be made from the render
// thread.

#pragma once

#if !defined(__APPLE__)

#include <cstddef>
#include <cstdint>
#include <string>

// Suppress C function-prototype declarations in vulkan.h: all Vulkan entry
// points are resolved at runtime through vulkan-hpp's per-object RAII
// dispatchers (ContextDispatcher / InstanceDispatcher / DeviceDispatcher),
// not as statically-linked symbols.
//
// NOTE: BlueVK (used by Filament) also defines VK_NO_PROTOTYPES before
// including vulkan.h, so the two includes are mutually compatible when
// this header is included after Filament headers.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_raii.hpp>

// Forward-declare / define Filament's VulkanSharedContext struct here to
// avoid pulling in Filament headers (and their BlueVK dependency) in this
// public header.  Must match filament/backend/include/backend/platforms/
// VulkanPlatform.h exactly.
namespace filament::backend {
struct VulkanSharedContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logicalDevice = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamilyIndex = 0xFFFFFFFF;
    uint32_t graphicsQueueIndex = 0xFFFFFFFF;
};
}  // namespace filament::backend

namespace open3d {
namespace visualization {
namespace rendering {

/// Describes a single GPU image owned by Vulkan for sharing with Filament via
/// importTextureR(). The VkImage and VkDeviceMemory are allocated by
/// CreateImage() and destroyed by DestroyImage().
struct VkImageDesc {
    VkImage vk_image = VK_NULL_HANDLE;
    VkDeviceMemory vk_memory = VK_NULL_HANDLE;

    std::uint32_t width = 0;
    std::uint32_t height = 0;

    bool IsValid() const { return vk_image != VK_NULL_HANDLE; }
};

// ---------------------------------------------------------------------------
// GaussianSplatVulkanContext — headless Vulkan context
// ---------------------------------------------------------------------------

/// Manages a headless Vulkan instance, physical device selection, logical
/// device and compute queue. The same VkDevice is shared with Filament so
/// that GS compute output images are immediately visible without export/import.
///
/// Singleton, replaces the old GaussianSplatVulkanInteropContext.
class GaussianSplatVulkanContext {
public:
    static GaussianSplatVulkanContext& GetInstance();

    /// Load Vulkan via BlueVK, select a physical device
    /// (discrete-GPU-preferred), and create a logical device with compute
    /// capabilities. Two queue indices are requested from the graphics queue
    /// family: index 0 for GS compute, index 1 for Filament. On single-queue
    /// GPUs both indices are 0 and flushAndWait bracketing provides mutual
    /// exclusion (see plan Q2). Returns false on failure; call GetLastError()
    /// for a diagnostic string.
    bool Initialize();

    /// Release all Vulkan resources and invalidate the context.
    /// Must be called AFTER Filament's Engine::destroy() because Filament does
    /// NOT destroy a shared VkDevice/VkInstance (mSharedContext guard in
    /// VulkanPlatform::terminate()).
    void Shutdown();

    bool IsValid() const { return initialized_; }

    /// Human-readable description of the last failure.
    const std::string& GetLastError() const { return last_error_; }

    // --- Filament integration ---------------------------------------------

    /// Returns a VulkanSharedContext suitable for passing as the sharedContext
    /// argument to Engine::create(). The returned pointer stays valid until
    /// Shutdown().
    const filament::backend::VulkanSharedContext* GetVulkanSharedContext()
            const {
        return &shared_context_;
    }

    // --- Device accessors (used by VulkanBackend and ComputeGPUVulkan) -----

    VkInstance GetVkInstance() const {
        return static_cast<vk::Instance::CType>(*instance_);
    }
    VkDevice GetDevice() const {
        return static_cast<vk::Device::CType>(*device_);
    }
    VkPhysicalDevice GetPhysicalDevice() const {
        return static_cast<vk::PhysicalDevice::CType>(*physical_device_);
    }
    /// Compute queue at index 0 in the graphics family. Used for GS dispatch.
    VkQueue GetComputeQueue() const {
        return static_cast<vk::Queue::CType>(*compute_queue_);
    }
    std::uint32_t GetComputeQueueFamily() const {
        return graphics_queue_family_;
    }
    /// True when VK_EXT_debug_utils was available and enabled at instance
    /// creation.
    bool GetDebugUtilsEnabled() const { return debug_utils_enabled_; }
    /// Hardware subgroup size (gl_SubgroupSize) for compute shaders on this
    /// device. Returns 0 before Initialize().
    std::uint32_t GetSubgroupSize() const { return subgroup_size_; }
    std::uint32_t GetSubgroupSupportedStages() const {
        return subgroup_supported_stages_;
    }
    std::uint32_t GetSubgroupSupportedOperations() const {
        return subgroup_supported_operations_;
    }

    // RAII accessors used by ComputeGPUVulkan to create sub-objects.
    const vk::raii::Instance& GetRaiiInstance() const { return instance_; }
    const vk::raii::Device& GetRaiiDevice() const { return device_; }

    // --- Image lifecycle --------------------------------------------------

    /// Allocate a VkImage for sharing with Filament.
    VkImageDesc CreateImage(std::uint32_t width,
                            std::uint32_t height,
                            VkFormat vk_format,
                            VkImageUsageFlags usage,
                            const char* label = nullptr);

    /// Destroy an image previously created by CreateImage().
    void DestroyImage(VkImageDesc& desc);

    // --- Vulkan device memory type helpers --------------------------------

    /// Find a memory type index that satisfies type_filter (bitmask from
    /// VkMemoryRequirements) and the required property flags.
    /// Returns UINT32_MAX on failure.
    std::uint32_t FindMemoryType(std::uint32_t type_filter,
                                 VkMemoryPropertyFlags props) const;

private:
    GaussianSplatVulkanContext() = default;
    ~GaussianSplatVulkanContext();

    GaussianSplatVulkanContext(const GaussianSplatVulkanContext&) = delete;
    GaussianSplatVulkanContext& operator=(const GaussianSplatVulkanContext&) =
            delete;

    // --- Internal helpers -------------------------------------------------

    bool CreateInstance();
    bool SelectPhysicalDevice();
    bool CreateLogicalDevice();

    // --- State ------------------------------------------------------------
    bool initialized_ = false;
    bool debug_utils_enabled_ = false;
    std::uint32_t subgroup_size_ = 0;
    std::uint32_t subgroup_supported_stages_ = 0;
    std::uint32_t subgroup_supported_operations_ = 0;
    std::string last_error_;

    // Queue family indices within the graphics queue family.
    std::uint32_t graphics_queue_family_ = UINT32_MAX;
    std::uint32_t filament_queue_index_ = 1;  // index 0 = GS compute
    // True when the family has only 1 queue; both indices are 0.
    bool single_queue_device_ = false;

    // VulkanSharedContext held alive for Filament's lifetime.
    filament::backend::VulkanSharedContext shared_context_{};

    // RAII handles. Destruction order is reverse of declaration order:
    // compute_queue_ → device_ → physical_device_ → instance_ → context_.
    vk::raii::Context context_;
    vk::raii::Instance instance_{nullptr};
    vk::raii::PhysicalDevice physical_device_{nullptr};
    vk::raii::Device device_{nullptr};
    vk::raii::Queue compute_queue_{nullptr};

    VkPhysicalDeviceMemoryProperties memory_props_{};
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)