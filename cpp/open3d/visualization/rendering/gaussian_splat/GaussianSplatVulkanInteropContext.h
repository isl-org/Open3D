// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan-OpenGL interop context for Gaussian splatting compute (Linux/Windows).
//
// Owns a headless Vulkan instance and logical device used solely for compute
// work. All resources that must be visible to both Vulkan compute and
// OpenGL/Filament are allocated here as exportable memory or semaphores.
//
// Key relationships:
//   - GaussianSplatOpenGLContext creates the GL/GLFW context and calls
//     glewInit(). After that, call ProbeGLExtensions() to verify the
//     GL interop extensions are present.
//   - FilamentEngine.cpp calls Initialize() before GL context setup, and
//     Shutdown() on engine teardown.
//   - GaussianSplatVulkanBackend uses CreateSharedColorImage(),
//     CreateSharedDepthImage(), CreateSemaphorePair() per view, and holds the
//     resulting handles in OutputTargets.
//
// Uses vulkan-hpp (Vulkan-Headers) for Vulkan loading and RAII handle
// lifetime management, and VMA (vk_mem_alloc.h from the pinned 3rdparty
// download) for suballocated internal-only buffer allocations.
//
// Thread-safety: not thread-safe. All calls must be made from the render
// thread. Shared images / semaphores must be created or destroyed while the
// GL context is current (for the GL import calls).

#pragma once

#if !defined(__APPLE__)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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

namespace open3d {
namespace visualization {
namespace rendering {

// ---------------------------------------------------------------------------
// Shared resource descriptors
// ---------------------------------------------------------------------------

/// Image format selector used for interop image creation.
enum class VkInteropImageFormat {
    kRGBA16F,   ///< RGBA16F colour (VK_FORMAT_R16G16B16A16_SFLOAT / GL RGBA16F)
    kDepth32F,  ///< Depth-only (VK_FORMAT_D32_SFLOAT / GL DEPTH_COMPONENT32F)
};

/// Describes a single GPU image that is owned by Vulkan but visible to OpenGL
/// via the EXT_memory_object mechanism.
///
/// Lifetimesemantics:
///   1. CreateSharedColorImage() / CreateSharedDepthImage() allocates the
///      Vulkan image with a dedicated exportable allocation and simultaneously
///      imports it into OpenGL, returning a fully-initialised SharedImageDesc.
///   2. The gl_texture name (uint32_t) is passed to
///      FilamentResourceManager::CreateImportedTexture() exactly as today.
///   3. DestroySharedImage() first deletes the GL semaphore/memory-object, then
///      the Vulkan image and memory.
struct SharedImageDesc {
    VkImage vk_image = VK_NULL_HANDLE;
    VkDeviceMemory vk_memory = VK_NULL_HANDLE;  ///< Dedicated exportable alloc

    /// OpenGL objects wrapping the exported memory. These have a lifetime
    /// equal to vk_memory; destroy them before vkFreeMemory.
    std::uint32_t gl_memory_object = 0;  ///< glCreateMemoryObjectsEXT result
    std::uint32_t gl_texture = 0;        ///< GL texture name; pass to Filament

    /// Dimensions and format (stored for resize/recreate checks).
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkInteropImageFormat format = VkInteropImageFormat::kRGBA16F;

    /// Vulkan image layout currently assumed by the Vulkan side. Updated after
    /// each VkImageMemoryBarrier. Used to construct semaphore signal parameters
    /// (glSignalSemaphoreEXT layout argument).
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    bool IsValid() const { return vk_image != VK_NULL_HANDLE; }
};

/// Direction of a cross-API semaphore.
enum class SemaphoreDirection {
    kGLtoVK,  ///< GL signals → Vulkan waits (e.g. after Filament writes depth)
    kVKtoGL,  ///< Vulkan signals → GL waits (e.g. after GS composite writes
              ///< color)
};

/// Describes a binary semaphore exported from Vulkan and imported into OpenGL.
///
/// Usage per direction:
///   kGLtoVK: call glSignalSemaphoreEXT(gl_semaphore, ...) + glFlush(),
///            then VkSubmitInfo2.waitSemaphoreCount with vk_semaphore.
///   kVKtoGL: VkSubmitInfo2.signalSemaphoreCount with vk_semaphore,
///            then glWaitSemaphoreEXT(gl_semaphore, ...).
///
/// Binary semaphores must alternate strictly between signal and wait;
/// each member of the pair is used once per frame.
struct SharedSemaphoreDesc {
    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    std::uint32_t gl_semaphore = 0;  ///< glGenSemaphoresEXT result
    SemaphoreDirection direction = SemaphoreDirection::kGLtoVK;

    bool IsValid() const { return vk_semaphore != VK_NULL_HANDLE; }
};

// ---------------------------------------------------------------------------
// GaussianSplatVulkanInteropContext — headless Vulkan compute context
// ---------------------------------------------------------------------------

/// Manages a headless Vulkan instance, physical device selection, and logical
/// device for OpenGL-Vulkan interop in the GaussianSplat compute pipeline.
///
/// Singleton, parallel to GaussianSplatOpenGLContext. Initialize() must be
/// called before the OpenGL context because the GL-Vulkan interop texture
/// import path needs both contexts alive simultaneously.
class GaussianSplatVulkanInteropContext {
public:
    static GaussianSplatVulkanInteropContext& GetInstance();

    /// Load Vulkan via BlueVK, select a physical device with external-memory /
    /// external-semaphore extension support, and create a compute queue.
    /// Must be called BEFORE
    /// GaussianSplatOpenGLContext::InitializeStandalone(). Returns false on
    /// failure; call GetLastError() for a diagnostic string.
    bool Initialize();

    /// Release all Vulkan resources and invalidate the context.
    void Shutdown();

    bool IsValid() const { return initialized_; }

    /// Call this once after glewInit() to verify all required GL interop
    /// extension entry points are present. Returns false and sets the last
    /// error string if any required extension is missing.
    bool ProbeGLExtensions();

    bool AreGLExtensionsReady() const { return gl_extensions_ok_; }

    /// Human-readable description of the last failure (extension name, etc.).
    const std::string& GetLastError() const { return last_error_; }

    // --- Device accessors (used by VulkanBackend and ComputeGPUVulkan) ----
    // vk::Handle::CType provides the underlying C type (e.g.
    // vk::Instance::CType
    // == VkInstance). On 64-bit platforms vk::raii handles implicitly convert
    // to their C equivalents; static_cast is used here to be explicit and
    // 32-bit safe.
    VkInstance GetVkInstance() const {
        return static_cast<vk::Instance::CType>(*instance_);
    }
    VkDevice GetDevice() const {
        return static_cast<vk::Device::CType>(*device_);
    }
    VkPhysicalDevice GetPhysicalDevice() const {
        return static_cast<vk::PhysicalDevice::CType>(*physical_device_);
    }
    VkQueue GetComputeQueue() const {
        return static_cast<vk::Queue::CType>(*compute_queue_);
    }
    std::uint32_t GetComputeQueueFamily() const {
        return compute_queue_family_;
    }
    /// True when VK_EXT_debug_utils was available and enabled at instance
    /// creation.
    bool GetDebugUtilsEnabled() const { return debug_utils_enabled_; }
    /// Hardware subgroup size (gl_SubgroupSize) for compute shaders on this
    /// device. The two-level subgroup prefix-sum shader requires subgroupSize^2
    /// >= WG_SIZE (WG_SIZE=256 → need subgroupSize >= 16). Returns 0 before
    /// Initialize().
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

    // --- Shared-image lifecycle -------------------------------------------

    /// Allocate a Vulkan RGBA16F colour image with a dedicated exportable
    /// memory allocation, export the FD, and import into an OpenGL texture
    /// via EXT_memory_object. Requires the GL context to be current.
    /// Returns an invalid (IsValid()==false) descriptor on failure.
    SharedImageDesc CreateSharedColorImage(std::uint32_t width,
                                           std::uint32_t height,
                                           const char* label = nullptr);

    /// Same as CreateSharedColorImage() but allocates a DEPTH32F image
    /// suitable for Filament DEPTH_ATTACHMENT | SAMPLEABLE usage.
    SharedImageDesc CreateSharedDepthImage(std::uint32_t width,
                                           std::uint32_t height,
                                           const char* label = nullptr);

    /// Destroy a shared image: delete GL texture / memory-object, then free
    /// the Vulkan image and device memory.
    /// Requires the GL context to be current.
    void DestroySharedImage(SharedImageDesc& desc);

    // --- Semaphore lifecycle ----------------------------------------------

    /// Create one cross-API binary semaphore pair (one GL→VK, one VK→GL).
    /// The returned semaphores are ready to use; their initial state is
    /// unsignalled. Requires the GL context to be current.
    bool CreateSemaphorePair(SharedSemaphoreDesc& out_gl_to_vk,
                             SharedSemaphoreDesc& out_vk_to_gl);

    /// Destroy a single cross-API semaphore. Requires GL context current.
    void DestroySemaphore(SharedSemaphoreDesc& desc);

    // --- Vulkan device memory type helpers --------------------------------

    /// Find a memory type index that satisfies type_filter (bitmask from
    /// VkMemoryRequirements) and the required property flags.
    /// Returns UINT32_MAX on failure.
    std::uint32_t FindMemoryType(std::uint32_t type_filter,
                                 VkMemoryPropertyFlags props) const;

private:
    GaussianSplatVulkanInteropContext() = default;
    ~GaussianSplatVulkanInteropContext();

    GaussianSplatVulkanInteropContext(
            const GaussianSplatVulkanInteropContext&) = delete;
    GaussianSplatVulkanInteropContext& operator=(
            const GaussianSplatVulkanInteropContext&) = delete;

    // --- Internal helpers -------------------------------------------------

    bool CreateInstance();
    bool SelectPhysicalDevice();
    bool CreateLogicalDevice();

    /// Allocate a VkImage with a dedicated exportable memory allocation and
    /// export the platform file descriptor. Returns VK_NULL_HANDLE on failure.
    bool AllocateExportableImage(std::uint32_t width,
                                 std::uint32_t height,
                                 VkFormat vk_format,
                                 VkImageUsageFlags usage,
                                 VkImage& out_image,
                                 VkDeviceMemory& out_memory,
                                 int& out_fd) const;

    /// Import a Vulkan FD into an OpenGL memory-object and create a GL
    /// texture backed by that memory object.
    bool ImportFDIntoGL(int fd,
                        std::uint32_t width,
                        std::uint32_t height,
                        VkDeviceSize memory_size,
                        VkInteropImageFormat format,
                        std::uint32_t& out_gl_memory_object,
                        std::uint32_t& out_gl_texture) const;

    /// Create one VkSemaphore with VkExportSemaphoreCreateInfo and export its
    /// FD. Returns VK_NULL_HANDLE on failure.
    bool CreateExportableSemaphore(VkSemaphore& out_semaphore,
                                   int& out_fd) const;

    /// Import an FD into a GL semaphore object.
    bool ImportFDIntoGLSemaphore(int fd, std::uint32_t& out_gl_semaphore) const;

    SharedImageDesc CreateSharedImage(std::uint32_t width,
                                      std::uint32_t height,
                                      VkInteropImageFormat format,
                                      const char* label);

    // --- State ------------------------------------------------------------
    bool initialized_ = false;
    bool gl_extensions_ok_ = false;
    bool debug_utils_enabled_ =
            false;                     // VK_EXT_debug_utils enabled in instance
    std::uint32_t subgroup_size_ = 0;  // queried after logical device creation
    std::uint32_t subgroup_supported_stages_ = 0;
    std::uint32_t subgroup_supported_operations_ = 0;
    std::string last_error_;

    // RAII handles. Destruction order is reverse of declaration order:
    // compute_queue_ → device_ → physical_device_ → instance_ → context_.
    // vk::raii::Context loads the Vulkan loader; all others are sub-objects.
    vk::raii::Context context_;
    vk::raii::Instance instance_{nullptr};
    vk::raii::PhysicalDevice physical_device_{nullptr};
    vk::raii::Device device_{nullptr};
    vk::raii::Queue compute_queue_{nullptr};
    std::uint32_t compute_queue_family_ = UINT32_MAX;

    VkPhysicalDeviceMemoryProperties memory_props_{};
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
