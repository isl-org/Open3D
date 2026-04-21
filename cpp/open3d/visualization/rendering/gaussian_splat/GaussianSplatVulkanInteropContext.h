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
// Uses BlueVK (libbluevk in the Filament prebuilt) for dynamic Vulkan loading
// and VMA (vk_mem_alloc.h vendored in 3rdparty/vkmemalloc/) for suballocated
// internal-only buffer allocations.
//
// Thread-safety: not thread-safe. All calls must be made from the render
// thread. Shared images / semaphores must be created or destroyed while the
// GL context is current (for the GL import calls).

#pragma once

#if !defined(__APPLE__)

#include <cstdint>
#include <string>
#include <vector>

// BlueVK defines VK_NO_PROTOTYPES and includes <vulkan/vulkan.h>.
// This makes Vulkan types available without linking to libvulkan.so at
// compile time; all function pointers are loaded dynamically at runtime.
#include <bluevk/BlueVK.h>

namespace open3d {
namespace visualization {
namespace rendering {

// ---------------------------------------------------------------------------
// Shared resource descriptors
// ---------------------------------------------------------------------------

/// Image format selector used for interop image creation.
enum class VkInteropImageFormat {
    kRGBA16F,        ///< RGBA16F colour (VK_FORMAT_R16G16B16A16_SFLOAT / GL RGBA16F)
    kDepth32F,       ///< Depth-only (VK_FORMAT_D32_SFLOAT / GL DEPTH_COMPONENT32F)
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
    kVKtoGL,  ///< Vulkan signals → GL waits (e.g. after GS composite writes color)
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
    /// Must be called BEFORE GaussianSplatOpenGLContext::InitializeStandalone().
    /// Returns false on failure; call GetLastError() for a diagnostic string.
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

    // --- Device accessors (used by VulkanBackend) --------------------------
    VkDevice GetDevice() const { return device_; }
    VkPhysicalDevice GetPhysicalDevice() const { return physical_device_; }
    VkQueue GetComputeQueue() const { return compute_queue_; }
    std::uint32_t GetComputeQueueFamily() const { return compute_queue_family_; }

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

    GaussianSplatVulkanInteropContext(const GaussianSplatVulkanInteropContext&) =
            delete;
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
    bool CreateExportableSemaphore(VkSemaphore& out_semaphore, int& out_fd) const;

    /// Import an FD into a GL semaphore object.
    bool ImportFDIntoGLSemaphore(int fd,
                                 std::uint32_t& out_gl_semaphore) const;

    SharedImageDesc CreateSharedImage(std::uint32_t width,
                                      std::uint32_t height,
                                      VkInteropImageFormat format,
                                      const char* label);

    // --- State ------------------------------------------------------------
    bool initialized_ = false;
    bool gl_extensions_ok_ = false;
    std::string last_error_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_ = UINT32_MAX;

    VkPhysicalDeviceMemoryProperties memory_props_{};
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
