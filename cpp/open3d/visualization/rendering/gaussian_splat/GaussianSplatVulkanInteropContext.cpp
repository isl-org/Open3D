// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Vulkan-OpenGL interop context implementation (Linux and Windows).
//
// Sequence of calls for one shared colour image (depth is symmetric):
//   1. AllocateExportableImage() → VkImage + VkDeviceMemory + FD
//   2. ImportFDIntoGL()          → gl_memory_object + gl_texture
//   3. (caller) CreateImportedTexture(gl_texture, ...) into Filament
// Semaphore pair (one GL→VK, one VK→GL per view):
//   1. CreateExportableSemaphore() × 2 → VkSemaphore + FD each
//   2. ImportFDIntoGLSemaphore()   × 2 → gl_semaphore each

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"

#if !defined(__APPLE__)

#include <GL/glew.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "open3d/utility/Logging.h"

// Initialize VMA in this single translation unit.
// VK_NO_PROTOTYPES is already defined by bluevk/BlueVK.h (included via
// the context header), so VMA must use VmaVulkanFunctions rather than
// calling Vulkan symbols directly.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
// Suppress pedantic warnings inside the VMA header-only implementation.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#pragma GCC diagnostic pop

namespace open3d {
namespace visualization {
namespace rendering {

// All Vulkan function pointers live in namespace bluevk after
// bluevk::initialize() and bluevk::bindInstance(). Bring them into scope
// so the rest of this TU can call vkCreateInstance() etc. directly.
using namespace bluevk;  // NOLINT(build/namespaces)

// ---------------------------------------------------------------------------
// Required Vulkan device extensions for interop
// ---------------------------------------------------------------------------
namespace {

#if defined(_WIN32)
constexpr const char* kRequiredDeviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
};
#else
constexpr const char* kRequiredDeviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
};
#endif

constexpr const char* kRequiredInstanceExtensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

/// Returns true when all extensions in 'required' are present in
/// 'available'.  Sets out_missing to the first missing extension name.
bool CheckExtensions(const std::vector<VkExtensionProperties>& available,
                     const char* const* required,
                     std::size_t required_count,
                     std::string& out_missing) {
    for (std::size_t i = 0; i < required_count; ++i) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, required[i]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            out_missing = required[i];
            return false;
        }
    }
    return true;
}

/// Score a physical device for interop suitability.  Higher is better.
///  +2  : discrete GPU
///  +1  : integrated GPU
///   0  : any compute-capable device
///  -∞  : no compute queue or missing required extensions → reject
int ScoreDevice(VkPhysicalDevice dev) {
    // Check for a compute queue.
    std::uint32_t qfam_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfam_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfams(qfam_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfam_count, qfams.data());
    bool has_compute = false;
    for (const auto& qf : qfams) {
        if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            has_compute = true;
            break;
        }
    }
    if (!has_compute) return -1;

    // Check device extensions.
    std::uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, exts.data());

    std::string missing;
    const bool ok = CheckExtensions(
            exts, kRequiredDeviceExtensions,
            std::size(kRequiredDeviceExtensions), missing);
    if (!ok) return -1;

    // Score device type.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 2;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 1;
    return 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

GaussianSplatVulkanInteropContext&
GaussianSplatVulkanInteropContext::GetInstance() {
    static GaussianSplatVulkanInteropContext instance;
    return instance;
}

GaussianSplatVulkanInteropContext::~GaussianSplatVulkanInteropContext() {
    Shutdown();
}

// ---------------------------------------------------------------------------
// Initialize / Shutdown
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::Initialize() {
    if (initialized_) return true;

    // BlueVK: dynamically load the Vulkan library and global entry points.
    if (!bluevk::initialize()) {
        last_error_ = "bluevk::initialize() failed: Vulkan loader not found";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    if (!CreateInstance()) return false;

    // Bind instance-level Vulkan function pointers through BlueVK.
    bluevk::bindInstance(instance_);

    if (!SelectPhysicalDevice()) return false;
    if (!CreateLogicalDevice()) return false;

    initialized_ = true;
    utility::LogDebug("GaussianSplat VulkanInteropContext: initialized ({})",
                      [&] {
                          VkPhysicalDeviceProperties p{};
                          vkGetPhysicalDeviceProperties(physical_device_, &p);
                          return std::string(p.deviceName);
                      }());
    return true;
}

void GaussianSplatVulkanInteropContext::Shutdown() {
    if (!initialized_) return;

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    physical_device_ = VK_NULL_HANDLE;
    compute_queue_ = VK_NULL_HANDLE;
    compute_queue_family_ = UINT32_MAX;
    initialized_ = false;
    gl_extensions_ok_ = false;
    utility::LogDebug("GaussianSplat VulkanInteropContext: shutdown");
}

// ---------------------------------------------------------------------------
// ProbeGLExtensions
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::ProbeGLExtensions() {
    // All required GL interop extension entry points are loaded by GLEW
    // after glewInit(). GLEW_GET_FUN returns nullptr for absent extensions.
    struct GLExtEntry {
        const char* name;
        bool present;
    } entries[] = {
            {"GL_EXT_memory_object", GLEW_EXT_memory_object != 0},
            {"GL_EXT_semaphore", GLEW_EXT_semaphore != 0},
#if defined(_WIN32)
            {"GL_EXT_memory_object_win32", GLEW_EXT_memory_object_win32 != 0},
            {"GL_EXT_semaphore_win32", GLEW_EXT_semaphore_win32 != 0},
#else
            {"GL_EXT_memory_object_fd", GLEW_EXT_memory_object_fd != 0},
            {"GL_EXT_semaphore_fd", GLEW_EXT_semaphore_fd != 0},
#endif
    };

    for (const auto& e : entries) {
        if (!e.present) {
            last_error_ = std::string("Missing GL extension: ") + e.name;
            utility::LogWarning("GaussianSplat Vulkan interop: {}", last_error_);
            return false;
        }
    }
    gl_extensions_ok_ = true;
    utility::LogDebug("GaussianSplat Vulkan interop: all GL extensions present");
    return true;
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateInstance() {
    // Check instance extension availability.
    std::uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available_exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count,
                                           available_exts.data());

    std::string missing;
    if (!CheckExtensions(available_exts, kRequiredInstanceExtensions,
                         std::size(kRequiredInstanceExtensions), missing)) {
        last_error_ = "Missing Vulkan instance extension: " + missing;
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Open3D GaussianSplat";
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app_info;
    ci.enabledExtensionCount =
            static_cast<std::uint32_t>(std::size(kRequiredInstanceExtensions));
    ci.ppEnabledExtensionNames = kRequiredInstanceExtensions;

    const VkResult res = vkCreateInstance(&ci, nullptr, &instance_);
    if (res != VK_SUCCESS) {
        last_error_ = "vkCreateInstance failed (VkResult=" +
                      std::to_string(static_cast<int>(res)) + ")";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::SelectPhysicalDevice() {
    std::uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_count, nullptr);
    if (dev_count == 0) {
        last_error_ = "No Vulkan-capable devices found";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance_, &dev_count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int best_score = -1;
    for (VkPhysicalDevice dev : devices) {
        const int score = ScoreDevice(dev);
        if (score > best_score) {
            best_score = score;
            best = dev;
        }
    }

    if (best == VK_NULL_HANDLE) {
        last_error_ =
                "No suitable Vulkan device found with required interop "
                "extensions. Required extensions: "
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
                ", " VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    physical_device_ = best;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_props_);
    utility::LogDebug("GaussianSplat Vulkan: selected device '{}'",
                      props.deviceName);
    return true;
}

// ---------------------------------------------------------------------------
// Logical device and compute queue
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateLogicalDevice() {
    std::uint32_t qfam_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qfam_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> qfams(qfam_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &qfam_count,
                                             qfams.data());

    // Prefer a compute-only queue (no graphics flag) for dedicated compute.
    std::uint32_t compute_family = UINT32_MAX;
    std::uint32_t graphics_compute_family = UINT32_MAX;
    for (std::uint32_t i = 0; i < qfam_count; ++i) {
        if (qfams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (!(qfams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                compute_family = i;  // dedicated compute
                break;
            }
            if (graphics_compute_family == UINT32_MAX) {
                graphics_compute_family = i;
            }
        }
    }
    compute_queue_family_ = (compute_family != UINT32_MAX)
                                    ? compute_family
                                    : graphics_compute_family;
    if (compute_queue_family_ == UINT32_MAX) {
        last_error_ = "No compute queue family found";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = compute_queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_priority;

    // Enable required subgroup features for OneSweep (Vulkan 1.1+).
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &features2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount =
            static_cast<std::uint32_t>(std::size(kRequiredDeviceExtensions));
    dci.ppEnabledExtensionNames = kRequiredDeviceExtensions;

    const VkResult res =
            vkCreateDevice(physical_device_, &dci, nullptr, &device_);
    if (res != VK_SUCCESS) {
        last_error_ = "vkCreateDevice failed (VkResult=" +
                      std::to_string(static_cast<int>(res)) + ")";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    utility::LogDebug("GaussianSplat Vulkan: compute queue family {}",
                      compute_queue_family_);
    return true;
}

// ---------------------------------------------------------------------------
// Memory type helper
// ---------------------------------------------------------------------------

std::uint32_t GaussianSplatVulkanInteropContext::FindMemoryType(
        std::uint32_t type_filter, VkMemoryPropertyFlags props) const {
    for (std::uint32_t i = 0; i < memory_props_.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (memory_props_.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// Shared image creation
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::AllocateExportableImage(
        std::uint32_t width,
        std::uint32_t height,
        VkFormat vk_format,
        VkImageUsageFlags usage,
        VkImage& out_image,
        VkDeviceMemory& out_memory,
        int& out_fd) const {
    // Declare that this image's memory will be exported as a POSIX FD
    // (OpaqueFd handle type, compatible with GL EXT_memory_object_fd).
    VkExternalMemoryImageCreateInfo ext_img_ci{};
    ext_img_ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#if defined(_WIN32)
    ext_img_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    ext_img_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.pNext = &ext_img_ci;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = vk_format;
    img_ci.extent = {width, height, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = usage;
    img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device_, &img_ci, nullptr, &out_image) != VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkCreateImage failed");
        return false;
    }

    // Require a dedicated allocation (required by most drivers for
    // external-memory images and strongly recommended by the spec).
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.image = out_image;

    VkExportMemoryAllocateInfo export_ai{};
    export_ai.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_ai.pNext = &dedicated;
#if defined(_WIN32)
    export_ai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    export_ai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(device_, out_image, &mem_req);

    const std::uint32_t mem_type = FindMemoryType(
            mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        utility::LogWarning(
                "GaussianSplat Vulkan: no suitable device-local memory type");
        vkDestroyImage(device_, out_image, nullptr);
        out_image = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo alloc_i{};
    alloc_i.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_i.pNext = &export_ai;
    alloc_i.allocationSize = mem_req.size;
    alloc_i.memoryTypeIndex = mem_type;

    if (vkAllocateMemory(device_, &alloc_i, nullptr, &out_memory) !=
        VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkAllocateMemory failed");
        vkDestroyImage(device_, out_image, nullptr);
        out_image = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindImageMemory(device_, out_image, out_memory, 0) != VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkBindImageMemory failed");
        vkFreeMemory(device_, out_memory, nullptr);
        vkDestroyImage(device_, out_image, nullptr);
        out_image = VK_NULL_HANDLE;
        out_memory = VK_NULL_HANDLE;
        return false;
    }

    // Export the memory as a platform FD.
#if defined(_WIN32)
    VkMemoryGetWin32HandleInfoKHR gh_info{};
    gh_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    gh_info.memory = out_memory;
    gh_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32_handle = nullptr;
    if (vkGetMemoryWin32HandleKHR(device_, &gh_info, &win32_handle) !=
        VK_SUCCESS) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkGetMemoryWin32HandleKHR failed");
        vkFreeMemory(device_, out_memory, nullptr);
        vkDestroyImage(device_, out_image, nullptr);
        out_image = VK_NULL_HANDLE;
        out_memory = VK_NULL_HANDLE;
        return false;
    }
    // Store the HANDLE as an int via reinterpret so the API is uniform.
    out_fd = static_cast<int>(reinterpret_cast<intptr_t>(win32_handle));
#else
    VkMemoryGetFdInfoKHR gfd_info{};
    gfd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gfd_info.memory = out_memory;
    gfd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (vkGetMemoryFdKHR(device_, &gfd_info, &out_fd) != VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkGetMemoryFdKHR failed");
        vkFreeMemory(device_, out_memory, nullptr);
        vkDestroyImage(device_, out_image, nullptr);
        out_image = VK_NULL_HANDLE;
        out_memory = VK_NULL_HANDLE;
        return false;
    }
#endif
    return true;
}

bool GaussianSplatVulkanInteropContext::ImportFDIntoGL(
        int fd,
        std::uint32_t width,
        std::uint32_t height,
        VkDeviceSize memory_size,
        VkInteropImageFormat format,
        std::uint32_t& out_gl_memory_object,
        std::uint32_t& out_gl_texture) const {
    // Create a GL memory object and import the Vulkan FD into it.
    glCreateMemoryObjectsEXT(1, &out_gl_memory_object);
    if (out_gl_memory_object == 0) {
        utility::LogWarning(
                "GaussianSplat GL: glCreateMemoryObjectsEXT returned 0");
        return false;
    }

#if defined(_WIN32)
    const HANDLE win32_handle = reinterpret_cast<HANDLE>(
            static_cast<intptr_t>(fd));
    glImportMemoryWin32HandleEXT(out_gl_memory_object,
                                 static_cast<GLuint64>(memory_size),
                                 GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32_handle);
#else
    glImportMemoryFdEXT(out_gl_memory_object,
                        static_cast<GLuint64>(memory_size),
                        GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
    // FD ownership is transferred to GL; do not close it.
#endif

    // Create a GL texture backed by the imported memory object.
    glCreateTextures(GL_TEXTURE_2D, 1, &out_gl_texture);
    if (out_gl_texture == 0) {
        glDeleteMemoryObjectsEXT(1, &out_gl_memory_object);
        out_gl_memory_object = 0;
        utility::LogWarning("GaussianSplat GL: glCreateTextures returned 0");
        return false;
    }

    // Allocate GL texture storage bound to the imported memory.
    GLenum gl_internal_format = 0;
    switch (format) {
        case VkInteropImageFormat::kRGBA16F:
            gl_internal_format = GL_RGBA16F;
            break;
        case VkInteropImageFormat::kDepth32F:
            gl_internal_format = GL_DEPTH_COMPONENT32F;
            break;
    }
    glTextureStorageMem2DEXT(out_gl_texture, 1, gl_internal_format,
                             static_cast<GLsizei>(width),
                             static_cast<GLsizei>(height),
                             out_gl_memory_object, /*offset=*/0);

    const GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        utility::LogWarning(
                "GaussianSplat GL: glTextureStorageMem2DEXT error 0x{:x}",
                static_cast<unsigned>(err));
        glDeleteTextures(1, &out_gl_texture);
        glDeleteMemoryObjectsEXT(1, &out_gl_memory_object);
        out_gl_texture = 0;
        out_gl_memory_object = 0;
        return false;
    }
    return true;
}

SharedImageDesc GaussianSplatVulkanInteropContext::CreateSharedImage(
        std::uint32_t width,
        std::uint32_t height,
        VkInteropImageFormat interop_format,
        const char* label) {
    SharedImageDesc desc{};
    if (!initialized_ || !gl_extensions_ok_) {
        utility::LogWarning(
                "GaussianSplat Vulkan: CreateSharedImage called before "
                "Initialize()+ProbeGLExtensions()");
        return desc;
    }

    VkFormat vk_fmt = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    switch (interop_format) {
        case VkInteropImageFormat::kRGBA16F:
            vk_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
            // STORAGE for compute write; SAMPLED for Filament/UI read;
            // COLOR_ATTACHMENT: required for Filament COLOR_ATTACHMENT usage.
            usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            break;
        case VkInteropImageFormat::kDepth32F:
            vk_fmt = VK_FORMAT_D32_SFLOAT;
            // DEPTH_STENCIL_ATTACHMENT: Filament writes depth into it.
            // SAMPLED: Vulkan composite pass samples it for occlusion.
            usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
            break;
    }

    int export_fd = -1;
    VkMemoryRequirements mem_req{};
    {
        // Probe memory requirements before the full allocation.
        VkImageCreateInfo probe_ci{};
        probe_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        probe_ci.imageType = VK_IMAGE_TYPE_2D;
        probe_ci.format = vk_fmt;
        probe_ci.extent = {width, height, 1};
        probe_ci.mipLevels = 1;
        probe_ci.arrayLayers = 1;
        probe_ci.samples = VK_SAMPLE_COUNT_1_BIT;
        probe_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        probe_ci.usage = usage;
        VkImage probe_img = VK_NULL_HANDLE;
        vkCreateImage(device_, &probe_ci, nullptr, &probe_img);
        if (probe_img != VK_NULL_HANDLE) {
            vkGetImageMemoryRequirements(device_, probe_img, &mem_req);
            vkDestroyImage(device_, probe_img, nullptr);
        }
    }

    bool ok = AllocateExportableImage(width, height, vk_fmt, usage,
                                      desc.vk_image, desc.vk_memory, export_fd);
    if (!ok) return desc;

    // Query actual memory size for the GL import call.
    VkMemoryRequirements actual_req{};
    vkGetImageMemoryRequirements(device_, desc.vk_image, &actual_req);

    ok = ImportFDIntoGL(export_fd, width, height,
                        actual_req.size, interop_format,
                        desc.gl_memory_object, desc.gl_texture);
    if (!ok) {
        vkFreeMemory(device_, desc.vk_memory, nullptr);
        vkDestroyImage(device_, desc.vk_image, nullptr);
        desc = SharedImageDesc{};
        return desc;
    }

    desc.width = width;
    desc.height = height;
    desc.format = interop_format;
    desc.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (label) {
        utility::LogDebug("GaussianSplat Vulkan: created shared image '{}' "
                          "{}x{} GL={}",
                          label, width, height, desc.gl_texture);
    }
    return desc;
}

SharedImageDesc GaussianSplatVulkanInteropContext::CreateSharedColorImage(
        std::uint32_t width, std::uint32_t height, const char* label) {
    return CreateSharedImage(width, height, VkInteropImageFormat::kRGBA16F,
                             label);
}

SharedImageDesc GaussianSplatVulkanInteropContext::CreateSharedDepthImage(
        std::uint32_t width, std::uint32_t height, const char* label) {
    return CreateSharedImage(width, height, VkInteropImageFormat::kDepth32F,
                             label);
}

void GaussianSplatVulkanInteropContext::DestroySharedImage(
        SharedImageDesc& desc) {
    if (!desc.IsValid()) return;

    // GL objects must be destroyed before the underlying Vulkan memory.
    if (desc.gl_texture != 0) {
        glDeleteTextures(1, &desc.gl_texture);
        desc.gl_texture = 0;
    }
    if (desc.gl_memory_object != 0) {
        glDeleteMemoryObjectsEXT(1, &desc.gl_memory_object);
        desc.gl_memory_object = 0;
    }
    // Now safe to free Vulkan resources.
    if (desc.vk_image != VK_NULL_HANDLE) {
        vkDestroyImage(device_, desc.vk_image, nullptr);
        desc.vk_image = VK_NULL_HANDLE;
    }
    if (desc.vk_memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, desc.vk_memory, nullptr);
        desc.vk_memory = VK_NULL_HANDLE;
    }
    desc = SharedImageDesc{};
}

// ---------------------------------------------------------------------------
// Semaphore lifecycle
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateExportableSemaphore(
        VkSemaphore& out_semaphore, int& out_fd) const {
    VkExportSemaphoreCreateInfo export_ci{};
    export_ci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
#if defined(_WIN32)
    export_ci.handleTypes =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    export_ci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &export_ci;

    if (vkCreateSemaphore(device_, &sci, nullptr, &out_semaphore) !=
        VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkCreateSemaphore failed");
        return false;
    }

#if defined(_WIN32)
    VkSemaphoreGetWin32HandleInfoKHR gh_info{};
    gh_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    gh_info.semaphore = out_semaphore;
    gh_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE win32_handle = nullptr;
    if (vkGetSemaphoreWin32HandleKHR(device_, &gh_info, &win32_handle) !=
        VK_SUCCESS) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkGetSemaphoreWin32HandleKHR failed");
        vkDestroySemaphore(device_, out_semaphore, nullptr);
        out_semaphore = VK_NULL_HANDLE;
        return false;
    }
    out_fd = static_cast<int>(reinterpret_cast<intptr_t>(win32_handle));
#else
    VkSemaphoreGetFdInfoKHR gfd_info{};
    gfd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gfd_info.semaphore = out_semaphore;
    gfd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (vkGetSemaphoreFdKHR(device_, &gfd_info, &out_fd) != VK_SUCCESS) {
        utility::LogWarning("GaussianSplat Vulkan: vkGetSemaphoreFdKHR failed");
        vkDestroySemaphore(device_, out_semaphore, nullptr);
        out_semaphore = VK_NULL_HANDLE;
        return false;
    }
#endif
    return true;
}

bool GaussianSplatVulkanInteropContext::ImportFDIntoGLSemaphore(
        int fd, std::uint32_t& out_gl_semaphore) const {
    glGenSemaphoresEXT(1, &out_gl_semaphore);
    if (out_gl_semaphore == 0) {
        utility::LogWarning(
                "GaussianSplat GL: glGenSemaphoresEXT returned 0");
        return false;
    }
#if defined(_WIN32)
    const HANDLE win32_handle =
            reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
    glImportSemaphoreWin32HandleEXT(out_gl_semaphore,
                                    GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                                    win32_handle);
#else
    glImportSemaphoreFdEXT(out_gl_semaphore, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
    // FD ownership transferred to GL.
#endif
    const GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        utility::LogWarning(
                "GaussianSplat GL: semaphore import error 0x{:x}",
                static_cast<unsigned>(err));
        glDeleteSemaphoresEXT(1, &out_gl_semaphore);
        out_gl_semaphore = 0;
        return false;
    }
    return true;
}

bool GaussianSplatVulkanInteropContext::CreateSemaphorePair(
        SharedSemaphoreDesc& out_gl_to_vk, SharedSemaphoreDesc& out_vk_to_gl) {
    if (!initialized_ || !gl_extensions_ok_) return false;

    // GL→VK semaphore: GL signals, Vulkan waits.
    int fd0 = -1;
    if (!CreateExportableSemaphore(out_gl_to_vk.vk_semaphore, fd0)) {
        return false;
    }
    if (!ImportFDIntoGLSemaphore(fd0, out_gl_to_vk.gl_semaphore)) {
        vkDestroySemaphore(device_, out_gl_to_vk.vk_semaphore, nullptr);
        out_gl_to_vk.vk_semaphore = VK_NULL_HANDLE;
        return false;
    }
    out_gl_to_vk.direction = SemaphoreDirection::kGLtoVK;

    // VK→GL semaphore: Vulkan signals, GL waits.
    int fd1 = -1;
    if (!CreateExportableSemaphore(out_vk_to_gl.vk_semaphore, fd1)) {
        DestroySemaphore(out_gl_to_vk);
        return false;
    }
    if (!ImportFDIntoGLSemaphore(fd1, out_vk_to_gl.gl_semaphore)) {
        vkDestroySemaphore(device_, out_vk_to_gl.vk_semaphore, nullptr);
        out_vk_to_gl.vk_semaphore = VK_NULL_HANDLE;
        DestroySemaphore(out_gl_to_vk);
        return false;
    }
    out_vk_to_gl.direction = SemaphoreDirection::kVKtoGL;
    return true;
}

void GaussianSplatVulkanInteropContext::DestroySemaphore(
        SharedSemaphoreDesc& desc) {
    if (!desc.IsValid()) return;
    if (desc.gl_semaphore != 0) {
        glDeleteSemaphoresEXT(1, &desc.gl_semaphore);
        desc.gl_semaphore = 0;
    }
    if (desc.vk_semaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, desc.vk_semaphore, nullptr);
        desc.vk_semaphore = VK_NULL_HANDLE;
    }
    desc = SharedSemaphoreDesc{};
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
