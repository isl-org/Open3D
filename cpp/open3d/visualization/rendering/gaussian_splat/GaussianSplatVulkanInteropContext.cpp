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

// Provide storage for vulkan-hpp's global dynamic dispatch loader (exactly one
// translation unit in the program must define this macro).
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <GL/glew.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "open3d/utility/Logging.h"

// Initialize VMA in this single translation unit.
// VK_NO_PROTOTYPES is already defined via vulkan_raii.hpp (included in the
// context header), so VMA must use VmaVulkanFunctions rather than calling
// Vulkan symbols directly.
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

// ---------------------------------------------------------------------------
// Required Vulkan device extensions for interop
// ---------------------------------------------------------------------------
namespace {

#if defined(_WIN32)
constexpr const char* kRequiredDeviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
};
#else
constexpr const char* kRequiredDeviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
};
#endif

bool ShouldEnableDebugUtils() {
    const char* value = std::getenv("OPEN3D_VULKAN_DEBUG_UTILS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// Returns true when all extensions in 'required' are present in
/// 'available'.  Sets out_missing to the first missing extension name.
bool CheckExtensions(const std::vector<vk::ExtensionProperties>& available,
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
int ScoreDevice(const vk::raii::PhysicalDevice& dev) {
    // Check for a compute queue.
    const auto qfams = dev.getQueueFamilyProperties();
    bool has_compute = false;
    for (const auto& qf : qfams) {
        if (qf.queueFlags & vk::QueueFlagBits::eCompute) {
            has_compute = true;
            break;
        }
    }
    if (!has_compute) return -1;

    // Check device extensions.
    const auto exts = dev.enumerateDeviceExtensionProperties();

    std::string missing;
    const bool ok =
            CheckExtensions(exts, kRequiredDeviceExtensions,
                            std::size(kRequiredDeviceExtensions), missing);
    if (!ok) return -1;

    // Shaders are compiled for Vulkan 1.3 (SPIR-V 1.6); reject older devices.
    const auto props = dev.getProperties();
    if (props.apiVersion < VK_API_VERSION_1_3) return -1;

    // Score device type.
    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return 2;
    if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) return 1;
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

    // Initialize the global dynamic dispatcher with vkGetInstanceProcAddr
    // obtained from the Vulkan loader via vulkan-hpp's DynamicLoader utility.
    // This loads instance-independent entry points (e.g. vkCreateInstance).
    try {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(
                vk::detail::DynamicLoader()
                        .getProcAddress<PFN_vkGetInstanceProcAddr>(
                                "vkGetInstanceProcAddr"));
    } catch (const std::exception& e) {
        last_error_ = std::string("Vulkan loader not found: ") + e.what();
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    if (!CreateInstance()) return false;

    // After instance creation, update the dispatcher with instance-level
    // function pointers (required for physical device enumeration etc.).
    VULKAN_HPP_DEFAULT_DISPATCHER.init(static_cast<vk::Instance>(*instance_));

    if (!SelectPhysicalDevice()) return false;
    if (!CreateLogicalDevice()) return false;

    // After device creation, update the dispatcher with device-level
    // function pointers for push descriptors and other extensions.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(static_cast<vk::Device>(*device_));

    initialized_ = true;
    utility::LogDebug("GaussianSplat VulkanInteropContext: initialized ({})",
                      physical_device_.getProperties().deviceName.data());
    return true;
}

void GaussianSplatVulkanInteropContext::Shutdown() {
    if (!initialized_) return;

    // Wait for idle before releasing sub-objects.  The raw VkDevice is
    // extracted before the raii handle is reset so the call is valid.
    if (*device_) {
        try {
            device_.waitIdle();
        } catch (const vk::SystemError& e) {
            utility::LogWarning(
                    "GaussianSplat VulkanInteropContext: waitIdle during "
                    "shutdown failed: {}",
                    e.what());
        }
    }

    // Resetting in reverse construction order: queue → device → physical →
    // instance → context. vk::raii handles call vkDestroy* in their dtors.
    compute_queue_ = vk::raii::Queue{nullptr};
    device_ = vk::raii::Device{nullptr};
    physical_device_ = vk::raii::PhysicalDevice{nullptr};
    instance_ = vk::raii::Instance{nullptr};
    // context_ is a value member; its dtor unloads the Vulkan loader.

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
            utility::LogWarning("GaussianSplat Vulkan interop: {}",
                                last_error_);
            return false;
        }
    }
    gl_extensions_ok_ = true;
    utility::LogDebug(
            "GaussianSplat Vulkan interop: all GL extensions present");
    return true;
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateInstance() {
    // Query instance extensions so we can enable VK_EXT_debug_utils only when
    // explicitly requested for validation/profiling runs.
    const auto available_exts = context_.enumerateInstanceExtensionProperties();

    std::vector<const char*> extensions;
    if (ShouldEnableDebugUtils()) {
        for (const auto& ext : available_exts) {
            if (std::strcmp(ext.extensionName,
                            VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                debug_utils_enabled_ = true;
                break;
            }
        }
    }

    // Shaders are compiled for --target-env vulkan1.3 (SPIR-V 1.6).
    // The instance must advertise 1.3 or vkCreateShaderModule rejects them.
    vk::ApplicationInfo app_info{
            "Open3D GaussianSplat",  // pApplicationName
            1,                       // applicationVersion
            nullptr,                 // pEngineName
            0,                       // engineVersion
            VK_API_VERSION_1_3,      // apiVersion
    };
    vk::InstanceCreateInfo ci({}, &app_info, {}, extensions);

    try {
        instance_ = context_.createInstance(ci);
    } catch (const vk::SystemError& e) {
        last_error_ = std::string("vkCreateInstance failed: ") + e.what();
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::SelectPhysicalDevice() {
    auto devices = instance_.enumeratePhysicalDevices();
    if (devices.empty()) {
        last_error_ = "No Vulkan-capable devices found";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    int best_score = -1;
    std::size_t best_idx = devices.size();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const int score = ScoreDevice(devices[i]);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx == devices.size()) {
        last_error_ =
                "No suitable Vulkan device found with required interop "
                "extensions. Required "
                "extensions: " VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
                ", " VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    physical_device_ = std::move(devices[best_idx]);

    const auto props = physical_device_.getProperties();
    memory_props_ = static_cast<VkPhysicalDeviceMemoryProperties>(
            physical_device_.getMemoryProperties());
    utility::LogDebug("GaussianSplat Vulkan: selected device '{}'",
                      props.deviceName.data());
    return true;
}

// ---------------------------------------------------------------------------
// Logical device and compute queue
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateLogicalDevice() {
    const auto qfams = physical_device_.getQueueFamilyProperties();

    // Prefer a queue family that supports both GRAPHICS and COMPUTE — the
    // same hardware engine (RCS on Intel, GFX on AMD) that OpenGL
    // GL_ARB_gl_spirv uses. Dedicated compute-only queues (CCS on Intel) can
    // behave differently for the same SPIR-V shaders and may hang on some
    // driver/hardware combos. Fall back to a compute-only queue only when no
    // graphics+compute family exists.
    std::uint32_t graphics_compute_family = UINT32_MAX;
    std::uint32_t compute_only_family = UINT32_MAX;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(qfams.size());
         ++i) {
        const auto flags = qfams[i].queueFlags;
        if ((flags & vk::QueueFlagBits::eCompute) &&
            (flags & vk::QueueFlagBits::eGraphics)) {
            if (graphics_compute_family == UINT32_MAX)
                graphics_compute_family = i;
        } else if ((flags & vk::QueueFlagBits::eCompute) &&
                   compute_only_family == UINT32_MAX) {
            compute_only_family = i;
        }
    }
    compute_queue_family_ = (graphics_compute_family != UINT32_MAX)
                                    ? graphics_compute_family
                                    : compute_only_family;
    if (compute_queue_family_ == UINT32_MAX) {
        last_error_ = "No compute queue family found";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    const float queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo qci({}, compute_queue_family_, 1,
                                  &queue_priority);

    // Request only the Vulkan 1.3 features this backend actually uses.
    // The command path relies on vkQueueSubmit2 / vkCmdPipelineBarrier2, so
    // synchronization2 must be enabled. Subgroup variants are selected from
    // properties and supported-operation bits only.
    auto supported_features =
            physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2,
                                          vk::PhysicalDeviceVulkan13Features>();
    const auto& supported_vulkan13 =
            supported_features.get<vk::PhysicalDeviceVulkan13Features>();
    if (supported_vulkan13.synchronization2 != VK_TRUE) {
        last_error_ =
                "Selected Vulkan device does not support synchronization2";
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }

    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan13Features>
            enabled_features;
    enabled_features.get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 = VK_TRUE;

    vk::DeviceCreateInfo dci({}, qci, {}, kRequiredDeviceExtensions);
    dci.pNext = &enabled_features.get<vk::PhysicalDeviceFeatures2>();

    try {
        device_ = physical_device_.createDevice(dci);
    } catch (const vk::SystemError& e) {
        last_error_ = std::string("vkCreateDevice failed: ") + e.what();
        utility::LogWarning("GaussianSplat Vulkan: {}", last_error_);
        return false;
    }
    compute_queue_ = device_.getQueue(compute_queue_family_, 0);

    // Query the hardware subgroup size so callers can skip _subgroup shader
    // variants when the device subgroup size is too small (e.g. Intel Xe2=8).
    auto props11 =
            physical_device_
                    .getProperties2<vk::PhysicalDeviceProperties2,
                                    vk::PhysicalDeviceVulkan11Properties>();
    const auto& subgroup_props =
            props11.get<vk::PhysicalDeviceVulkan11Properties>();
    subgroup_size_ = subgroup_props.subgroupSize;
    subgroup_supported_stages_ =
            static_cast<std::uint32_t>(subgroup_props.subgroupSupportedStages);
    subgroup_supported_operations_ = static_cast<std::uint32_t>(
            subgroup_props.subgroupSupportedOperations);

    utility::LogDebug(
            "GaussianSplat Vulkan: compute queue family {}, "
            "subgroupSize={}, subgroupStages=0x{:x}, "
            "subgroupOps=0x{:x}, synchronization2={}",
            compute_queue_family_, subgroup_size_, subgroup_supported_stages_,
            subgroup_supported_operations_,
            supported_vulkan13.synchronization2 == VK_TRUE);
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
    const vk::Format format = static_cast<vk::Format>(vk_format);
    const vk::ImageUsageFlags vk_usage =
            static_cast<vk::ImageUsageFlags>(usage);

#if defined(_WIN32)
    constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType =
            vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
#else
    constexpr vk::ExternalMemoryHandleTypeFlagBits kHandleType =
            vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd;
#endif

    // Chain external-memory info into the image create info.
    vk::StructureChain<vk::ImageCreateInfo, vk::ExternalMemoryImageCreateInfo>
            img_chain{
                    vk::ImageCreateInfo{
                            {},
                            vk::ImageType::e2D,
                            format,
                            {width, height, 1},
                            1,
                            1,
                            vk::SampleCountFlagBits::e1,
                            vk::ImageTiling::eOptimal,
                            vk_usage,
                            vk::SharingMode::eExclusive,
                            {},
                            vk::ImageLayout::eUndefined,
                    },
                    vk::ExternalMemoryImageCreateInfo{kHandleType},
            };

    // Use the non-RAII vk::Device wrapper: AllocateExportableImage manages raw
    // VkImage/VkDeviceMemory lifetimes explicitly (exported to GL via FD).
    const vk::Device dev(*device_);

    vk::Image image;
    try {
        image = dev.createImage(img_chain.get<vk::ImageCreateInfo>());
    } catch (const vk::SystemError& e) {
        utility::LogWarning("GaussianSplat Vulkan: vkCreateImage failed: {}",
                            e.what());
        return false;
    }

    const vk::MemoryRequirements mem_req =
            dev.getImageMemoryRequirements(image);
    const std::uint32_t mem_type =
            FindMemoryType(mem_req.memoryTypeBits,
                           static_cast<VkMemoryPropertyFlags>(
                                   vk::MemoryPropertyFlagBits::eDeviceLocal));
    if (mem_type == UINT32_MAX) {
        utility::LogWarning(
                "GaussianSplat Vulkan: no suitable device-local memory type");
        dev.destroyImage(image);
        return false;
    }

    // Chain dedicated-alloc and export-alloc info into vk::MemoryAllocateInfo.
    vk::StructureChain<vk::MemoryAllocateInfo, vk::ExportMemoryAllocateInfo,
                       vk::MemoryDedicatedAllocateInfo>
            mem_chain{
                    vk::MemoryAllocateInfo{mem_req.size, mem_type},
                    vk::ExportMemoryAllocateInfo{kHandleType},
                    vk::MemoryDedicatedAllocateInfo{image},
            };

    vk::DeviceMemory memory;
    try {
        memory = dev.allocateMemory(mem_chain.get<vk::MemoryAllocateInfo>());
    } catch (const vk::SystemError& e) {
        utility::LogWarning("GaussianSplat Vulkan: vkAllocateMemory failed: {}",
                            e.what());
        dev.destroyImage(image);
        return false;
    }

    try {
        (void)dev.bindImageMemory(image, memory, 0);
    } catch (const vk::SystemError& e) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkBindImageMemory failed: {}", e.what());
        dev.freeMemory(memory);
        dev.destroyImage(image);
        return false;
    }

    // Export the memory as a platform handle.
#if defined(_WIN32)
    HANDLE win32_handle = nullptr;
    try {
        win32_handle = dev.getMemoryWin32HandleKHR(
                {memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
    } catch (const vk::SystemError& e) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkGetMemoryWin32HandleKHR failed: {}",
                e.what());
        dev.freeMemory(memory);
        dev.destroyImage(image);
        return false;
    }
    out_fd = static_cast<int>(reinterpret_cast<intptr_t>(win32_handle));
#else
    try {
        out_fd = dev.getMemoryFdKHR(
                {memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd});
    } catch (const vk::SystemError& e) {
        utility::LogWarning("GaussianSplat Vulkan: vkGetMemoryFdKHR failed: {}",
                            e.what());
        dev.freeMemory(memory);
        dev.destroyImage(image);
        return false;
    }
#endif

    out_image = static_cast<VkImage>(image);
    out_memory = static_cast<VkDeviceMemory>(memory);
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
    const HANDLE win32_handle =
            reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
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
    glTextureStorageMem2DEXT(
            out_gl_texture, 1, gl_internal_format, static_cast<GLsizei>(width),
            static_cast<GLsizei>(height), out_gl_memory_object, /*offset=*/0);

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
    {
        // Probe memory requirements using a temporary image (no exportable
        // memory yet) to determine the actual allocation size for GL import.
        vk::ImageCreateInfo probe_ci{{},
                                     vk::ImageType::e2D,
                                     static_cast<vk::Format>(vk_fmt),
                                     {width, height, 1},
                                     1,
                                     1,
                                     vk::SampleCountFlagBits::e1,
                                     vk::ImageTiling::eOptimal,
                                     static_cast<vk::ImageUsageFlags>(usage)};
        try {
            vk::Image probe_img = vk::Device(*device_).createImage(probe_ci);
            vk::Device(*device_).destroyImage(probe_img);
        } catch (...) {
            // Ignore probe failures; AllocateExportableImage will report.
        }
    }

    bool ok = AllocateExportableImage(width, height, vk_fmt, usage,
                                      desc.vk_image, desc.vk_memory, export_fd);
    if (!ok) return desc;

    // Query actual memory size for the GL import call.
    const vk::MemoryRequirements actual_req =
            vk::Device(*device_).getImageMemoryRequirements(
                    static_cast<vk::Image>(desc.vk_image));

    ok = ImportFDIntoGL(export_fd, width, height,
                        static_cast<VkDeviceSize>(actual_req.size),
                        interop_format, desc.gl_memory_object, desc.gl_texture);
    if (!ok) {
        vk::Device(*device_).freeMemory(
                static_cast<vk::DeviceMemory>(desc.vk_memory));
        vk::Device(*device_).destroyImage(
                static_cast<vk::Image>(desc.vk_image));
        desc = SharedImageDesc{};
        return desc;
    }

    desc.width = width;
    desc.height = height;
    desc.format = interop_format;
    desc.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (label) {
        utility::LogDebug(
                "GaussianSplat Vulkan: created shared image '{}' "
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
        vk::Device(*device_).destroyImage(
                static_cast<vk::Image>(desc.vk_image));
        desc.vk_image = VK_NULL_HANDLE;
    }
    if (desc.vk_memory != VK_NULL_HANDLE) {
        vk::Device(*device_).freeMemory(
                static_cast<vk::DeviceMemory>(desc.vk_memory));
        desc.vk_memory = VK_NULL_HANDLE;
    }
    desc = SharedImageDesc{};
}

// ---------------------------------------------------------------------------
// Semaphore lifecycle
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanInteropContext::CreateExportableSemaphore(
        VkSemaphore& out_semaphore, int& out_fd) const {
#if defined(_WIN32)
    constexpr vk::ExternalSemaphoreHandleTypeFlagBits kSemHandleType =
            vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
    constexpr vk::ExternalSemaphoreHandleTypeFlagBits kSemHandleType =
            vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif

    // Chain export info into the semaphore create info.
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::ExportSemaphoreCreateInfo>
            chain{{}, {kSemHandleType}};

    // Use the non-RAII vk::Device wrapper: semaphore lifetime is managed
    // manually here (exported to GL via FD, raw VkSemaphore in
    // SharedSemaphoreDesc).
    const vk::Device sdev(*device_);

    vk::Semaphore semaphore;
    try {
        semaphore = sdev.createSemaphore(chain.get<vk::SemaphoreCreateInfo>());
    } catch (const vk::SystemError& e) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkCreateSemaphore failed: {}", e.what());
        return false;
    }

#if defined(_WIN32)
    HANDLE win32_handle = nullptr;
    try {
        win32_handle = sdev.getSemaphoreWin32HandleKHR(
                {semaphore,
                 vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32});
    } catch (const vk::SystemError& e) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkGetSemaphoreWin32HandleKHR failed: {}",
                e.what());
        sdev.destroySemaphore(semaphore);
        return false;
    }
    out_fd = static_cast<int>(reinterpret_cast<intptr_t>(win32_handle));
#else
    try {
        out_fd = sdev.getSemaphoreFdKHR(
                {semaphore,
                 vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
    } catch (const vk::SystemError& e) {
        utility::LogWarning(
                "GaussianSplat Vulkan: vkGetSemaphoreFdKHR failed: {}",
                e.what());
        sdev.destroySemaphore(semaphore);
        return false;
    }
#endif
    out_semaphore = static_cast<VkSemaphore>(semaphore);
    return true;
}

bool GaussianSplatVulkanInteropContext::ImportFDIntoGLSemaphore(
        int fd, std::uint32_t& out_gl_semaphore) const {
    glGenSemaphoresEXT(1, &out_gl_semaphore);
    if (out_gl_semaphore == 0) {
        utility::LogWarning("GaussianSplat GL: glGenSemaphoresEXT returned 0");
        return false;
    }
#if defined(_WIN32)
    const HANDLE win32_handle =
            reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
    glImportSemaphoreWin32HandleEXT(
            out_gl_semaphore, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32_handle);
#else
    glImportSemaphoreFdEXT(out_gl_semaphore, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
    // FD ownership transferred to GL.
#endif
    const GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        utility::LogWarning("GaussianSplat GL: semaphore import error 0x{:x}",
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
        vk::Device(*device_).destroySemaphore(
                static_cast<vk::Semaphore>(out_gl_to_vk.vk_semaphore));
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
        vk::Device(*device_).destroySemaphore(
                static_cast<vk::Semaphore>(out_vk_to_gl.vk_semaphore));
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
        vk::Device(*device_).destroySemaphore(
                static_cast<vk::Semaphore>(desc.vk_semaphore));
        desc.vk_semaphore = VK_NULL_HANDLE;
    }
    desc = SharedSemaphoreDesc{};
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
