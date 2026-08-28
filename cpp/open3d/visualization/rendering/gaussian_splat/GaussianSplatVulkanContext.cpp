// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Headless Vulkan context for Gaussian splatting compute (Linux & Windows).
//
// Owns the VkInstance/VkPhysicalDevice/VkDevice used by both the GS compute
// pipeline and Filament's Vulkan backend.  The same VkDevice is shared via
// VulkanPlatform::VulkanSharedContext (passed as Engine::create()'s
// sharedContext argument), eliminating GL-Vulkan interop entirely.
//
// Adapter selection: the Vulkan loader picks the first suitable physical
// device (the platform default).  Use VK_LOADER_DEVICE_SELECT /
// VK_ICD_FILENAMES / VK_INSTANCE_LAYERS environment variables to override.
//
// Sequence:
//   1. Initialize()    → VkInstance, VkPhysicalDevice, VkDevice, 2 queues
//   2. CreateImage()   → VkImage + VkDeviceMemory for GS depth/color
//   3. VulkanSharedContext → passed to Engine::create()
//   4. (Filament) importTextureR() wraps those VkImages
//   5. Shutdown()      → after Engine::destroy()

#if defined(_WIN32)
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanContext.h"

#if !defined(__APPLE__)

// Include platform-specific Vulkan surface headers before vulkan_raii.hpp
// (the raii header only includes the core vulkan.hpp, not platform extensions).
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

bool ShouldEnableDebugUtils() {
    const char* value = std::getenv("OPEN3D_VULKAN_DEBUG_UTILS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

/// Returns true when all extensions in required[0..n-1] are present in
/// available.  Sets out_missing to the first missing name.
bool CheckExtensions(const std::vector<vk::ExtensionProperties>& available,
                     const char* const* required,
                     size_t count,
                     std::string& out_missing) {
    for (size_t i = 0; i < count; ++i) {
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

// Required device extensions.  VK_KHR_push_descriptor is needed by the GS
// compute pipeline.  VK_KHR_swapchain is unconditionally requested by
// Filament (even in the shared-context path; see
// VulkanPlatform::createLogicalDeviceAndQueues).
constexpr const char* kRequiredDeviceExts[] = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Instance extensions required by Filament's surface creation.
constexpr const char* kRequiredInstanceExts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#if defined(_WIN32)
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#else
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
};

}  // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

GaussianSplatVulkanContext& GaussianSplatVulkanContext::GetInstance() {
    // EngineInstance performs the required explicit shutdown after Filament
    // releases its shared-device resources. Avoid a competing static destructor
    // whose order relative to EngineInstance is unspecified across TUs.
    static auto* instance = new GaussianSplatVulkanContext;
    return *instance;
}

GaussianSplatVulkanContext::~GaussianSplatVulkanContext() { Shutdown(); }

// ---------------------------------------------------------------------------
// Initialize / Shutdown
// ---------------------------------------------------------------------------

bool GaussianSplatVulkanContext::Initialize() {
    if (initialized_) return true;

    // Load the global dispatcher via the Vulkan loader.
    try {
        VULKAN_HPP_DEFAULT_DISPATCHER.init(
                vk::detail::DynamicLoader()
                        .getProcAddress<PFN_vkGetInstanceProcAddr>(
                                "vkGetInstanceProcAddr"));
    } catch (const std::exception& e) {
        last_error_ = std::string("Vulkan loader not found: ") + e.what();
        utility::LogWarning("GaussianSplat VulkanContext: {}", last_error_);
        return false;
    }

    // ---- Instance --------------------------------------------------------
    std::vector<const char*> inst_exts(std::begin(kRequiredInstanceExts),
                                       std::end(kRequiredInstanceExts));
    if (ShouldEnableDebugUtils()) {
        inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        debug_utils_enabled_ = true;
    }

    vk::ApplicationInfo app_info{"Open3D GS", 1, "Open3D", 1,
                                 VK_API_VERSION_1_3};
    vk::InstanceCreateInfo ici({}, &app_info, {}, inst_exts);
    try {
        instance_ = context_.createInstance(ici);
    } catch (const vk::SystemError& e) {
        last_error_ = std::string("vkCreateInstance: ") + e.what();
        utility::LogWarning("GaussianSplat VulkanContext: {}", last_error_);
        return false;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(static_cast<vk::Instance>(*instance_));

    // ---- Physical device -------------------------------------------------
    auto phys_devices = instance_.enumeratePhysicalDevices();
    if (phys_devices.empty()) {
        last_error_ = "No Vulkan-capable devices";
        return false;
    }

    // Pick the first device that meets requirements.
    // The user can override via VK_LOADER_DEVICE_SELECT etc.
    std::size_t best = phys_devices.size();
    for (std::size_t i = 0; i < phys_devices.size(); ++i) {
        const auto& pd = phys_devices[i];
        const auto props = pd.getProperties();
        if (props.apiVersion < VK_API_VERSION_1_3) continue;

        // Need a graphics+compute queue family for Filament + GS on the
        // same family (no queue-family ownership transfers).
        const auto qfams = pd.getQueueFamilyProperties();
        bool has_gr_cp = false;
        for (const auto& qf : qfams) {
            if ((qf.queueFlags & vk::QueueFlagBits::eGraphics) &&
                (qf.queueFlags & vk::QueueFlagBits::eCompute)) {
                has_gr_cp = true;
                break;
            }
        }
        if (!has_gr_cp) continue;

        std::string missing;
        if (!CheckExtensions(pd.enumerateDeviceExtensionProperties(),
                             kRequiredDeviceExts,
                             std::size(kRequiredDeviceExts), missing)) {
            continue;
        }

        // Skip software renderers unless they're the only option.
        const std::string name(props.deviceName.data());
        bool software = name.find("llvmpipe") != std::string::npos ||
                        name.find("SwiftShader") != std::string::npos ||
                        name.find("WARP") != std::string::npos;
        if (software && i + 1 < phys_devices.size()) continue;

        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            best = i;  // discrete GPU is best
            break;
        }
        best = i;
    }

    if (best == phys_devices.size()) {
        last_error_ =
                "No suitable Vulkan device (need Vulkan 1.3, graphics+compute "
                "queue family, VK_KHR_push_descriptor, VK_KHR_swapchain)";
        utility::LogWarning("GaussianSplat VulkanContext: {}", last_error_);
        return false;
    }
    physical_device_ = std::move(phys_devices[best]);
    const auto pd_props = physical_device_.getProperties();
    memory_props_ = static_cast<VkPhysicalDeviceMemoryProperties>(
            physical_device_.getMemoryProperties());
    utility::LogDebug("GaussianSplat VulkanContext: device '{}'",
                      pd_props.deviceName.data());

    // ---- Logical device --------------------------------------------------
    const auto qfams = physical_device_.getQueueFamilyProperties();
    graphics_queue_family_ = UINT32_MAX;
    for (std::uint32_t i = 0; i < (std::uint32_t)qfams.size(); ++i) {
        const auto flags = qfams[i].queueFlags;
        if ((flags & vk::QueueFlagBits::eGraphics) &&
            (flags & vk::QueueFlagBits::eCompute)) {
            graphics_queue_family_ = i;
            break;
        }
    }

    // At least 2 queues preferred (index 0 = GS, index 1 = Filament).
    // Fall back to index 0 for both on single-queue families.
    // Mutual exclusion is provided by existing flushAndWait() bracketing.
    const auto qcount = qfams[graphics_queue_family_].queueCount;
    single_queue_device_ = (qcount < 2);
    filament_queue_index_ = single_queue_device_ ? 0 : 1;

    // Vulkan permits at most one DeviceQueueCreateInfo per queue family.
    // Request both queue indices through one record when available.
    std::vector<float> priorities(single_queue_device_ ? 1 : 2, 1.0f);
    std::vector<vk::DeviceQueueCreateInfo> qcis;
    qcis.push_back({{},
                    graphics_queue_family_,
                    static_cast<uint32_t>(priorities.size()),
                    priorities.data()});

    // Enable synchronization2 (required by GS compute pipeline).
    auto feat =
            physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2,
                                          vk::PhysicalDeviceVulkan13Features>();
    if (feat.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 !=
        VK_TRUE) {
        last_error_ = "Device does not support synchronization2";
        return false;
    }

    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan13Features>
            enabled_feat;
    enabled_feat.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 =
            VK_TRUE;

    vk::DeviceCreateInfo dci({}, qcis, {}, kRequiredDeviceExts);
    dci.pNext = &enabled_feat.get<vk::PhysicalDeviceFeatures2>();
    try {
        device_ = physical_device_.createDevice(dci);
    } catch (const vk::SystemError& e) {
        last_error_ = std::string("vkCreateDevice: ") + e.what();
        return false;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(static_cast<vk::Device>(*device_));

    compute_queue_ = device_.getQueue(graphics_queue_family_, 0);

    // Subgroup properties for shader capability queries.
    auto p11 = physical_device_
                       .getProperties2<vk::PhysicalDeviceProperties2,
                                       vk::PhysicalDeviceVulkan11Properties>();
    const auto& sub = p11.get<vk::PhysicalDeviceVulkan11Properties>();
    subgroup_size_ = sub.subgroupSize;
    subgroup_supported_stages_ = (std::uint32_t)sub.subgroupSupportedStages;
    subgroup_supported_operations_ =
            (std::uint32_t)sub.subgroupSupportedOperations;

    // ---- Filament shared context -----------------------------------------
    shared_context_.instance = GetVkInstance();
    shared_context_.physicalDevice = GetPhysicalDevice();
    shared_context_.logicalDevice = GetDevice();
    shared_context_.graphicsQueueFamilyIndex = graphics_queue_family_;
    shared_context_.graphicsQueueIndex = filament_queue_index_;

    initialized_ = true;
    utility::LogDebug(
            "GaussianSplat VulkanContext: ready '{}' (fam={} gs_q=0 "
            "fil_q={} sq={} sub={})",
            pd_props.deviceName.data(), graphics_queue_family_,
            filament_queue_index_, single_queue_device_, subgroup_size_);
    return true;
}

void GaussianSplatVulkanContext::Shutdown() {
    if (!initialized_) return;
    if (*device_) {
        try {
            device_.waitIdle();
        } catch (const vk::SystemError&) {
        }
    }
    compute_queue_ = vk::raii::Queue{nullptr};
    device_ = vk::raii::Device{nullptr};
    physical_device_ = vk::raii::PhysicalDevice{nullptr};
    instance_ = vk::raii::Instance{nullptr};
    graphics_queue_family_ = UINT32_MAX;
    filament_queue_index_ = 1;
    single_queue_device_ = false;
    initialized_ = false;
    utility::LogDebug("GaussianSplat VulkanContext: shutdown");
}

// ---------------------------------------------------------------------------
// Memory type helper
// ---------------------------------------------------------------------------

std::uint32_t GaussianSplatVulkanContext::FindMemoryType(
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
// Image lifecycle (plain VkImage, no export/import)
// ---------------------------------------------------------------------------

VkImageDesc GaussianSplatVulkanContext::CreateImage(std::uint32_t width,
                                                    std::uint32_t height,
                                                    VkFormat vk_format,
                                                    VkImageUsageFlags usage,
                                                    const char* label) {
    VkImageDesc desc{};
    if (!initialized_) return desc;

    const vk::Device dev(*device_);
    const vk::Format format = static_cast<vk::Format>(vk_format);

    vk::ImageCreateInfo ici{{},
                            vk::ImageType::e2D,
                            format,
                            {width, height, 1},
                            1,
                            1,
                            vk::SampleCountFlagBits::e1,
                            vk::ImageTiling::eOptimal,
                            static_cast<vk::ImageUsageFlags>(usage),
                            vk::SharingMode::eExclusive};
    vk::Image image;
    try {
        image = dev.createImage(ici);
    } catch (const vk::SystemError& e) {
        utility::LogWarning("VulkanContext: vkCreateImage '{}': {}", label,
                            e.what());
        return desc;
    }

    const vk::MemoryRequirements reqs = dev.getImageMemoryRequirements(image);
    const std::uint32_t mem_type = FindMemoryType(
            reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        dev.destroyImage(image);
        utility::LogWarning("VulkanContext: no mem type for '{}'", label);
        return desc;
    }

    vk::DeviceMemory memory;
    try {
        memory =
                dev.allocateMemory(vk::MemoryAllocateInfo{reqs.size, mem_type});
    } catch (const vk::SystemError& e) {
        dev.destroyImage(image);
        utility::LogWarning("VulkanContext: vkAllocateMemory '{}': {}", label,
                            e.what());
        return desc;
    }

    try {
        dev.bindImageMemory(image, memory, 0);
    } catch (const vk::SystemError& e) {
        dev.freeMemory(memory);
        dev.destroyImage(image);
        utility::LogWarning("VulkanContext: vkBindImageMemory '{}': {}", label,
                            e.what());
        return desc;
    }

    desc.vk_image = static_cast<VkImage>(image);
    desc.vk_memory = static_cast<VkDeviceMemory>(memory);
    desc.width = width;
    desc.height = height;
    return desc;
}

void GaussianSplatVulkanContext::DestroyImage(VkImageDesc& desc) {
    if (!desc.IsValid()) return;
    vk::Device dev(*device_);
    dev.destroyImage(static_cast<vk::Image>(desc.vk_image));
    dev.freeMemory(static_cast<vk::DeviceMemory>(desc.vk_memory));
    desc = VkImageDesc{};
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)