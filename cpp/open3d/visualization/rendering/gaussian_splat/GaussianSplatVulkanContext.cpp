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
// Adapter selection: score suitable devices by type: discrete GPU (200),
// integrated GPU (100), and CPU/software renderer (0). The first device with
// the highest score is selected.
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

// These must match settings in Filament VulkanMemory.h
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_mem_alloc.h"

// Include platform-specific Vulkan surface headers before vulkan_raii.hpp
// (the raii header only includes the core vulkan.hpp, not platform extensions).
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#endif

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

bool HasExtension(const std::vector<vk::ExtensionProperties>& available,
                  const char* name) {
    return std::any_of(available.begin(), available.end(),
                       [name](const vk::ExtensionProperties& ext) {
                           return std::strcmp(ext.extensionName, name) == 0;
                       });
}

bool HasAllExtensions(const std::vector<vk::ExtensionProperties>& available,
                      const char* const* required,
                      std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!HasExtension(available, required[i])) return false;
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

// Device suitability score; the highest-scoring device is selected.
constexpr int kScoreDiscreteGpu = 200;
constexpr int kScoreIntegratedGpu = 100;
constexpr int kScoreSoftware = 0;

/// A device is usable only if it exposes a single queue family with both
/// graphics (Filament) and compute (GS), avoiding queue-family ownership
/// transfers for the shared images.
bool FindGraphicsComputeQueueFamily(
        const std::vector<vk::QueueFamilyProperties>& families,
        std::uint32_t& out_index) {
    constexpr auto kNeeded =
            vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if ((families[i].queueFlags & kNeeded) == kNeeded) {
            out_index = i;
            return true;
        }
    }
    return false;
}

/// Devices meeting all hard requirements are ranked by type. Software
/// renderers score lowest so a real GPU always wins when present.
int ScorePhysicalDevice(const vk::raii::PhysicalDevice& device) {
    const auto props = device.getProperties();
    if (props.apiVersion < VK_API_VERSION_1_3) return -1;

    std::uint32_t queue_family = 0;
    if (!FindGraphicsComputeQueueFamily(device.getQueueFamilyProperties(),
                                        queue_family)) {
        return -1;
    }
    if (!HasAllExtensions(device.enumerateDeviceExtensionProperties(),
                          kRequiredDeviceExts,
                          std::size(kRequiredDeviceExts))) {
        return -1;
    }
    // The shared colour image is written as an optimal-tiled storage image.
    const auto color_features =
            device.getFormatProperties(vk::Format::eR16G16B16A16Sfloat)
                    .optimalTilingFeatures;
    if (!(color_features & vk::FormatFeatureFlagBits::eStorageImage)) {
        return -1;
    }

    const std::string name(props.deviceName.data());
    const bool software = name.find("llvmpipe") != std::string::npos ||
                          name.find("SwiftShader") != std::string::npos ||
                          name.find("WARP") != std::string::npos;
    if (software) return kScoreSoftware;
    return props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
                   ? kScoreDiscreteGpu
                   : kScoreIntegratedGpu;
}

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
    debug_utils_enabled_ =
            HasExtension(context_.enumerateInstanceExtensionProperties(),
                         VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debug_utils_enabled_) {
        inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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

    std::size_t best = phys_devices.size();
    int best_score = -1;
    for (std::size_t i = 0; i < phys_devices.size(); ++i) {
        const int score = ScorePhysicalDevice(phys_devices[i]);
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }

    if (best_score < 0) {
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
    // ScorePhysicalDevice() already verified such a family exists.
    const auto qfams = physical_device_.getQueueFamilyProperties();
    FindGraphicsComputeQueueFamily(qfams, graphics_queue_family_);

    // At least 2 queues preferred (index 0 = GS, index 1 = Filament).
    // Fall back to index 0 for both on single-queue families, where mutual
    // exclusion is provided by the existing flushAndWait() bracketing.
    // Vulkan permits at most one DeviceQueueCreateInfo per queue family, so
    // both indices are requested through a single record.
    const std::uint32_t queue_count =
            std::min(2u, qfams[graphics_queue_family_].queueCount);
    filament_queue_index_ = queue_count - 1u;

    const float priorities[2] = {1.0f, 1.0f};
    std::vector<vk::DeviceQueueCreateInfo> qcis;
    qcis.push_back({{}, graphics_queue_family_, queue_count, priorities});

    // Enable synchronization2 (required by GS compute pipeline).
    auto feat =
            physical_device_.getFeatures2<vk::PhysicalDeviceFeatures2,
                                          vk::PhysicalDeviceVulkan13Features>();
    if (feat.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 !=
        VK_TRUE) {
        last_error_ = "Device does not support synchronization2";
        return false;
    }

    // Enable only the core features Filament needs, plus synchronization2.
    const auto& available = feat.get<vk::PhysicalDeviceFeatures2>().features;
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan13Features>
            enabled_feat;
    auto& enabled = enabled_feat.get<vk::PhysicalDeviceFeatures2>().features;
    enabled.samplerAnisotropy = available.samplerAnisotropy;
    enabled.textureCompressionETC2 = available.textureCompressionETC2;
    enabled.textureCompressionBC = available.textureCompressionBC;
    enabled.shaderClipDistance = available.shaderClipDistance;
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

    // ---- Filament shared context -----------------------------------------
    shared_context_.instance = GetVkInstance();
    shared_context_.physical_device = GetPhysicalDevice();
    shared_context_.logical_device = GetDevice();
    shared_context_.graphics_queue_family_index = graphics_queue_family_;
    shared_context_.graphics_queue_index = filament_queue_index_;

    initialized_ = true;
    utility::LogDebug(
            "GaussianSplat VulkanContext: ready '{}' (fam={} gs_q=0 fil_q={})",
            pd_props.deviceName.data(), graphics_queue_family_,
            filament_queue_index_);
    return true;
}

bool GaussianSplatVulkanContext::IsSoftwareDevice() const {
    if (!initialized_) return false;
    const auto properties = physical_device_.getProperties();
    return properties.deviceType == vk::PhysicalDeviceType::eCpu;
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
    shared_context_ = FilamentVulkanSharedContext{};
    graphics_queue_family_ = UINT32_MAX;
    filament_queue_index_ = 1;
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

bool GaussianSplatVulkanContext::SupportsOptimalStorageImage(
        VkFormat vk_format) const {
    const auto properties = physical_device_.getFormatProperties(
            static_cast<vk::Format>(vk_format));
    return (properties.optimalTilingFeatures &
            vk::FormatFeatureFlagBits::eStorageImage) !=
           vk::FormatFeatureFlags{};
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

    const vk::ImageCreateInfo ici{{},
                                  vk::ImageType::e2D,
                                  static_cast<vk::Format>(vk_format),
                                  {width, height, 1},
                                  1,
                                  1,
                                  vk::SampleCountFlagBits::e1,
                                  vk::ImageTiling::eOptimal,
                                  static_cast<vk::ImageUsageFlags>(usage),
                                  vk::SharingMode::eExclusive};

    // RAII handles release the image and memory automatically if a later step
    // throws; ownership is released to the plain handles only on success.
    try {
        vk::raii::Image image(device_, ici);
        const vk::MemoryRequirements reqs = image.getMemoryRequirements();
        const std::uint32_t mem_type = FindMemoryType(
                reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type == UINT32_MAX) {
            utility::LogWarning("VulkanContext: no mem type for '{}'", label);
            return desc;
        }
        vk::raii::DeviceMemory memory(
                device_, vk::MemoryAllocateInfo{reqs.size, mem_type});
        image.bindMemory(*memory, 0);

        desc.vk_image = static_cast<VkImage>(image.release());
        desc.vk_memory = static_cast<VkDeviceMemory>(memory.release());
    } catch (const vk::SystemError& e) {
        utility::LogWarning("VulkanContext: image creation for '{}': {}", label,
                            e.what());
        desc = VkImageDesc{};
    }
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