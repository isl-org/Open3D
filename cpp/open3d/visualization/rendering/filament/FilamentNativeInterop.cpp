// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"

#if !defined(__APPLE__) && __has_include(<backend/platforms/VulkanPlatform.h>)
#include <backend/platforms/VulkanPlatform.h>
#define OPEN3D_HAS_FILAMENT_VULKAN_PLATFORM 1
#endif

namespace open3d {
namespace visualization {
namespace rendering {

FilamentVulkanNativeHandles GetFilamentVulkanNativeHandles(
        filament::backend::Platform* platform) {
    FilamentVulkanNativeHandles handles;

#if defined(OPEN3D_HAS_FILAMENT_VULKAN_PLATFORM)
    auto* vulkan_platform =
            dynamic_cast<filament::backend::VulkanPlatform*>(platform);
    if (!vulkan_platform) {
        return handles;
    }

    handles.instance =
            reinterpret_cast<std::uintptr_t>(vulkan_platform->getInstance());
    handles.physical_device = reinterpret_cast<std::uintptr_t>(
            vulkan_platform->getPhysicalDevice());
    handles.device =
            reinterpret_cast<std::uintptr_t>(vulkan_platform->getDevice());
    handles.graphics_queue = reinterpret_cast<std::uintptr_t>(
            vulkan_platform->getGraphicsQueue());
    handles.graphics_queue_family_index =
            vulkan_platform->getGraphicsQueueFamilyIndex();
    handles.graphics_queue_index = vulkan_platform->getGraphicsQueueIndex();
    handles.valid = handles.instance != 0 && handles.physical_device != 0 &&
                    handles.device != 0 && handles.graphics_queue != 0;
#else
    (void)platform;
#endif

    return handles;
}

#if !defined(__APPLE__)
FilamentMetalNativeHandles GetFilamentMetalNativeHandles(
        filament::backend::Platform* platform) {
    (void)platform;
    return {};
}
#endif

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d