// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

/// @cond
namespace filament {
namespace backend {
class Platform;
}  // namespace backend
}  // namespace filament
/// @endcond

namespace open3d {
namespace visualization {
namespace rendering {

struct FilamentVulkanNativeHandles {
    std::uintptr_t instance = 0;
    std::uintptr_t physical_device = 0;
    std::uintptr_t device = 0;
    std::uintptr_t graphics_queue = 0;
    std::uint32_t graphics_queue_family_index = ~std::uint32_t(0);
    std::uint32_t graphics_queue_index = ~std::uint32_t(0);
    bool valid = false;
};

struct FilamentMetalNativeHandles {
    std::uintptr_t device = 0;
    std::uintptr_t command_queue = 0;
    bool valid = false;
};

FilamentVulkanNativeHandles GetFilamentVulkanNativeHandles(
        filament::backend::Platform* platform);

FilamentMetalNativeHandles GetFilamentMetalNativeHandles(
        filament::backend::Platform* platform);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d