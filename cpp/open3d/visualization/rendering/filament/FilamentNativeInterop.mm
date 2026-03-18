// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"

#if __has_include(<backend/platforms/PlatformMetal.h>) && \
        __has_include(<backend/platforms/PlatformMetal-ObjC.h>)
#include <backend/platforms/PlatformMetal.h>
#include <backend/platforms/PlatformMetal-ObjC.h>
#define OPEN3D_HAS_FILAMENT_METAL_PLATFORM 1
#endif

namespace open3d {
namespace visualization {
namespace rendering {

FilamentMetalNativeHandles GetFilamentMetalNativeHandles(
        filament::backend::Platform* platform) {
    FilamentMetalNativeHandles handles;

#if defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
    auto* metal_platform =
            dynamic_cast<filament::backend::PlatformMetal*>(platform);
    if (!metal_platform) {
        return handles;
    }

    filament::backend::MetalDevice device = {nil};
    metal_platform->createDevice(device);
    if (!device.device) {
        return handles;
    }

    filament::backend::MetalCommandQueue command_queue = {nil};
    metal_platform->createCommandQueue(device, command_queue);

    handles.device = reinterpret_cast<std::uintptr_t>((__bridge void*)device.device);
    handles.command_queue = reinterpret_cast<std::uintptr_t>(
            (__bridge void*)command_queue.commandQueue);
    handles.valid = handles.device != 0 && handles.command_queue != 0;
#else
    (void)platform;
#endif

    return handles;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d