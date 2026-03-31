// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#ifdef __APPLE__

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

/// Metal native device and command-queue handles, retrieved from Filament's
/// PlatformMetal.  Used by the Gaussian-splatting Metal backend to share GPU
/// resources with Filament without extra allocations.
struct FilamentMetalNativeHandles {
    std::uintptr_t device = 0;
    std::uintptr_t command_queue = 0;
    bool valid = false;
};

FilamentMetalNativeHandles GetFilamentMetalNativeHandles(
        filament::backend::Platform* platform);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif