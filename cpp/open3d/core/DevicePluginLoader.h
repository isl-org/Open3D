// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

namespace open3d {
namespace core {

class MemoryManagerDevice;

/// Check if the device plugin libraries were successfully loaded at
/// initialization.
bool IsCudaDeviceLibraryLoaded();
bool IsXpuDeviceLibraryLoaded();

#if defined(BUILD_SHARED_LIBS)
std::shared_ptr<MemoryManagerDevice> CreateCudaMemoryManagerDevice();
#endif

}  // namespace core
}  // namespace open3d
