// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>

#include "open3d/core/MemoryManager.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void* MemoryManagerCPU::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    ptr = std::malloc(byte_size);
    if (byte_size != 0 && !ptr) {
        utility::LogError("CPU malloc failed");
    }
    return ptr;
}

void MemoryManagerCPU::Free(void* ptr, const Device& device) {
    if (ptr) {
        std::free(ptr);
    }
}

void MemoryManagerCPU::Memcpy(void* dst_ptr,
                              const Device& dst_device,
                              const void* src_ptr,
                              const Device& src_device,
                              size_t num_bytes) {
    std::memcpy(dst_ptr, src_ptr, num_bytes);
}

}  // namespace core
}  // namespace open3d
