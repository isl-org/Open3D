// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <sycl/sycl.hpp>
#include <unordered_map>

#include "open3d/core/Device.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void* MemoryManagerSYCL::Malloc(size_t byte_size, const Device& device) {
    const sycl::queue& queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);

#ifdef ENABLE_SYCL_UNIFIED_SHARED_MEMORY
    return static_cast<void*>(sycl::malloc_shared(byte_size, queue));
#else
    return static_cast<void*>(sycl::malloc_device(byte_size, queue));
#endif
}

void MemoryManagerSYCL::Free(void* ptr, const Device& device) {
    if (ptr) {
        const sycl::queue& queue =
                sy::SYCLContext::GetInstance().GetDefaultQueue(device);
        sycl::free(ptr, queue);
    }
}

void MemoryManagerSYCL::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    Device device_with_queue;

    if (src_device.IsCPU() && dst_device.IsCPU()) {
        utility::LogError(
                "Internal error: trying to transfer {}->{}, should not reach "
                "this function.",
                src_device.ToString(), dst_device.ToString());
    } else if (src_device.IsCPU() && dst_device.IsSYCL()) {
        device_with_queue = dst_device;
    } else if (src_device.IsSYCL() && dst_device.IsCPU()) {
        device_with_queue = src_device;
    } else if (src_device.IsSYCL() && dst_device.IsSYCL()) {
        device_with_queue = src_device;
    } else {
        utility::LogError("Wrong device {}->{}.", src_device.ToString(),
                          dst_device.ToString());
    }

    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device_with_queue);
    queue.memcpy(dst_ptr, src_ptr, num_bytes).wait_and_throw();
}

}  // namespace core
}  // namespace open3d
