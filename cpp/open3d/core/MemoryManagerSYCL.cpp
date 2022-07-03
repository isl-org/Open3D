// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <CL/sycl.hpp>
#include <cstdlib>
#include <unordered_map>

#include "open3d/core/Device.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

namespace sy = cl::sycl;

void* MemoryManagerSYCL::Malloc(size_t byte_size, const Device& device) {
    const sy::queue& queue =
            sycl::SYCLContext::GetInstance().GetDefaultQueue(device);

#ifdef ENABLE_SYCL_UNIFIED_SHARED_MEMORY
    return static_cast<void*>(sy::malloc_shared(byte_size, queue));
#else
    return static_cast<void*>(sy::malloc_device(byte_size, queue));
#endif
}

void MemoryManagerSYCL::Free(void* ptr, const Device& device) {
    if (ptr) {
        const sy::queue& queue =
                sycl::SYCLContext::GetInstance().GetDefaultQueue(device);
        sy::free(ptr, queue);
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

    sy::queue queue =
            sycl::SYCLContext::GetInstance().GetDefaultQueue(device_with_queue);
    queue.memcpy(dst_ptr, src_ptr, num_bytes).wait_and_throw();
}

}  // namespace core
}  // namespace open3d
