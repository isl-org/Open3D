// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Open3D/Core/Device.h"

namespace open3d {

class DeviceMemoryManager;

class MemoryManager {
public:
    static void* Malloc(size_t byte_size, const Device& device);
    static void Free(void* ptr, const Device& device);
    static void Memcpy(void* dst_ptr,
                       const Device& dst_device,
                       const void* src_ptr,
                       const Device& src_device,
                       size_t num_bytes);
    /// Same as Memcpy, but with host (CPU:0) as default src_device
    static void MemcpyFromHost(void* dst_ptr,
                               const Device& dst_device,
                               const void* host_ptr,
                               size_t num_bytes);
    /// Same as Memcpy, but with host (CPU:0) as default dst_device
    static void MemcpyToHost(void* host_ptr,
                             const void* src_ptr,
                             const Device& src_device,
                             size_t num_bytes);

protected:
    static std::shared_ptr<DeviceMemoryManager> GetDeviceMemoryManager(
            const Device& device);
};

class DeviceMemoryManager {
public:
    virtual void* Malloc(size_t byte_size, const Device& device) = 0;
    virtual void Free(void* ptr, const Device& device) = 0;
    virtual void Memcpy(void* dst_ptr,
                        const Device& dst_device,
                        const void* src_ptr,
                        const Device& src_device,
                        size_t num_bytes) = 0;
};

class CPUMemoryManager : public DeviceMemoryManager {
public:
    CPUMemoryManager();
    void* Malloc(size_t byte_size, const Device& device) override;
    void Free(void* ptr, const Device& device) override;
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;
};

#ifdef BUILD_CUDA_MODULE
class CUDAMemoryManager : public DeviceMemoryManager {
public:
    CUDAMemoryManager();
    void* Malloc(size_t byte_size, const Device& device) override;
    void Free(void* ptr, const Device& device) override;
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;

protected:
    bool IsCUDAPointer(const void* ptr);
};
#endif

}  // namespace open3d
