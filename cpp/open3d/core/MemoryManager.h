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

#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "open3d/core/Device.h"

namespace open3d {
namespace core {

class DeviceMemoryManager;

/// Top-level memory interface. Calls to any of the member functions will
/// automatically dispatch the appropriate DeviceMemoryManager instance based on
/// the provided device which is used to execute the requested functionality.
///
/// The memory managers are dispatched as follows:
///
/// DeviceType = CPU : CPUMemoryManager
/// DeviceType = CUDA :
///   BUILD_CACHED_CUDA_MANAGER = ON : CachedMemoryManager w/ CUDAMemoryManager
///   Otherwise :                      CUDAMemoryManager
///
class MemoryManager {
public:
    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    static void* Malloc(size_t byte_size, const Device& device);

    /// Frees previously allocated memory at address \p ptr on device \p device.
    static void Free(void* ptr, const Device& device);

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    static void Memcpy(void* dst_ptr,
                       const Device& dst_device,
                       const void* src_ptr,
                       const Device& src_device,
                       size_t num_bytes);

    /// Same as Memcpy, but with host (CPU:0) as default src_device.
    static void MemcpyFromHost(void* dst_ptr,
                               const Device& dst_device,
                               const void* host_ptr,
                               size_t num_bytes);

    /// Same as Memcpy, but with host (CPU:0) as default dst_device.
    static void MemcpyToHost(void* host_ptr,
                             const void* src_ptr,
                             const Device& src_device,
                             size_t num_bytes);

protected:
    /// Internally dispatches the appropriate DeviceMemoryManager instance.
    static std::shared_ptr<DeviceMemoryManager> GetDeviceMemoryManager(
            const Device& device);
};

/// Interface for all concrete memory manager classes.
class DeviceMemoryManager {
public:
    virtual ~DeviceMemoryManager() = default;

    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    virtual void* Malloc(size_t byte_size, const Device& device) = 0;

    /// Frees previously allocated memory at address \p ptr on device \p device.
    virtual void Free(void* ptr, const Device& device) = 0;

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    virtual void Memcpy(void* dst_ptr,
                        const Device& dst_device,
                        const void* src_ptr,
                        const Device& src_device,
                        size_t num_bytes) = 0;
};

/// Generic cached memory manager. This class can be used to speed-up memory
/// allocations and deallocations from arbitrary direct memory managers.
///
/// - Successful queries, i.e. cache hits, will result in constant-time
/// allocations, but small direct allocations might still be faster.
///
/// - Failed queries, i.e. cache misses, will result in direct allocations.
///
/// - Direct frees will be delayed until either the end of the program or a
/// cache release is triggered.
///
/// - (Partial) cache releases will be triggered either manually by calling
/// \p ReleaseCache or automatically if a direct allocation fails after
/// observing a cache miss.
///
class CachedMemoryManager : public DeviceMemoryManager {
public:
    /// Constructs a cached memory manager instance that wraps the existing
    /// direct memory manager \p device_mm.
    explicit CachedMemoryManager(
            const std::shared_ptr<DeviceMemoryManager>& device_mm);

    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    void* Malloc(size_t byte_size, const Device& device) override;

    /// Frees previously allocated memory at address \p ptr on device \p device.
    void Free(void* ptr, const Device& device) override;

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;

public:
    /// Frees all releasable memory blocks on device \p device.
    static void ReleaseCache(const Device& device);

    /// Frees all releasable memory blocks on all known devices.
    /// Note that this may also affect other instances of CachedMemoryManager.
    static void ReleaseCache();

protected:
    std::shared_ptr<DeviceMemoryManager> device_mm_;
};

/// Direct memory manager which performs allocations and deallocations on the
/// CPU via \p std::malloc and \p std::free.
class CPUMemoryManager : public DeviceMemoryManager {
public:
    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    void* Malloc(size_t byte_size, const Device& device) override;

    /// Frees previously allocated memory at address \p ptr on device \p device.
    void Free(void* ptr, const Device& device) override;

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;
};

#ifdef BUILD_CUDA_MODULE
/// Direct memory manager which performs allocations and deallocations on CUDA
/// devices via \p cudaMalloc and \p cudaFree.
class CUDAMemoryManager : public DeviceMemoryManager {
public:
    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    void* Malloc(size_t byte_size, const Device& device) override;

    /// Frees previously allocated memory at address \p ptr on device \p device.
    void Free(void* ptr, const Device& device) override;

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                const void* src_ptr,
                const Device& src_device,
                size_t num_bytes) override;

protected:
    bool IsCUDAPointer(const void* ptr, const Device& device);
};
#endif

}  // namespace core
}  // namespace open3d
