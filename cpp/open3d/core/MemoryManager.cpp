// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/MemoryManager.h"

#include <numeric>
#include <unordered_map>

#include "open3d/core/Blob.h"
#include "open3d/core/Device.h"
#include "open3d/core/MemoryManagerStatistic.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void* MemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr = GetMemoryManagerDevice(device)->Malloc(byte_size, device);
    MemoryManagerStatistic::GetInstance().CountMalloc(ptr, byte_size, device);
    return ptr;
}

void MemoryManager::Free(void* ptr, const Device& device) {
    // Update statistics before freeing the memory. This ensures a consistent
    // order in case a subsequent Malloc requires the currently freed memory.
    MemoryManagerStatistic::GetInstance().CountFree(ptr, device);
    GetMemoryManagerDevice(device)->Free(ptr, device);
}

void MemoryManager::Memcpy(void* dst_ptr,
                           const Device& dst_device,
                           const void* src_ptr,
                           const Device& src_device,
                           size_t num_bytes) {
    // 0-element Tensor's data_ptr_ is nullptr
    if (num_bytes == 0) {
        return;
    } else if (src_ptr == nullptr || dst_ptr == nullptr) {
        utility::LogError("src_ptr and dst_ptr cannot be nullptr.");
    }

    std::shared_ptr<MemoryManagerDevice> device_mm;
    // CPU.
    if (src_device.IsCPU() && dst_device.IsCPU()) {
        device_mm = GetMemoryManagerDevice(src_device);
    }
    // CUDA.
    else if (src_device.IsCPU() && dst_device.IsCUDA()) {
        device_mm = GetMemoryManagerDevice(dst_device);
    } else if (src_device.IsCUDA() && dst_device.IsCPU()) {
        device_mm = GetMemoryManagerDevice(src_device);
    } else if (src_device.IsCUDA() && dst_device.IsCUDA()) {
        device_mm = GetMemoryManagerDevice(src_device);
    }
    // SYCL.
    else if (src_device.IsCPU() && dst_device.IsSYCL()) {
        device_mm = GetMemoryManagerDevice(dst_device);
    } else if (src_device.IsSYCL() && dst_device.IsCPU()) {
        device_mm = GetMemoryManagerDevice(src_device);
    } else if (src_device.IsSYCL() && dst_device.IsSYCL()) {
        device_mm = GetMemoryManagerDevice(src_device);
    }
    // Not supporting other combinations at the moment, e.g. SYCL->CUDA.
    else {
        utility::LogError("Unsupported device type from {} to {}.",
                          src_device.ToString(), dst_device.ToString());
    }

    device_mm->Memcpy(dst_ptr, dst_device, src_ptr, src_device, num_bytes);
}

void MemoryManager::MemcpyFromHost(void* dst_ptr,
                                   const Device& dst_device,
                                   const void* host_ptr,
                                   size_t num_bytes) {
    // Currently default host is CPU:0
    Memcpy(dst_ptr, dst_device, host_ptr, Device("CPU:0"), num_bytes);
}

void MemoryManager::MemcpyToHost(void* host_ptr,
                                 const void* src_ptr,
                                 const Device& src_device,
                                 size_t num_bytes) {
    // Currently default host is CPU:0
    Memcpy(host_ptr, Device("CPU:0"), src_ptr, src_device, num_bytes);
}

std::shared_ptr<MemoryManagerDevice> MemoryManager::GetMemoryManagerDevice(
        const Device& device) {
    static std::unordered_map<Device::DeviceType,
                              std::shared_ptr<MemoryManagerDevice>,
                              utility::hash_enum_class>
            map_device_type_to_memory_manager = {
                    {Device::DeviceType::CPU,
                     std::make_shared<MemoryManagerCPU>()},
#ifdef BUILD_CUDA_MODULE
#ifdef ENABLE_CACHED_CUDA_MANAGER
                    {Device::DeviceType::CUDA,
                     std::make_shared<MemoryManagerCached>(
                             std::make_shared<MemoryManagerCUDA>())},
#else
                    {Device::DeviceType::CUDA,
                     std::make_shared<MemoryManagerCUDA>()},
#endif
#endif
#ifdef BUILD_SYCL_MODULE
                    {Device::DeviceType::SYCL,
                     std::make_shared<MemoryManagerSYCL>()},
#endif
            };

    if (map_device_type_to_memory_manager.find(device.GetType()) ==
        map_device_type_to_memory_manager.end()) {
        utility::LogError(
                "Unsupported device \"{}\". Set BUILD_CUDA_MODULE=ON to "
                "compile for CUDA support and BUILD_SYCL_MODULE=ON to compile "
                "for SYCL support.",
                device.ToString());
    }
    return map_device_type_to_memory_manager.at(device.GetType());
}

}  // namespace core
}  // namespace open3d
