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

#include "Open3D/Container/MemoryManager.h"

#include <numeric>
#include <unordered_map>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {

void* MemoryManager::Malloc(size_t byte_size, const Device& device) {
    return GetDeviceMemoryManager(device)->Malloc(byte_size, device);
}

void MemoryManager::Free(void* ptr, const Device& device) {
    return GetDeviceMemoryManager(device)->Free(ptr, device);
}

void MemoryManager::MemcpyBlob(const std::shared_ptr<Blob>& dst_blob,
                               const std::shared_ptr<Blob>& src_blob) {
    if (dst_blob == nullptr || src_blob == nullptr) {
        utility::LogError("Either dst_blob or src_blob is null");
    }
    if (dst_blob->v_ == src_blob->v_) {
        utility::LogError("dst and src have same buffer address");
    }
    if (dst_blob->byte_size_ != src_blob->byte_size_) {
        utility::LogError(
                "dst and src do not have the same byte_size, {} != {}",
                dst_blob->byte_size_, src_blob->byte_size_);
    }
    Memcpy(dst_blob->v_, dst_blob->device_, src_blob->v_, src_blob->device_,
           src_blob->byte_size_);
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

    if ((dst_device.device_type_ != Device::DeviceType::CPU &&
         dst_device.device_type_ != Device::DeviceType::CUDA) ||
        (src_device.device_type_ != Device::DeviceType::CPU &&
         src_device.device_type_ != Device::DeviceType::CUDA)) {
        utility::LogError("Unimplemented device for Memcpy.");
    }

    std::shared_ptr<DeviceMemoryManager> device_mm;
    if (dst_device.device_type_ == Device::DeviceType::CPU &&
        src_device.device_type_ == Device::DeviceType::CPU) {
        device_mm = GetDeviceMemoryManager(src_device);
    } else if (src_device.device_type_ == Device::DeviceType::CUDA) {
        device_mm = GetDeviceMemoryManager(src_device);
    } else {
        device_mm = GetDeviceMemoryManager(dst_device);
    }

    device_mm->Memcpy(dst_ptr, dst_device, src_ptr, src_device, num_bytes);
}

void MemoryManager::MemcpyFromHost(void* dst_ptr,
                                   const Device& dst_device,
                                   const void* host_ptr,
                                   size_t num_bytes) {
    // Currenlty default host is CPU:0
    Memcpy(dst_ptr, dst_device, host_ptr, Device("CPU:0"), num_bytes);
}

void MemoryManager::MemcpyToHost(void* host_ptr,
                                 const void* src_ptr,
                                 const Device& src_device,
                                 size_t num_bytes) {
    // Currenlty default host is CPU:0
    Memcpy(host_ptr, Device("CPU:0"), src_ptr, src_device, num_bytes);
}

std::shared_ptr<DeviceMemoryManager> MemoryManager::GetDeviceMemoryManager(
        const Device& device) {
    static std::unordered_map<Device::DeviceType,
                              std::shared_ptr<DeviceMemoryManager>,
                              utility::hash_enum_class::hash>
            map_device_type_to_memory_manager = {
                    {Device::DeviceType::CPU,
                     std::make_shared<CPUMemoryManager>()},
#ifdef BUILD_CUDA_MODULE
                    {Device::DeviceType::CUDA,
                     std::make_shared<CUDAMemoryManager>()},
#endif
            };
    if (map_device_type_to_memory_manager.find(device.device_type_) ==
        map_device_type_to_memory_manager.end()) {
        utility::LogError("Unimplemented device");
    }
    return map_device_type_to_memory_manager.at(device.device_type_);
}

}  // namespace open3d
