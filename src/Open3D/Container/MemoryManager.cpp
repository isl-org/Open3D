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

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/CudaUtils.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

void* MemoryManager::Alloc(size_t byte_size, const Device& device) {
    return GetDeviceMemoryManager(device)->Alloc(byte_size, device);
}

void MemoryManager::Free(Blob* blob) {
    return GetDeviceMemoryManager(blob->device_)->Free(blob);
}

std::shared_ptr<DeviceMemoryManager> MemoryManager::GetDeviceMemoryManager(
        const Device& device) {
    static std::unordered_map<Device::DeviceType,
                              std::shared_ptr<DeviceMemoryManager>>
            map_device_type_to_memory_manager = {
                    {Device::DeviceType::kCPU,
                     std::make_shared<CPUMemoryManager>()},
                    {Device::DeviceType::kGPU,
                     std::make_shared<GPUMemoryManager>()},
            };
    if (map_device_type_to_memory_manager.find(device.device_type_) ==
        map_device_type_to_memory_manager.end()) {
        utility::LogFatal("Unimplemented device\n");
    }
    return map_device_type_to_memory_manager.at(device.device_type_);
}

CPUMemoryManager::CPUMemoryManager() {}

void* CPUMemoryManager::Alloc(size_t byte_size, const Device& device) {
    void* ptr;
    if (device.device_type_ == Device::DeviceType::kCPU) {
        ptr = malloc(byte_size);
        if (byte_size != 0 && !ptr) {
            utility::LogFatal("CPU malloc failed\n");
        }
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
    return ptr;
}

void CPUMemoryManager::Free(Blob* blob) {
    if (blob->device_.device_type_ == Device::DeviceType::kCPU) {
        if (blob->v_) {
            free(blob->v_);
        }
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
}

GPUMemoryManager::GPUMemoryManager() {
    // Enable P2P
}

void* GPUMemoryManager::Alloc(size_t byte_size, const Device& device) {
    void* ptr;
    if (device.device_type_ == Device::DeviceType::kGPU) {
        OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
    return ptr;
}

void GPUMemoryManager::Free(Blob* blob) {
    if (blob->device_.device_type_ == Device::DeviceType::kGPU) {
        if (blob->v_) {
            OPEN3D_CUDA_CHECK(cudaFree(blob->v_));
        }
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
}

}  // namespace open3d
