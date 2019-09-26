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
#include <numeric>
#include <unordered_map>

#include "Open3D/Container/Blob.h"
#include "Open3D/Container/CudaUtils.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

void* MemoryManager::Malloc(size_t byte_size, const Device& device) {
    return GetDeviceMemoryManager(device)->Malloc(byte_size, device);
}

void MemoryManager::Free(Blob* blob) {
    return GetDeviceMemoryManager(blob->device_)->Free(blob->v_, blob->device_);
}

void MemoryManager::Free(void* ptr, const Device& device) {
    return GetDeviceMemoryManager(device)->Free(ptr, device);
}

void MemoryManager::Memcpy(void* dst_ptr,
                           const Device& dst_device,
                           const void* src_ptr,
                           const Device& src_device,
                           size_t num_bytes) {
    if ((dst_device.device_type_ != Device::DeviceType::CPU &&
         dst_device.device_type_ != Device::DeviceType::GPU) ||
        (src_device.device_type_ != Device::DeviceType::CPU &&
         src_device.device_type_ != Device::DeviceType::GPU)) {
        utility::LogFatal("Unimplemented device for Memcpy\n");
    }

    std::shared_ptr<DeviceMemoryManager> device_mm;
    if (dst_device.device_type_ == Device::DeviceType::CPU &&
        src_device.device_type_ == Device::DeviceType::CPU) {
        device_mm = GetDeviceMemoryManager(src_device);
    } else if (src_device.device_type_ == Device::DeviceType::GPU) {
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
                              std::shared_ptr<DeviceMemoryManager>>
            map_device_type_to_memory_manager = {
                    {Device::DeviceType::CPU,
                     std::make_shared<CPUMemoryManager>()},
                    {Device::DeviceType::GPU,
                     std::make_shared<GPUMemoryManager>()},
            };
    if (map_device_type_to_memory_manager.find(device.device_type_) ==
        map_device_type_to_memory_manager.end()) {
        utility::LogFatal("Unimplemented device\n");
    }
    return map_device_type_to_memory_manager.at(device.device_type_);
}

CPUMemoryManager::CPUMemoryManager() {}

void* CPUMemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    ptr = malloc(byte_size);
    if (byte_size != 0 && !ptr) {
        utility::LogFatal("CPU malloc failed\n");
    }
    return ptr;
}

void CPUMemoryManager::Free(void* ptr, const Device& device) {
    if (ptr) {
        free(ptr);
    }
}

void CPUMemoryManager::Memcpy(void* dst_ptr,
                              const Device& dst_device,
                              const void* src_ptr,
                              const Device& src_device,
                              size_t num_bytes) {
    // TODO: safer memcpy_s
    memcpy(dst_ptr, src_ptr, num_bytes);
}

GPUMemoryManager::GPUMemoryManager() {
    // TODO: reenable this when p2p is supported
    // EnableP2P();
}

void GPUMemoryManager::EnableP2P() {
    int device_count = -1;
    OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        utility::LogFatal("CUDA device not found, device_count={}\n",
                          device_count);
    } else {
        utility::LogDebug("device_count = {}\n", device_count);
    }

    // Enable P2P
    for (int curr_id = 0; curr_id < device_count; ++curr_id) {
        SetDevice(curr_id);
        for (int peer_id = 0; peer_id < device_count; ++peer_id) {
            if (curr_id == peer_id) {
                continue;
            }
            int accessible = 0;
            OPEN3D_CUDA_CHECK(
                    cudaDeviceCanAccessPeer(&accessible, curr_id, peer_id));
            if (accessible == 1) {
                OPEN3D_CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_id, 0));
            } else {
                utility::LogWarning("{} can't access {}\n", curr_id, peer_id);
            }
        }
    }
}

void GPUMemoryManager::SetDevice(int device_id) {
    int curr_device_id = -1;
    OPEN3D_CUDA_CHECK(cudaGetDevice(&curr_device_id));
    if (curr_device_id != device_id) {
        OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
    }
}

void* GPUMemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    if (device.device_type_ == Device::DeviceType::GPU) {
        OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
    return ptr;
}

void GPUMemoryManager::Free(void* ptr, const Device& device) {
    if (device.device_type_ == Device::DeviceType::GPU) {
        if (!IsCUDAPointer(ptr)) {
            utility::LogFatal("Freeing non-CUDA pointer\n");
        }
        if (ptr) {
            OPEN3D_CUDA_CHECK(cudaFree(ptr));
        }
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
}

void GPUMemoryManager::Memcpy(void* dst_ptr,
                              const Device& dst_device,
                              const void* src_ptr,
                              const Device& src_device,
                              size_t num_bytes) {
    cudaMemcpyKind memcpy_kind;

    if (dst_device.device_type_ == Device::DeviceType::GPU &&
        src_device.device_type_ == Device::DeviceType::CPU) {
        memcpy_kind = cudaMemcpyHostToDevice;
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogFatal("dst_ptr is not a CUDA pointer\n");
        }
    } else if (dst_device.device_type_ == Device::DeviceType::CPU &&
               src_device.device_type_ == Device::DeviceType::GPU) {
        memcpy_kind = cudaMemcpyDeviceToHost;
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogFatal("src_ptr is not a CUDA pointer\n");
        }
    } else if (dst_device.device_type_ == Device::DeviceType::GPU &&
               src_device.device_type_ == Device::DeviceType::GPU) {
        memcpy_kind = cudaMemcpyDeviceToDevice;
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogFatal("dst_ptr is not a CUDA pointer\n");
        }
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogFatal("src_ptr is not a CUDA pointer\n");
        }
    } else {
        utility::LogFatal("Wrong cudaMemcpyKind\n");
    }

    OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes, memcpy_kind));
}

bool GPUMemoryManager::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

}  // namespace open3d
