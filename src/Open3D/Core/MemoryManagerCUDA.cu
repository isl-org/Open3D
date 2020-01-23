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

#include "Open3D/Core/MemoryManager.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/CUDAState.cuh"
#include "Open3D/Core/CUDAUtils.h"

namespace open3d {

CUDAMemoryManager::CUDAMemoryManager() {}

void* CUDAMemoryManager::Malloc(size_t byte_size, const Device& device) {
    CUDADeviceSwitcher switcher(device);
    void* ptr;
    if (device.GetType() == Device::DeviceType::CUDA) {
        OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
    } else {
        utility::LogError("CUDAMemoryManager::Malloc: Unimplemented device");
    }
    return ptr;
}

void CUDAMemoryManager::Free(void* ptr, const Device& device) {
    CUDADeviceSwitcher switcher(device);
    if (device.GetType() == Device::DeviceType::CUDA) {
        if (ptr && IsCUDAPointer(ptr)) {
            OPEN3D_CUDA_CHECK(cudaFree(ptr));
        }
    } else {
        utility::LogError("CUDAMemoryManager::Free: Unimplemented device");
    }
}

void CUDAMemoryManager::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    if (dst_device.GetType() == Device::DeviceType::CUDA &&
        src_device.GetType() == Device::DeviceType::CPU) {
        CUDADeviceSwitcher switcher(dst_device);
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
        OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                     cudaMemcpyHostToDevice));
    } else if (dst_device.GetType() == Device::DeviceType::CPU &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        CUDADeviceSwitcher switcher(src_device);
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }
        OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                     cudaMemcpyDeviceToHost));
    } else if (dst_device.GetType() == Device::DeviceType::CUDA &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        CUDADeviceSwitcher switcher(dst_device);
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
        switcher.SwitchTo(src_device);
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }

        if (dst_device == src_device) {
            CUDADeviceSwitcher switcher(src_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes,
                                         cudaMemcpyDeviceToDevice));
        } else if (CUDAState::GetInstance()->IsP2PEnabled(src_device.GetID(),
                                                          dst_device.GetID())) {
            OPEN3D_CUDA_CHECK(cudaMemcpyPeer(dst_ptr, dst_device.GetID(),
                                             src_ptr, src_device.GetID(),
                                             num_bytes));
        } else {
            void* cpu_buf = MemoryManager::Malloc(num_bytes, Device("CPU:0"));
            CUDADeviceSwitcher switcher(src_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(cpu_buf, src_ptr, num_bytes,
                                         cudaMemcpyDeviceToHost));
            switcher.SwitchTo(dst_device);
            OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, cpu_buf, num_bytes,
                                         cudaMemcpyHostToDevice));
            MemoryManager::Free(cpu_buf, Device("CPU:0"));
        }
    } else {
        utility::LogError("Wrong cudaMemcpyKind");
    }
}

bool CUDAMemoryManager::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

}  // namespace open3d
