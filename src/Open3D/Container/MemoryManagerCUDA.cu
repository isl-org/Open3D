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

#include "Open3D/Container/CudaUtils.h"

namespace open3d {

CUDAMemoryManager::CUDAMemoryManager() {}

void CUDAMemoryManager::SetDevice(int device_id) {
    int curr_device_id = -1;
    OPEN3D_CUDA_CHECK(cudaGetDevice(&curr_device_id));
    if (curr_device_id != device_id) {
        OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
    }
}

void* CUDAMemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    if (device.device_type_ == Device::DeviceType::CUDA) {
        OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
    } else {
        utility::LogFatal("Unimplemented device\n");
    }
    return ptr;
}

void CUDAMemoryManager::Free(void* ptr, const Device& device) {
    if (device.device_type_ == Device::DeviceType::CUDA) {
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

void CUDAMemoryManager::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    cudaMemcpyKind memcpy_kind;

    if (dst_device.device_type_ == Device::DeviceType::CUDA &&
        src_device.device_type_ == Device::DeviceType::CPU) {
        memcpy_kind = cudaMemcpyHostToDevice;
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogFatal("dst_ptr is not a CUDA pointer\n");
        }
    } else if (dst_device.device_type_ == Device::DeviceType::CPU &&
               src_device.device_type_ == Device::DeviceType::CUDA) {
        memcpy_kind = cudaMemcpyDeviceToHost;
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogFatal("src_ptr is not a CUDA pointer\n");
        }
    } else if (dst_device.device_type_ == Device::DeviceType::CUDA &&
               src_device.device_type_ == Device::DeviceType::CUDA) {
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

bool CUDAMemoryManager::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

}  // namespace open3d
