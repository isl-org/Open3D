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

#include "Open3D/Container/MemoryManagerGPU.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace open3d {

void* MemoryManagerGPU::Allocate(size_t byte_size) {
    void* ptr;
    cudaMalloc(static_cast<void**>(&ptr), byte_size);
    return ptr;
}

void MemoryManagerGPU::Free(void* ptr) {
    if (ptr) {
        if (IsCUDAPointer(ptr)) {
            cudaFree(ptr);
        } else {
            throw std::runtime_error("MemoryManagerGPU::Free: host pointer");
        }
    }
}

void MemoryManagerGPU::CopyTo(void* dst_ptr,
                              const void* src_ptr,
                              std::size_t num_bytes) {
    if (dst_ptr == nullptr || src_ptr == nullptr) {
        throw std::runtime_error("CopyTo: nullptr detected");
    }

    std::string dst_device = IsCUDAPointer(dst_ptr) ? "GPU" : "CPU";
    std::string src_device = IsCUDAPointer(src_ptr) ? "GPU" : "CPU";

    if (src_device == "CPU" && dst_device == "CPU") {
        std::memcpy(dst_ptr, src_ptr, num_bytes);
    } else if (src_device == "CPU" && dst_device == "GPU") {
        cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice);
    } else if (src_device == "GPU" && dst_device == "CPU") {
        cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost);
    } else if (src_device == "GPU" && dst_device == "GPU") {
        cudaMemcpy(dst_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToDevice);
    }
}

bool MemoryManagerGPU::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

OPEN3D_REGISTER_SINGLETON_OBJECT(MemoryManagerBackendRegistry,
                                 "GPU",
                                 std::make_shared<MemoryManagerGPU>());

}  // namespace open3d
