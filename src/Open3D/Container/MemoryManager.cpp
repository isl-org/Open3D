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

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Open3D/Utility/Console.h"

namespace open3d {

OPEN3D_DEFINE_REGISTRY_FOR_SINGLETON(MemoryManagerBackendRegistry,
                                     std::shared_ptr<MemoryManagerBackend>);

std::shared_ptr<MemoryManagerBackend> MemoryManager::GetImpl(
        const std::string& device) {
    if (MemoryManagerBackendRegistry()->Has(device)) {
        return MemoryManagerBackendRegistry()->GetSingletonObject(device);
    } else {
        throw std::runtime_error("Cannot get MemoryManager impl for " + device);
    }
}

void* MemoryManager::Alloc(size_t byte_size, const std::string& device) {
    return GetImpl(device)->Alloc(byte_size);
}

// TODO: consider removing the "device" argument, check ptr device first
void MemoryManager::Free(void* ptr) {
    if (ptr) {
        if (IsCUDAPointer(ptr)) {
            GetImpl("GPU")->Free(ptr);
        } else {
            GetImpl("CPU")->Free(ptr);
        }
    }
}

void MemoryManager::CopyTo(void* dst_ptr,
                           const void* src_ptr,
                           std::size_t num_bytes) {
    if (dst_ptr == nullptr || src_ptr == nullptr) {
        throw std::runtime_error("CopyTo: nullptr detected");
    }
    std::string dst_device = IsCUDAPointer(dst_ptr) ? "GPU" : "CPU";
    std::string src_device = IsCUDAPointer(src_ptr) ? "GPU" : "CPU";

    if (src_device == "GPU" || dst_device == "GPU") {
        GetImpl("GPU")->CopyTo(dst_ptr, src_ptr, num_bytes);
    } else {
        GetImpl("CPU")->CopyTo(dst_ptr, src_ptr, num_bytes);
    }
}

bool MemoryManager::IsCUDAPointer(const void* ptr) {
    if (MemoryManagerBackendRegistry()->Has("GPU")) {
        return GetImpl("GPU")->IsCUDAPointer(ptr);
    } else {
        return GetImpl("CPU")->IsCUDAPointer(ptr);
    }
}

}  // namespace open3d
