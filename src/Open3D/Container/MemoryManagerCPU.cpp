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

#include "Open3D/Container/MemoryManagerCPU.h"
#include "Open3D/Registry.h"

namespace open3d {

void* MemoryManagerCPU::Allocate(size_t byte_size) {
    void* ptr = malloc(byte_size);
    if (byte_size != 0 && !ptr) {
        std::runtime_error("CPU malloc failed");
        throw std::bad_alloc();
    }
    return ptr;
}

void MemoryManagerCPU::Free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

void MemoryManagerCPU::CopyTo(void* dst_ptr,
                              const void* src_ptr,
                              std::size_t num_bytes) {
    if (dst_ptr == nullptr || src_ptr == nullptr) {
        throw std::runtime_error("CopyTo: nullptr detected");
    }
    std::memcpy(dst_ptr, src_ptr, num_bytes);
}

bool MemoryManagerCPU::IsCUDAPointer(const void* ptr) { return false; }

OPEN3D_REGISTER_SINGLETON_OBJECT(MemoryManagerBackendRegistry,
                                 "cpu",
                                 std::make_shared<MemoryManagerCPU>());

}  // namespace open3d
