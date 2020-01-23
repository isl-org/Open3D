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

#include <cstdlib>

namespace open3d {

CPUMemoryManager::CPUMemoryManager() {}

void* CPUMemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    ptr = std::malloc(byte_size);
    if (byte_size != 0 && !ptr) {
        utility::LogError("CPU malloc failed");
    }
    return ptr;
}

void CPUMemoryManager::Free(void* ptr, const Device& device) {
    if (ptr) {
        std::free(ptr);
    }
}

void CPUMemoryManager::Memcpy(void* dst_ptr,
                              const Device& dst_device,
                              const void* src_ptr,
                              const Device& src_device,
                              size_t num_bytes) {
    std::memcpy(dst_ptr, src_ptr, num_bytes);
}

}  // namespace open3d
