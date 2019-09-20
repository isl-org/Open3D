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

#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Open3D/Registry.h"

namespace open3d {

// Base class for MemoryManagerCPU and MemoryManagerGPU
// This can futher be extened to a stateful memory manager (e.g memory pool)
class MemoryManagerBackend {
public:
    virtual void* Alloc(size_t byte_size) = 0;
    virtual void Free(void* ptr) = 0;
    virtual void CopyTo(void* dst_ptr,
                        const void* src_ptr,
                        std::size_t num_bytes) = 0;
    virtual bool IsCUDAPointer(const void* ptr) = 0;
};

OPEN3D_DECLARE_REGISTRY_FOR_SINGLETON(MemoryManagerBackendRegistry,
                                      std::shared_ptr<MemoryManagerBackend>);

// MemoryManager external API
// MemoryManager is stateless (static functions), except for the registry
class MemoryManager {
public:
    static void* Alloc(size_t byte_size, const std::string& device);
    static void Free(void* ptr);
    static void CopyTo(void* dst_ptr,
                       const void* src_ptr,
                       std::size_t num_bytes);
    static bool IsCUDAPointer(const void* ptr);

protected:
    static std::shared_ptr<MemoryManagerBackend> GetImpl(
            const std::string& device);
};

}  // namespace open3d
