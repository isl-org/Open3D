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

#include <cstddef>
#include <iostream>
#include <string>

#include "Open3D/Container/Device.h"
#include "Open3D/Container/MemoryManager.h"

namespace open3d {

class Blob : public std::enable_shared_from_this<Blob> {
public:
    Blob(size_t byte_size, const Device& device)
        : byte_size_(byte_size), device_(device) {
        v_ = MemoryManager::Malloc(byte_size_, device_);
    }
    ~Blob() { MemoryManager::Free(shared_from_this()); };

public:
    /// Device data pointer
    void* v_ = nullptr;
    /// Size of Blob in bytes
    size_t byte_size_ = 0;
    /// Device context for the blob
    Device device_;
};

}  // namespace open3d
