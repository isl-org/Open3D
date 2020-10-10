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

#include <assert.h>

#include <memory>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/Macros.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {

class KvPairs {
public:
    KvPairs(size_t capacity,
            size_t dsize_key,
            size_t dsize_value,
            const Device& device)
        : capacity_(capacity),
          dsize_key_(dsize_key),
          dsize_val_(dsize_value),
          device_(device) {}
    virtual ~KvPairs() {}

    virtual void ResetHeap() = 0;
    virtual void* GetKeyBufferPtr() = 0;
    virtual void* GetValueBufferPtr() = 0;

    virtual int heap_counter() = 0;

protected:
    size_t capacity_;
    size_t dsize_key_;
    size_t dsize_val_;

    Device device_;
};
}  // namespace core
}  // namespace open3d
