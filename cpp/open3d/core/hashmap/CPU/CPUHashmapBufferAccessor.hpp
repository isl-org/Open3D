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

#include <atomic>
#include <memory>
#include <vector>

#include "open3d/core/hashmap/HashmapBuffer.h"

namespace open3d {
namespace core {

/// Dynamic memory allocation and free are expensive on kernels.
/// We pre-allocate a chunk of memory and manually manage them on kernels.
class CPUHashmapBufferAccessor {
public:
    CPUHashmapBufferAccessor(int64_t capacity,
                             int64_t dsize_key,
                             int64_t dsize_value,
                             Tensor &keys,
                             Tensor &values,
                             Tensor &heap)
        : capacity_(capacity),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          keys_(keys.GetDataPtr<uint8_t>()),
          values_(values.GetDataPtr<uint8_t>()),
          heap_(static_cast<addr_t *>(heap.GetDataPtr())) {
        std::memset(values_, 0, capacity_ * dsize_value_);
    }

    void Reset() {
#pragma omp parallel for
        for (int i = 0; i < capacity_; ++i) {
            heap_[i] = i;
        }

        heap_counter_ = 0;
    }

    addr_t DeviceAllocate() { return heap_[heap_counter_.fetch_add(1)]; }

    void DeviceFree(addr_t ptr) { heap_[heap_counter_.fetch_sub(1) - 1] = ptr; }

    int HeapCounter() const { return heap_counter_.load(); }

    std::pair<void *, void *> ExtractIterator(addr_t ptr) {
        return std::make_pair(keys_ + ptr * dsize_key_,
                              values_ + ptr * dsize_value_);
    }

public:
    int64_t capacity_;
    int64_t dsize_key_;
    int64_t dsize_value_;

    uint8_t *keys_;                 /* [N] * sizeof(Key) */
    uint8_t *values_;               /* [N] * sizeof(Value) */
    addr_t *heap_;                  /* [N] */
    std::atomic<int> heap_counter_; /* [1] */
};

}  // namespace core
}  // namespace open3d
