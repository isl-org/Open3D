// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {

class CPUHashmapBufferAccessor {
public:
    /// Must initialize from a non-const buffer to grab the heap top.
    CPUHashmapBufferAccessor(HashmapBuffer &hashmap_buffer)
        : capacity_(hashmap_buffer.GetCapacity()),
          dsize_key_(hashmap_buffer.GetKeyDsize()),
          dsize_values_(hashmap_buffer.GetValueDsizes()),
          heap_(hashmap_buffer.GetIndexHeap().GetDataPtr<buf_index_t>()),
          keys_(hashmap_buffer.GetKeyBuffer().GetDataPtr<uint8_t>()) {
        std::vector<Tensor> value_buffers = hashmap_buffer.GetValueBuffers();
        for (size_t i = 0; i < value_buffers.size(); ++i) {
            void *data_ptr = value_buffers[i].GetDataPtr();
            std::memset(data_ptr, 0, capacity_ * dsize_values_[i]);
            values_.push_back(static_cast<uint8_t *>(data_ptr));
        }
        heap_counter_ = &(hashmap_buffer.GetHeapTop().cpu);
    }

    void Reset() {
#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
        for (int i = 0; i < capacity_; ++i) {
            heap_[i] = i;
        }

        *heap_counter_ = 0;
    }

    buf_index_t DeviceAllocate() {
        return heap_[(*heap_counter_).fetch_add(1)];
    }
    void DeviceFree(buf_index_t ptr) {
        heap_[(*heap_counter_).fetch_sub(1) - 1] = ptr;
    }

    int HeapCounter() const { return (*heap_counter_).load(); }

    void *GetKeyPtr(buf_index_t ptr) { return keys_ + ptr * dsize_key_; }
    void *GetValuePtr(buf_index_t ptr, int value_idx = 0) {
        return values_[value_idx] + ptr * dsize_values_[value_idx];
    }

public:
    int64_t capacity_;
    int64_t dsize_key_;
    std::vector<int64_t> dsize_values_;

    buf_index_t *heap_;              /* [N] */
    std::atomic<int> *heap_counter_; /* [1] */

    uint8_t *keys_;                 /* [N] * sizeof(Key) */
    std::vector<uint8_t *> values_; /* [N] * sizeof(Value) */
};

}  // namespace core
}  // namespace open3d
