// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2023 www.open3d.org
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

#include "open3d/core/hashmap/HashBackendBuffer.h"

namespace open3d {
namespace core {

class CPUHashBackendBufferAccessor {
public:
    /// Must initialize from a non-const buffer to grab the heap top.
    CPUHashBackendBufferAccessor(HashBackendBuffer &hashmap_buffer)
        : capacity_(hashmap_buffer.GetCapacity()),
          key_dsize_(hashmap_buffer.GetKeyDsize()),
          value_dsizes_(hashmap_buffer.GetValueDsizes()),
          heap_(hashmap_buffer.GetIndexHeap().GetDataPtr<buf_index_t>()),
          key_buffer_ptr_(hashmap_buffer.GetKeyBuffer().GetDataPtr<uint8_t>()) {
        std::vector<Tensor> value_buffers = hashmap_buffer.GetValueBuffers();
        for (size_t i = 0; i < value_buffers.size(); ++i) {
            void *value_buffer_ptr = value_buffers[i].GetDataPtr();
            std::memset(value_buffer_ptr, 0, capacity_ * value_dsizes_[i]);
            value_buffer_ptrs_.push_back(
                    static_cast<uint8_t *>(value_buffer_ptr));
        }
        heap_top_ = &(hashmap_buffer.GetHeapTop().cpu);
    }

    buf_index_t DeviceAllocate() { return heap_[(*heap_top_).fetch_add(1)]; }
    void DeviceFree(buf_index_t buf_index) {
        heap_[(*heap_top_).fetch_sub(1) - 1] = buf_index;
    }

    void *GetKeyPtr(buf_index_t buf_index) {
        return key_buffer_ptr_ + buf_index * key_dsize_;
    }
    void *GetValuePtr(buf_index_t buf_index, int value_idx = 0) {
        return value_buffer_ptrs_[value_idx] +
               buf_index * value_dsizes_[value_idx];
    }

public:
    int64_t capacity_;
    int64_t key_dsize_;
    std::vector<int64_t> value_dsizes_;

    buf_index_t *heap_;          /* [N] */
    std::atomic<int> *heap_top_; /* [1] */

    uint8_t *key_buffer_ptr_;                  /* [N] * sizeof(Key) */
    std::vector<uint8_t *> value_buffer_ptrs_; /* [N] * sizeof(Value) */
};

}  // namespace core
}  // namespace open3d
