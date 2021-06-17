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

// The value_ array's size is FIXED.
// The heap_ array stores the addresses of the values.
// Only the unallocated part is maintained.
// (ONLY care about the heap above the heap counter. Below is
// meaningless.)
// During Allocate, ptr is extracted from the heap;
// During Free, ptr is put back to the top of the heap.
// ---------------------------------------------------------------------
// heap  ---Malloc-->  heap  ---Malloc-->  heap  ---Free(0)-->  heap
// N-1                 N-1                  N-1                  N-1   |
//  .                   .                    .                    .    |
//  .                   .                    .                    .    |
//  .                   .                    .                    .    |
//  3                   3                    3                    3    |
//  2                   2                    2 <-                 2    |
//  1                   1 <-                 1                    0 <- |
//  0 <- heap_counter   0                    0                    0
#pragma once

#include <assert.h>

#include <memory>
#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

// Type for the internal heap. Dtype::Int32 is used to store it in Tensors.
typedef uint32_t addr_t;

class HashmapBuffer {
public:
    HashmapBuffer(int64_t capacity,
                  int64_t dsize_key,
                  int64_t dsize_value,
                  const Device &device)
        : capacity_(capacity),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device) {
        key_buffer_ =
                Tensor({capacity_},
                       Dtype(Dtype::DtypeCode::Object, dsize_key_, "_hash_k"),
                       device_);
        value_buffer_ =
                Tensor({capacity_},
                       Dtype(Dtype::DtypeCode::Object, dsize_value_, "_hash_v"),
                       device_);
        heap_ = Tensor({capacity_}, Dtype::Int32, device_);
    }

    Tensor &GetKeyBuffer() { return key_buffer_; }
    Tensor &GetValueBuffer() { return value_buffer_; }
    Tensor &GetHeap() { return heap_; }

protected:
    int64_t capacity_;
    int64_t dsize_key_;
    int64_t dsize_value_;

    Tensor key_buffer_;
    Tensor value_buffer_;
    Tensor heap_;

    Device device_;
};
}  // namespace core
}  // namespace open3d
