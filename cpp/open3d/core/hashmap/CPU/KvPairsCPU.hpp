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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/Macros.h"
#include "open3d/core/hashmap/KvPairs.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {

/// Dynamic memory allocation and free are expensive on kernels.
/// We pre-allocate a chunk of memory and manually manage them on kernels.
class CPUKvPairsContext {
public:
    uint8_t *keys_;                 /* [N] * sizeof(Key) */
    uint8_t *values_;               /* [N] * sizeof(Value) */
    addr_t *heap_;                  /* [N] */
    std::atomic<int> heap_counter_; /* [1] */

public:
    int dsize_key_;
    int dsize_value_;
    int capacity_;

public:
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

    addr_t Allocate() { return heap_[heap_counter_.fetch_add(1)]; }

    void Free(addr_t ptr) { heap_[heap_counter_.fetch_sub(1) - 1] = ptr; }

    iterator_t extract_iterator(addr_t ptr) {
        return iterator_t(keys_ + ptr * dsize_key_,
                          values_ + ptr * dsize_value_);
    }
};

class CPUKvPairs : public KvPairs {
public:
    CPUKvPairs(size_t capacity,
               size_t dsize_key,
               size_t dsize_value,
               const Device &device)
        : KvPairs(capacity, dsize_key, dsize_value, device) {
        context_ = std::make_shared<CPUKvPairsContext>();

        context_->capacity_ = capacity;
        context_->dsize_key_ = dsize_key;
        context_->dsize_value_ = dsize_value;

        context_->heap_ = static_cast<addr_t *>(
                MemoryManager::Malloc(capacity * sizeof(addr_t), device_));
        context_->keys_ = static_cast<uint8_t *>(
                MemoryManager::Malloc(capacity * dsize_key, device_));
        context_->values_ = static_cast<uint8_t *>(
                MemoryManager::Malloc(capacity * dsize_value, device_));

        ResetHeap();
    }

    ~CPUKvPairs() override {
        MemoryManager::Free(context_->heap_, device_);
        MemoryManager::Free(context_->keys_, device_);
        MemoryManager::Free(context_->values_, device_);
    }

    void ResetHeap() override {
        for (int i = 0; i < context_->capacity_; ++i) {
            context_->heap_[i] = i;
        }

        context_->heap_counter_ = 0;
        std::memset(context_->values_, 0, capacity_ * dsize_val_);
    }

    void *GetKeyBufferPtr() override { return context_->keys_; }
    void *GetValueBufferPtr() override { return context_->values_; }

    int heap_counter() override { return context_->heap_counter_.load(); }

    std::shared_ptr<CPUKvPairsContext> &GetContext() { return context_; }

protected:
    std::shared_ptr<CPUKvPairsContext> context_;
};
}  // namespace core
}  // namespace open3d
