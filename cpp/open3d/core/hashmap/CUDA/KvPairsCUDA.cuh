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
#include "open3d/core/hashmap/KvPairs.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {

/// Dynamic memory allocation and free are expensive on kernels.
/// We pre-allocate a chunk of memory and manually manage them on kernels.
class CUDAKvPairsContext {
public:
    uint8_t *keys_;     /* [N] * sizeof(Key) */
    uint8_t *values_;   /* [N] * sizeof(Value) */
    addr_t *heap_;      /* [N] */
    int *heap_counter_; /* [1] */

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

    __device__ addr_t Allocate() {
        int index = atomicAdd(heap_counter_, 1);
        return heap_[index];
    }

    __device__ void Free(addr_t ptr) {
        int index = atomicSub(heap_counter_, 1);
        heap_[index - 1] = ptr;
    }

    __device__ iterator_t extract_iterator(addr_t ptr) {
        return iterator_t(keys_ + ptr * dsize_key_,
                          values_ + ptr * dsize_value_);
    }
};

__global__ void ResetKvPairsKernel(CUDAKvPairsContext ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.capacity_) {
        ctx.heap_[i] = i;
    }
}

class CUDAKvPairs : public KvPairs {
public:
    CUDAKvPairs(size_t capacity,
                size_t dsize_key,
                size_t dsize_value,
                const Device &device)
        : KvPairs(capacity, dsize_key, dsize_value, device) {
        context_.heap_counter_ =
                static_cast<int *>(MemoryManager::Malloc(sizeof(int), device_));
        context_.heap_ = static_cast<addr_t *>(
                MemoryManager::Malloc(capacity * sizeof(addr_t), device_));

        context_.capacity_ = capacity;
        context_.dsize_key_ = dsize_key;
        context_.dsize_value_ = dsize_value;
        context_.keys_ = static_cast<uint8_t *>(key_blob_->GetDataPtr());
        context_.values_ = static_cast<uint8_t *>(val_blob_->GetDataPtr());

        ResetHeap();
    }

    ~CUDAKvPairs() override {
        MemoryManager::Free(context_.heap_counter_, device_);
        MemoryManager::Free(context_.heap_, device_);
    }

    void ResetHeap() override {
        const int blocks =
                (capacity_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
        ResetKvPairsKernel<<<blocks, kThreadsPerBlock>>>(context_);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        OPEN3D_CUDA_CHECK(cudaGetLastError());

        int heap_counter = 0;
        OPEN3D_CUDA_CHECK(
                cudaMemset(context_.values_, 0, capacity_ * dsize_val_));
        MemoryManager::Memcpy(context_.heap_counter_, device_, &heap_counter,
                              Device("CPU:0"), sizeof(int));
    }

    int heap_counter() override {
        int heap_counter;
        MemoryManager::Memcpy(&heap_counter, Device("CPU:0"),
                              context_.heap_counter_, device_, sizeof(int));
        return heap_counter;
    }

    CUDAKvPairsContext &GetContext() { return context_; }

    std::vector<int> DownloadHeap() {
        std::vector<int> ret;
        ret.resize(context_.capacity_);
        MemoryManager::Memcpy(ret.data(), Device("CPU:0"), context_.heap_,
                              device_, sizeof(int) * context_.capacity_);
        return ret;
    }

protected:
    CUDAKvPairsContext context_;
};
}  // namespace core
}  // namespace open3d
