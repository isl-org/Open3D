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
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
/// Dynamic memory allocation and free are expensive on kernels.
/// We pre-allocate a chunk of memory and manually manage them on kernels.
// REVIEW: type T?
/// For simplicity, we maintain a chunk per array (type T) instead of managing a
/// universal one. This causes more redundancy but is easier to maintain.
class InternalKvPairManagerContext {
public:
    uint8_t *keys_;     /* [N] * sizeof(Key) */
    uint8_t *values_;   /* [N] * sizeof(Value) */
    ptr_t *heap_;       /* [N] */
    int *heap_counter_; /* [1] */

public:
    int dsize_key_;
    int dsize_value_;
    int max_capacity_;

public:
    // REVIEW: rename @value -> values_; @heap -> heap_
    // The @value array's size is FIXED.
    // The @heap array stores the addresses of the values.
    // Only the unallocated part is maintained.
    // (ONLY care about the heap above the heap counter. Below is
    // meaningless.)
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

    // REVIEW: Not used, remove?
    __device__ ptr_t Allocate() {
        int index = atomicAdd(heap_counter_, 1);
        return heap_[index];
    }

    // REVIEW: function not used, remove?
    __device__ ptr_t SafeAllocate() {
        int index = atomicAdd(heap_counter_, 1);
        assert(index < max_capacity_);
        return heap_[index];
    }

    __device__ void Free(ptr_t ptr) {
        // REVIEW: equivalent to
        // ```
        // atomicSub(heap_counter_, 1);
        // heap_[heap_counter_] = ptr;
        // ```
        // ?
        int index = atomicSub(heap_counter_, 1);
        // REVIEW: add comment on why we need to assign ptr to the heap entry.
        heap_[index - 1] = ptr;
    }

    // REVIEW: function not used, remove?
    __device__ void SafeFree(ptr_t ptr) {
        int index = atomicSub(heap_counter_, 1);
        assert(index >= 1);
        heap_[index - 1] = ptr;
    }

    __device__ iterator_t extract_iterator(ptr_t ptr) {
        return iterator_t(keys_ + ptr * dsize_key_,
                          values_ + ptr * dsize_value_);
    }

    // REVIEW: the const version is not used, remove?
    __device__ const iterator_t extract_iterator(ptr_t ptr) const {
        return iterator_t(keys_ + ptr * dsize_key_,
                          values_ + ptr * dsize_value_);
    }

    // REVIEW: function not used, remove?
    // Or, in the caller, like InsertKernelPass0, actually call this function.
    __device__ iterator_t extract_iterator_from_heap_index(int index) {
        ptr_t ptr = heap_[index];
        return extract_iterator(ptr);
    }

    // REVIEW: function not used, remove?
    __device__ const iterator_t
    extract_iterator_from_heap_index(int index) const {
        ptr_t ptr = heap_[index];
        return extract_iterator(ptr);
    }
};

__global__ void ResetInternalKvPairManagerKernel(
        InternalKvPairManagerContext ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.max_capacity_) {
        ctx.heap_[i] = i;
    }
}

class InternalKvPairManager {
public:
    int max_capacity_;
    int dsize_key_;
    int dsize_value_;
    InternalKvPairManagerContext gpu_context_;
    Device device_;

public:
    // REVIEW: max_capacity -> capacity
    InternalKvPairManager(int max_capacity,
                          int dsize_key,
                          int dsize_value,
                          const Device &device) {
        // REVIEW: use initializer list?
        device_ = device;
        // REVIEW: why are we keeping two copies of these varaibles? One in
        // InternalKvPairManager and one in InternalKvPairManager::gpu_context_.
        // Can we just retrive it from the gpu_context_?
        max_capacity_ = max_capacity;
        dsize_key_ = dsize_key;
        dsize_value_ = dsize_value;

        gpu_context_.max_capacity_ = max_capacity;
        gpu_context_.dsize_key_ = dsize_key;
        gpu_context_.dsize_value_ = dsize_value;

        gpu_context_.heap_counter_ =
                static_cast<int *>(MemoryManager::Malloc(sizeof(int), device_));
        gpu_context_.heap_ = static_cast<ptr_t *>(
                MemoryManager::Malloc(max_capacity_ * sizeof(ptr_t), device_));
        gpu_context_.keys_ = static_cast<uint8_t *>(
                MemoryManager::Malloc(max_capacity_ * dsize_key_, device_));
        gpu_context_.values_ = static_cast<uint8_t *>(
                MemoryManager::Malloc(max_capacity_ * dsize_value_, device_));

        // REVIEW: use the global BLOCKSIZE_ for now?
        // TODO: auto tune
        const int blocks = (max_capacity_ + 128 - 1) / 128;
        const int threads = 128;

        ResetInternalKvPairManagerKernel<<<blocks, threads>>>(gpu_context_);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        OPEN3D_CUDA_CHECK(cudaGetLastError());

        int heap_counter = 0;
        // REVIEW: OPEN3D_CUDA_CHECK
        cudaMemset(gpu_context_.values_, 0, max_capacity_ * dsize_value_);
        MemoryManager::Memcpy(gpu_context_.heap_counter_, device_,
                              &heap_counter, Device("CPU:0"), sizeof(int));
    }

    ~InternalKvPairManager() {
        MemoryManager::Free(gpu_context_.heap_counter_, device_);
        MemoryManager::Free(gpu_context_.heap_, device_);
        MemoryManager::Free(gpu_context_.keys_, device_);
        MemoryManager::Free(gpu_context_.values_, device_);
    }

    // void FillZero() { cudaMemset(gpu_conetxt_.values_, 0, max_capacity_ *); }

    std::vector<int> DownloadHeap() {
        std::vector<int> ret;
        ret.resize(max_capacity_);
        MemoryManager::Memcpy(ret.data(), Device("CPU:0"), gpu_context_.heap_,
                              device_, sizeof(int) * max_capacity_);
        return ret;
    }

    int heap_counter() {
        int heap_counter;
        MemoryManager::Memcpy(&heap_counter, Device("CPU:0"),
                              gpu_context_.heap_counter_, device_, sizeof(int));
        return heap_counter;
    }

    /* std::vector<uint8_t> DownloadValue() { */
    /*     std::vector<uint8_t> ret; */
    /*     ret.resize(max_capacity_); */
    /*     MemoryManager::Memcpy(ret.data(), Device("CPU:0"),
     * gpu_context_.data_, */
    /*                           device_, max_capacity_ * dsize_); */
    /*     return ret; */
    /* } */
};
}  // namespace core
}  // namespace open3d
