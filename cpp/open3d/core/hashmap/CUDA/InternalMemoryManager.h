// REVIEW: can this file be deleted? Looks like it is not used. Most
// functionalities are implemented in InternalKvPairManager.h.
//
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

/**
 * Created by wei on 18-3-29.
 */

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
/// For simplicity, we maintain a chunk per array (type T) instead of managing a
/// universal one. This causes more redundancy but is easier to maintain.
class InternalMemoryManagerContext {
public:
    uint8_t *data_;     /* [N] * sizeof(T) */
    ptr_t *heap_;       /* [N] */
    int *heap_counter_; /* [1] */

    int dsize_;
    int max_capacity_;

public:
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
    __device__ ptr_t Allocate() {
        int index = atomicAdd(heap_counter_, 1);
        assert(index < max_capacity_);
        return heap_[index];
    }

    __device__ void Free(ptr_t ptr) {
        int index = atomicSub(heap_counter_, 1);
        assert(index >= 1);
        heap_[index - 1] = ptr;
    }

    __device__ uint8_t *extract_ptr(ptr_t ptr) { return data_ + ptr * dsize_; }

    __device__ const uint8_t *extract_ptr(ptr_t ptr) const {
        return data_ + ptr * dsize_;
    }
};

__global__ void ResetInternalMemoryManagerKernel(
        InternalMemoryManagerContext ctx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ctx.max_capacity_) {
        ctx.heap_[i] = i;

        /// Memset
        for (int j = 0; j < ctx.dsize_; ++j) {
            ctx.data_[i * ctx.dsize_ + j] = 0;
        }
    }
}

class InternalMemoryManager {
public:
    int max_capacity_;
    int dsize_;
    InternalMemoryManagerContext gpu_context_;
    Device device_;

public:
    InternalMemoryManager(int max_capacity, int dsize, const Device &device) {
        device_ = device;
        max_capacity_ = max_capacity;
        dsize_ = dsize;

        gpu_context_.max_capacity_ = max_capacity;
        gpu_context_.dsize_ = dsize_;

        gpu_context_.heap_counter_ = static_cast<int *>(
                MemoryManager::Malloc(size_t(1) * sizeof(int), device_));
        gpu_context_.heap_ = static_cast<ptr_t *>(MemoryManager::Malloc(
                size_t(max_capacity_) * sizeof(ptr_t), device_));
        gpu_context_.data_ = static_cast<uint8_t *>(
                MemoryManager::Malloc(size_t(max_capacity_) * dsize_, device_));

        const int blocks = (max_capacity_ + 128 - 1) / 128;
        const int threads = 128;

        ResetInternalMemoryManagerKernel<<<blocks, threads>>>(gpu_context_);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        OPEN3D_CUDA_CHECK(cudaGetLastError());

        int heap_counter = 0;
        MemoryManager::Memcpy(gpu_context_.heap_counter_, device_,
                              &heap_counter, Device("CPU:0"), sizeof(int));
    }

    ~InternalMemoryManager() {
        MemoryManager::Free(gpu_context_.heap_counter_, device_);
        MemoryManager::Free(gpu_context_.heap_, device_);
        MemoryManager::Free(gpu_context_.data_, device_);
    }

    std::vector<int> DownloadHeap() {
        std::vector<int> ret;
        ret.resize(max_capacity_);
        MemoryManager::Memcpy(ret.data(), Device("CPU:0"), gpu_context_.heap_,
                              device_, sizeof(int) * max_capacity_);
        return ret;
    }

    std::vector<uint8_t> DownloadValue() {
        std::vector<uint8_t> ret;
        ret.resize(max_capacity_);
        MemoryManager::Memcpy(ret.data(), Device("CPU:0"), gpu_context_.data_,
                              device_, max_capacity_ * dsize_);
        return ret;
    }

    int heap_counter() {
        int heap_counter;
        MemoryManager::Memcpy(&heap_counter, Device("CPU:0"),
                              gpu_context_.heap_counter_, device_, sizeof(int));
        return heap_counter;
    }
};
}  // namespace core
}  // namespace open3d
