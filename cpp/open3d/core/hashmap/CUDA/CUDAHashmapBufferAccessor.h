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
#include "open3d/core/hashmap/CUDA/SlabMacros.h"
#include "open3d/core/hashmap/CUDA/SlabTraits.h"
#include "open3d/core/hashmap/HashmapBuffer.h"

namespace open3d {
namespace core {

/// Dynamic memory allocation and free are expensive on kernels.
/// We pre-allocate a chunk of memory and manually manage them on kernels.

__global__ void ResetHashmapBufferKernel(addr_t *heap, int64_t capacity);

class CUDAHashmapBufferAccessor {
public:
    __host__ void Setup(int64_t capacity,
                        int64_t dsize_key,
                        int64_t dsize_value,
                        Tensor &keys,
                        Tensor &values,
                        Tensor &heap) {
        capacity_ = capacity;
        dsize_key_ = dsize_key;
        dsize_value_ = dsize_value;
        keys_ = keys.GetDataPtr<uint8_t>();
        values_ = values.GetDataPtr<uint8_t>();
        heap_ = static_cast<addr_t *>(heap.GetDataPtr());
        OPEN3D_CUDA_CHECK(cudaMemset(values_, 0, capacity_ * dsize_value_));
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void Reset(const Device &device) {
        int heap_counter = 0;
        MemoryManager::Memcpy(heap_counter_, device, &heap_counter,
                              Device("CPU:0"), sizeof(int));

        const int blocks =
                (capacity_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
        ResetHashmapBufferKernel<<<blocks, kThreadsPerBlock>>>(heap_,
                                                               capacity_);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HostAllocate(const Device &device) {
        heap_counter_ =
                static_cast<int *>(MemoryManager::Malloc(sizeof(int), device));
    }

    __host__ void HostFree(const Device &device) {
        if (heap_counter_ != nullptr) {
            MemoryManager::Free(heap_counter_, device);
        }
        heap_counter_ = nullptr;
    }

    __device__ addr_t DeviceAllocate() {
        int index = atomicAdd(heap_counter_, 1);
        return heap_[index];
    }

    __device__ void DeviceFree(addr_t ptr) {
        int index = atomicSub(heap_counter_, 1);
        heap_[index - 1] = ptr;
    }

    __host__ int HeapCounter(const Device &device) const {
        int heap_counter;
        MemoryManager::Memcpy(&heap_counter, Device("CPU:0"), heap_counter_,
                              device, sizeof(int));
        return heap_counter;
    }

    __device__ iterator_t ExtractIterator(addr_t ptr) {
        return iterator_t(keys_ + ptr * dsize_key_,
                          values_ + ptr * dsize_value_);
    }

public:
    uint8_t *keys_;               /* [N] * sizeof(Key) */
    uint8_t *values_;             /* [N] * sizeof(Value) */
    addr_t *heap_;                /* [N] */
    int *heap_counter_ = nullptr; /* [1] */

    int64_t dsize_key_;
    int64_t dsize_value_;
    int64_t capacity_;
};

}  // namespace core
}  // namespace open3d
