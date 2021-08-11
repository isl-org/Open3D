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
#include <thrust/device_vector.h>

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
                        std::vector<int64_t> dsize_values,
                        Tensor &keys,
                        std::vector<Tensor> &values,
                        Tensor &heap) {
        capacity_ = capacity;
        heap_ = static_cast<addr_t *>(heap.GetDataPtr());

        keys_ = keys.GetDataPtr<uint8_t>();
        dsize_key_ = dsize_key;

        Device device = keys.GetDevice();

        n_values_ = dsize_values.size();

        // Copy value sizes
        dsize_values_ = static_cast<int64_t *>(
                MemoryManager::Malloc(n_values_ * sizeof(int64_t), device));
        MemoryManager::MemcpyFromHost(dsize_values_, device,
                                      dsize_values.data(),
                                      n_values_ * sizeof(int64_t));

        // Copy values
        std::vector<uint8_t *> value_ptrs(n_values_);
        for (size_t i = 0; i < n_values_; ++i) {
            value_ptrs[i] = static_cast<uint8_t *>(values[i].GetDataPtr());
            // std::cout << "input pointer " << i << " : " << (void
            // *)value_ptrs[i]
            //           << "\n";
        }
        values_ = static_cast<uint8_t **>(
                MemoryManager::Malloc(n_values_ * sizeof(uint8_t *), device));
        MemoryManager::MemcpyFromHost(values_, device, value_ptrs.data(),
                                      n_values_ * sizeof(uint8_t *));

        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void Shutdown(const Device &device) {
        MemoryManager::Free(values_, device);
        MemoryManager::Free(dsize_values_, device);
    }

    __host__ void Reset(const Device &device) {
        int heap_counter = 0;
        MemoryManager::Memcpy(heap_counter_, device, &heap_counter,
                              Device("CPU:0"), sizeof(int));

        const int blocks =
                (capacity_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
        ResetHashmapBufferKernel<<<blocks, kThreadsPerBlock, 0,
                                   core::cuda::GetStream()>>>(heap_, capacity_);
        cuda::Synchronize();
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

    __device__ void *GetKeyPtr(addr_t ptr) { return keys_ + ptr * dsize_key_; }
    __device__ void *GetValuePtr(addr_t ptr, int value_idx = 0) {
        // printf("device pointer %d: %p, dsize: %ld\n", value_idx,
        //        (void *)values_[value_idx], dsize_values_[value_idx]);
        return values_[value_idx] + ptr * dsize_values_[value_idx];
    }

public:
    addr_t *heap_;                /* [N] */
    int *heap_counter_ = nullptr; /* [1] */

    uint8_t *keys_; /* [N] * sizeof(Key) */
    int64_t dsize_key_;

    size_t n_values_;
    uint8_t **values_;
    int64_t *dsize_values_;

    int64_t capacity_;
};

}  // namespace core
}  // namespace open3d
