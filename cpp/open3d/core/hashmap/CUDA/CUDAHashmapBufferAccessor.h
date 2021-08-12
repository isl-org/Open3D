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

class CUDAHashmapBufferAccessor {
public:
    __host__ void Setup(HashmapBuffer &hashmap_buffer) {
        Device device = hashmap_buffer.GetDevice();

        // Properties
        capacity_ = hashmap_buffer.GetCapacity();
        std::vector<int64_t> dsize_values_host =
                hashmap_buffer.GetValueDsizes();
        n_values_ = dsize_values_host.size();

        dsize_key_ = hashmap_buffer.GetKeyDsize();
        dsize_values_ = static_cast<int64_t *>(
                MemoryManager::Malloc(n_values_ * sizeof(int64_t), device));
        MemoryManager::MemcpyFromHost(dsize_values_, device,
                                      dsize_values_host.data(),
                                      n_values_ * sizeof(int64_t));

        // Pointers
        heap_ = hashmap_buffer.GetIndexHeap().GetDataPtr<buf_index_t>();
        keys_ = hashmap_buffer.GetKeyBuffer().GetDataPtr<uint8_t>();

        std::vector<Tensor> value_buffers = hashmap_buffer.GetValueBuffers();
        std::vector<uint8_t *> value_ptrs(n_values_);
        for (size_t i = 0; i < n_values_; ++i) {
            value_ptrs[i] = value_buffers[i].GetDataPtr<uint8_t>();
            cudaMemset(value_ptrs[i], 0, capacity_ * dsize_values_host[i]);
        }
        values_ = static_cast<uint8_t **>(
                MemoryManager::Malloc(n_values_ * sizeof(uint8_t *), device));
        MemoryManager::MemcpyFromHost(values_, device, value_ptrs.data(),
                                      n_values_ * sizeof(uint8_t *));

        heap_counter_ = hashmap_buffer.GetHeapTop().cuda.GetDataPtr<int>();
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void Shutdown(const Device &device) {
        MemoryManager::Free(values_, device);
        MemoryManager::Free(dsize_values_, device);
    }

    __host__ void Reset(const Device &device) {
        int heap_counter = 0;
        MemoryManager::MemcpyFromHost(heap_counter_, device, &heap_counter,
                                      sizeof(int));

        thrust::sequence(thrust::device, heap_, heap_ + capacity_, 0);
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __device__ buf_index_t DeviceAllocate() {
        int index = atomicAdd(heap_counter_, 1);
        return heap_[index];
    }
    __device__ void DeviceFree(buf_index_t ptr) {
        int index = atomicSub(heap_counter_, 1);
        heap_[index - 1] = ptr;
    }

    __device__ void *GetKeyPtr(buf_index_t ptr) {
        return keys_ + ptr * dsize_key_;
    }
    __device__ void *GetValuePtr(buf_index_t ptr, int value_idx = 0) {
        return values_[value_idx] + ptr * dsize_values_[value_idx];
    }

public:
    buf_index_t *heap_;           /* [N] */
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
