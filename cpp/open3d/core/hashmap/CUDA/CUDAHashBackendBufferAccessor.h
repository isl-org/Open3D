// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <assert.h>

#include <memory>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/SlabMacros.h"
#include "open3d/core/hashmap/CUDA/SlabTraits.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"

namespace open3d {
namespace core {

class CUDAHashBackendBufferAccessor {
public:
    __host__ void Setup(HashBackendBuffer &hashmap_buffer) {
        Device device = hashmap_buffer.GetDevice();

        // Properties
        capacity_ = hashmap_buffer.GetCapacity();
        key_dsize_ = hashmap_buffer.GetKeyDsize();

        std::vector<int64_t> value_dsizes_host =
                hashmap_buffer.GetValueDsizes();
        std::vector<int64_t> value_blocks_per_element_host =
                hashmap_buffer.GetValueBlocksPerElement();
        n_values_ = value_blocks_per_element_host.size();

        value_dsizes_ = static_cast<int64_t *>(
                MemoryManager::Malloc(n_values_ * sizeof(int64_t), device));
        value_blocks_per_element_ = static_cast<int64_t *>(
                MemoryManager::Malloc(n_values_ * sizeof(int64_t), device));

        MemoryManager::MemcpyFromHost(value_dsizes_, device,
                                      value_dsizes_host.data(),
                                      n_values_ * sizeof(int64_t));
        MemoryManager::MemcpyFromHost(value_blocks_per_element_, device,
                                      value_blocks_per_element_host.data(),
                                      n_values_ * sizeof(int64_t));

        common_block_size_ = hashmap_buffer.GetCommonBlockSize();

        // Pointers
        heap_ = hashmap_buffer.GetIndexHeap().GetDataPtr<buf_index_t>();
        keys_ = hashmap_buffer.GetKeyBuffer().GetDataPtr<uint8_t>();

        std::vector<Tensor> value_buffers = hashmap_buffer.GetValueBuffers();
        std::vector<uint8_t *> value_ptrs(n_values_);
        for (size_t i = 0; i < n_values_; ++i) {
            value_ptrs[i] = value_buffers[i].GetDataPtr<uint8_t>();
            cudaMemset(value_ptrs[i], 0, capacity_ * value_dsizes_host[i]);
        }
        values_ = static_cast<uint8_t **>(
                MemoryManager::Malloc(n_values_ * sizeof(uint8_t *), device));
        MemoryManager::MemcpyFromHost(values_, device, value_ptrs.data(),
                                      n_values_ * sizeof(uint8_t *));

        heap_top_ = hashmap_buffer.GetHeapTop().cuda.GetDataPtr<int>();
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void Shutdown(const Device &device) {
        MemoryManager::Free(values_, device);
        MemoryManager::Free(value_dsizes_, device);
        MemoryManager::Free(value_blocks_per_element_, device);
    }

    __device__ buf_index_t DeviceAllocate() {
        int index = atomicAdd(heap_top_, 1);
        return heap_[index];
    }
    __device__ void DeviceFree(buf_index_t ptr) {
        int index = atomicSub(heap_top_, 1);
        heap_[index - 1] = ptr;
    }

    __device__ void *GetKeyPtr(buf_index_t ptr) {
        return keys_ + ptr * key_dsize_;
    }
    __device__ void *GetValuePtr(buf_index_t ptr, int value_idx = 0) {
        return values_[value_idx] + ptr * value_dsizes_[value_idx];
    }

public:
    buf_index_t *heap_;       /* [N] */
    int *heap_top_ = nullptr; /* [1] */

    uint8_t *keys_; /* [N] * sizeof(Key) */
    int64_t key_dsize_;

    size_t n_values_;
    uint8_t **values_;

    int64_t common_block_size_;

    int64_t *value_dsizes_;
    int64_t *value_blocks_per_element_;

    int64_t capacity_;
};

}  // namespace core
}  // namespace open3d
