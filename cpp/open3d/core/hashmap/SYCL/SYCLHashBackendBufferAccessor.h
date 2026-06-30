// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"

namespace open3d {
namespace core {

/// Device-side accessor for the external key/value buffer of a SYCL hash map.
///
/// The struct is captured by value into SYCL kernels, so it must remain
/// trivially copyable: it only holds raw USM pointers and scalar sizes. The
/// Setup/Shutdown methods run on the host; the Device* / Get* methods run on
/// the device inside kernels.
///
/// This mirrors CUDAHashBackendBufferAccessor, replacing CUDA atomics with
/// sycl::atomic_ref and CUDA memory ops with the SYCL queue / MemoryManager.
class SYCLHashBackendBufferAccessor {
public:
    static constexpr buf_index_t kInvalidBufIndex =
            static_cast<buf_index_t>(-1);

    void Setup(HashBackendBuffer &hashmap_buffer) {
        Device device = hashmap_buffer.GetDevice();
        sycl::queue queue =
                sy::SYCLContext::GetInstance().GetDefaultQueue(device);

        // Properties.
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

        // Pointers.
        heap_ = hashmap_buffer.GetIndexHeap().GetDataPtr<buf_index_t>();
        keys_ = hashmap_buffer.GetKeyBuffer().GetDataPtr<uint8_t>();

        std::vector<Tensor> value_buffers = hashmap_buffer.GetValueBuffers();
        std::vector<uint8_t *> value_ptrs(n_values_);
        for (size_t i = 0; i < n_values_; ++i) {
            value_ptrs[i] = value_buffers[i].GetDataPtr<uint8_t>();
            queue.memset(value_ptrs[i], 0, capacity_ * value_dsizes_host[i]);
        }
        queue.wait_and_throw();
        values_ = static_cast<uint8_t **>(
                MemoryManager::Malloc(n_values_ * sizeof(uint8_t *), device));
        MemoryManager::MemcpyFromHost(values_, device, value_ptrs.data(),
                                      n_values_ * sizeof(uint8_t *));

        heap_top_ = hashmap_buffer.GetHeapTop().cuda.GetDataPtr<int>();
    }

    void Shutdown(const Device &device) {
        MemoryManager::Free(values_, device);
        MemoryManager::Free(value_dsizes_, device);
        MemoryManager::Free(value_blocks_per_element_, device);
    }

    // Atomically pop a free buffer slot from the top of the heap.
    buf_index_t DeviceAllocate() const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
                top(*heap_top_);
        const int index = top.fetch_add(1);
        if (index >= static_cast<int>(capacity_)) {
            top.fetch_sub(1);
            return kInvalidBufIndex;
        }
        return heap_[index];
    }

    // Atomically push a buffer slot back to the top of the heap.
    void DeviceFree(buf_index_t buf_index) const {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
                top(*heap_top_);
        int index = top.fetch_sub(1);
        heap_[index - 1] = buf_index;
    }

    void *GetKeyPtr(buf_index_t buf_index) const {
        return keys_ + buf_index * key_dsize_;
    }
    void *GetValuePtr(buf_index_t buf_index, int value_idx = 0) const {
        return values_[value_idx] + buf_index * value_dsizes_[value_idx];
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
