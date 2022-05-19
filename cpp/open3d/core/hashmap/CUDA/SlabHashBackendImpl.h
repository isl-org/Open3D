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

// Copyright 2019 Saman Ashkiani
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing permissions
// and limitations under the License.

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/hashmap/CUDA/CUDAHashBackendBufferAccessor.h"
#include "open3d/core/hashmap/CUDA/SlabMacros.h"
#include "open3d/core/hashmap/CUDA/SlabNodeManager.h"
#include "open3d/core/hashmap/CUDA/SlabTraits.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"

namespace open3d {
namespace core {

// Each slab contains a collection of uint32_t entries.
// Each uint32_t entry can represent:
// 0) an empty placeholder;
// 1) a stored buf_index;
// 2) a ptr to the next slab if at the end of the slab.
// In case 0) and 1), it is interpreted as a buf_index_t.
// In case 2), it is interpreted as uint32_t.
// They are equivalent, but we differentiate them in the implementation to
// emphasize the differences.

template <typename Key, typename Hash, typename Eq>
class SlabHashBackendImpl {
public:
    SlabHashBackendImpl();

    __host__ void Setup(int64_t init_buckets,
                        const SlabNodeManagerImpl& node_mgr_impl,
                        const CUDAHashBackendBufferAccessor& buffer_accessor);

    /// Warp-insert a pre-allocated buf_index at key.
    __device__ bool Insert(bool lane_active,
                           uint32_t lane_id,
                           uint32_t bucket_id,
                           const Key& key,
                           buf_index_t buf_index);

    /// Warp-find a buf_index and its mask at key.
    __device__ Pair<buf_index_t, bool> Find(bool lane_active,
                                            uint32_t lane_id,
                                            uint32_t bucket_id,
                                            const Key& key);

    /// Warp-erase an entry at key.
    __device__ Pair<buf_index_t, bool> Erase(bool lane_active,
                                             uint32_t lane_id,
                                             uint32_t bucket_id,
                                             const Key& key);

    /// Warp-synchronize a key in a slab.
    __device__ void WarpSyncKey(const Key& key, uint32_t lane_id, Key& ret_key);

    /// Warp-find a key in a slab.
    __device__ int32_t WarpFindKey(const Key& src_key,
                                   uint32_t lane_id,
                                   uint32_t slab_entry);

    /// Warp-find the first empty slot in a slab.
    __device__ int32_t WarpFindEmpty(uint32_t slab_entry);

    // Hash function.
    __device__ int64_t ComputeBucket(const Key& key) const;

    // Node manager.
    __device__ uint32_t AllocateSlab(uint32_t lane_id);
    __device__ void FreeSlab(uint32_t slab_ptr);

    // Helpers.
    __device__ uint32_t* SlabEntryPtr(uint32_t bucket_id,
                                      uint32_t lane_id,
                                      uint32_t slab_ptr) {
        return (slab_ptr == kHeadSlabAddr)
                       ? SlabEntryPtrFromHead(bucket_id, lane_id)
                       : SlabEntryPtrFromNodes(slab_ptr, lane_id);
    }

    __device__ uint32_t* SlabEntryPtrFromNodes(uint32_t slab_ptr,
                                               uint32_t lane_id) {
        return node_mgr_impl_.get_unit_ptr_from_slab(slab_ptr, lane_id);
    }
    __device__ uint32_t* SlabEntryPtrFromHead(uint32_t bucket_id,
                                              uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(bucket_list_head_) +
               bucket_id * kWarpSize + lane_id;
    }

public:
    Hash hash_fn_;
    Eq eq_fn_;
    int64_t bucket_count_;

    Slab* bucket_list_head_;
    SlabNodeManagerImpl node_mgr_impl_;
    CUDAHashBackendBufferAccessor buffer_accessor_;

    // TODO: verify size with alignment
    int key_size_in_int_ = sizeof(Key) / sizeof(int);
};

/// Kernels
template <typename Key, typename Hash, typename Eq>
__global__ void InsertKernelPass0(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* input_keys,
                                  buf_index_t* output_buf_indices,
                                  int heap_counter_prev,
                                  int64_t count);

template <typename Key, typename Hash, typename Eq>
__global__ void InsertKernelPass1(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* input_keys,
                                  buf_index_t* output_buf_indices,
                                  bool* output_masks,
                                  int64_t count);

template <typename Key, typename Hash, typename Eq, typename block_t>
__global__ void InsertKernelPass2(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* const* input_values_soa,
                                  buf_index_t* output_buf_indices,
                                  bool* output_masks,
                                  int64_t count,
                                  int64_t n_values);

template <typename Key, typename Hash, typename Eq>
__global__ void FindKernel(SlabHashBackendImpl<Key, Hash, Eq> impl,
                           const void* input_keys,
                           buf_index_t* output_buf_indices,
                           bool* output_masks,
                           int64_t count);

template <typename Key, typename Hash, typename Eq>
__global__ void EraseKernelPass0(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                 const void* input_keys,
                                 buf_index_t* output_buf_indices,
                                 bool* output_masks,
                                 int64_t count);

template <typename Key, typename Hash, typename Eq>
__global__ void EraseKernelPass1(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                 buf_index_t* output_buf_indices,
                                 bool* output_masks,
                                 int64_t count);

template <typename Key, typename Hash, typename Eq>
__global__ void GetActiveIndicesKernel(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                       buf_index_t* output_buf_indices,
                                       uint32_t* output_count);

template <typename Key, typename Hash, typename Eq>
__global__ void CountElemsPerBucketKernel(
        SlabHashBackendImpl<Key, Hash, Eq> impl, int64_t* bucket_elem_counts);

template <typename Key, typename Hash, typename Eq>
SlabHashBackendImpl<Key, Hash, Eq>::SlabHashBackendImpl()
    : bucket_count_(0), bucket_list_head_(nullptr) {}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackendImpl<Key, Hash, Eq>::Setup(
        int64_t init_buckets,
        const SlabNodeManagerImpl& allocator_impl,
        const CUDAHashBackendBufferAccessor& buffer_accessor) {
    bucket_count_ = init_buckets;
    node_mgr_impl_ = allocator_impl;
    buffer_accessor_ = buffer_accessor;
}

template <typename Key, typename Hash, typename Eq>
__device__ bool SlabHashBackendImpl<Key, Hash, Eq>::Insert(
        bool lane_active,
        uint32_t lane_id,
        uint32_t bucket_id,
        const Key& key,
        buf_index_t buf_index) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t slab_ptr = kHeadSlabAddr;
    Key src_key;

    bool mask = false;

    // > Loop when we have active lanes
    while ((work_queue = __ballot_sync(kSyncLanesMask, lane_active))) {
        // 0. Restart from linked list head if last insertion is finished
        slab_ptr = (prev_work_queue != work_queue) ? kHeadSlabAddr : slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(kSyncLanesMask, bucket_id, src_lane, kWarpSize);
        WarpSyncKey(key, src_lane, src_key);

        uint32_t slab_entry = *SlabEntryPtr(src_bucket, lane_id, slab_ptr);

        int32_t lane_found = WarpFindKey(src_key, lane_id, slab_entry);
        int32_t lane_empty = WarpFindEmpty(slab_entry);

        // Branch 1: key already existing, ABORT
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                lane_active = false;
            }
        }

        // Branch 2: empty slot available, try to insert
        else if (lane_empty >= 0) {
            // Cannot merge if statements.
            // otherwise the warp flow will be interrupted.
            if (lane_id == src_lane) {
                // Now regard the entry as a value of buf_index
                const uint32_t* empty_entry_ptr =
                        SlabEntryPtr(src_bucket, lane_empty, slab_ptr);

                uint32_t old_empty_entry_value =
                        atomicCAS((unsigned int*)empty_entry_ptr,
                                  kEmptyNodeAddr, buf_index);

                // Branch 2.1: SUCCEED
                if (old_empty_entry_value == kEmptyNodeAddr) {
                    lane_active = false;
                    mask = true;
                }
                // Branch 2.2: failed: RESTART
                // In the consequent attempt,
                // > if the same key was inserted in this slot,
                //   we fall back to Branch 1;
                // > if a different key was inserted,
                //   we go to Branch 2 or 3.
            }
        }

        // Branch 3: nothing found in this slab, goto next slab
        else {
            // broadcast next slab
            uint32_t next_slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry,
                                                 kNextSlabPtrLaneId, kWarpSize);

            // Branch 3.1: next slab existing, RESTART at updated slab ptr
            if (next_slab_ptr != kEmptySlabAddr) {
                slab_ptr = next_slab_ptr;
            }

            // Branch 3.2: next slab empty, try to allocate one from the Slab
            // buffer.
            else {
                // Warp allocate, must be outside the condition clause.
                uint32_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == kNextSlabPtrLaneId) {
                    const uint32_t* next_slab_entry_ptr = SlabEntryPtr(
                            src_bucket, kNextSlabPtrLaneId, slab_ptr);

                    uint32_t old_next_slab_entry_value =
                            atomicCAS((unsigned int*)next_slab_entry_ptr,
                                      kEmptySlabAddr, new_next_slab_ptr);

                    // Branch 3.2.1: other thread has allocated,
                    // RESTART. In the consequent attempt, goto Branch 2.
                    if (old_next_slab_entry_value != kEmptySlabAddr) {
                        FreeSlab(new_next_slab_ptr);
                    }

                    // Branch 3.2.2: this thread allocated successfully.
                    // RESTART, goto Branch 2
                }
            }
        }

        prev_work_queue = work_queue;
    }

    return mask;
}

template <typename Key, typename Hash, typename Eq>
__device__ Pair<buf_index_t, bool> SlabHashBackendImpl<Key, Hash, Eq>::Find(
        bool lane_active,
        uint32_t lane_id,
        uint32_t bucket_id,
        const Key& query_key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t slab_ptr = kHeadSlabAddr;

    buf_index_t buf_index = kNullAddr;
    bool mask = false;

    // > Loop when we have active lanes.
    while ((work_queue = __ballot_sync(kSyncLanesMask, lane_active))) {
        // 0. Restart from linked list head if the last query is finished.
        slab_ptr = (prev_work_queue != work_queue) ? kHeadSlabAddr : slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(kSyncLanesMask, bucket_id, src_lane, kWarpSize);

        Key src_key;
        WarpSyncKey(query_key, src_lane, src_key);

        // Each lane in the warp reads a unit in the slab in parallel.
        const uint32_t slab_entry =
                *SlabEntryPtr(src_bucket, lane_id, slab_ptr);

        int32_t lane_found = WarpFindKey(src_key, lane_id, slab_entry);

        // 1. Found in this slab, SUCCEED.
        if (lane_found >= 0) {
            // broadcast found value
            uint32_t found_buf_index = __shfl_sync(kSyncLanesMask, slab_entry,
                                                   lane_found, kWarpSize);

            if (lane_id == src_lane) {
                lane_active = false;
                buf_index = found_buf_index;
                mask = true;
            }
        }

        // 2. Not found in this slab.
        else {
            // Broadcast next slab: lane 31 reads 'next'.
            uint32_t next_slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry,
                                                 kNextSlabPtrLaneId, kWarpSize);

            // 2.1. Next slab is empty, ABORT.
            if (next_slab_ptr == kEmptySlabAddr) {
                if (lane_id == src_lane) {
                    lane_active = false;
                }
            }
            // 2.2. Next slab exists, RESTART.
            else {
                slab_ptr = next_slab_ptr;
            }
        }

        prev_work_queue = work_queue;
    }

    return make_pair(buf_index, mask);
}

template <typename Key, typename Hash, typename Eq>
__device__ Pair<buf_index_t, bool> SlabHashBackendImpl<Key, Hash, Eq>::Erase(
        bool lane_active,
        uint32_t lane_id,
        uint32_t bucket_id,
        const Key& key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t slab_ptr = kHeadSlabAddr;
    Key src_key;

    buf_index_t buf_index = 0;
    bool mask = false;

    // > Loop when we have active lanes.
    while ((work_queue = __ballot_sync(kSyncLanesMask, lane_active))) {
        // 0. Restart from linked list head if last insertion is finished.
        slab_ptr = (prev_work_queue != work_queue) ? kHeadSlabAddr : slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(kSyncLanesMask, bucket_id, src_lane, kWarpSize);

        WarpSyncKey(key, src_lane, src_key);

        const uint32_t slab_entry =
                *SlabEntryPtr(src_bucket, lane_id, slab_ptr);

        int32_t lane_found = WarpFindKey(src_key, lane_id, slab_entry);

        // Branch 1: key found.
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                uint32_t* found_entry_ptr =
                        SlabEntryPtr(src_bucket, lane_found, slab_ptr);

                uint32_t old_found_entry_value = atomicExch(
                        (unsigned int*)found_entry_ptr, kEmptyNodeAddr);

                // Branch 1.2: other thread might have done the job,
                // avoid double free.
                mask = (old_found_entry_value != kEmptyNodeAddr);
                buf_index = old_found_entry_value;
            }
        } else {  // no matching slot found:
            uint32_t next_slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry,
                                                 kNextSlabPtrLaneId, kWarpSize);
            if (next_slab_ptr == kEmptySlabAddr) {
                // not found:
                if (lane_id == src_lane) {
                    lane_active = false;
                }
            } else {
                slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }

    return make_pair(buf_index, mask);
}

template <typename Key, typename Hash, typename Eq>
__device__ void SlabHashBackendImpl<Key, Hash, Eq>::WarpSyncKey(
        const Key& key, uint32_t lane_id, Key& ret_key) {
    auto dst_key_ptr = reinterpret_cast<int*>(&ret_key);
    auto src_key_ptr = reinterpret_cast<const int*>(&key);
    for (int i = 0; i < key_size_in_int_; ++i) {
        dst_key_ptr[i] =
                __shfl_sync(kSyncLanesMask, src_key_ptr[i], lane_id, kWarpSize);
    }
}

template <typename Key, typename Hash, typename Eq>
__device__ int32_t SlabHashBackendImpl<Key, Hash, Eq>::WarpFindKey(
        const Key& key, uint32_t lane_id, uint32_t slab_entry) {
    bool is_lane_found =
            // Select key lanes.
            ((1 << lane_id) & kNodePtrLanesMask)
            // Validate key buf_indices.
            && (slab_entry != kEmptyNodeAddr)
            // Find keys in buffer. Now slab_entry is interpreted as buf_index.
            &&
            eq_fn_(*static_cast<Key*>(buffer_accessor_.GetKeyPtr(slab_entry)),
                   key);

    return __ffs(__ballot_sync(kNodePtrLanesMask, is_lane_found)) - 1;
}

template <typename Key, typename Hash, typename Eq>
__device__ int32_t
SlabHashBackendImpl<Key, Hash, Eq>::WarpFindEmpty(uint32_t slab_entry) {
    bool is_lane_empty = (slab_entry == kEmptyNodeAddr);
    return __ffs(__ballot_sync(kNodePtrLanesMask, is_lane_empty)) - 1;
}

template <typename Key, typename Hash, typename Eq>
__device__ int64_t
SlabHashBackendImpl<Key, Hash, Eq>::ComputeBucket(const Key& key) const {
    return hash_fn_(key) % bucket_count_;
}

template <typename Key, typename Hash, typename Eq>
__device__ uint32_t
SlabHashBackendImpl<Key, Hash, Eq>::AllocateSlab(uint32_t lane_id) {
    return node_mgr_impl_.WarpAllocate(lane_id);
}

template <typename Key, typename Hash, typename Eq>
__device__ __forceinline__ void SlabHashBackendImpl<Key, Hash, Eq>::FreeSlab(
        uint32_t slab_ptr) {
    node_mgr_impl_.FreeUntouched(slab_ptr);
}

template <typename Key, typename Hash, typename Eq>
__global__ void InsertKernelPass0(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* input_keys,
                                  buf_index_t* output_buf_indices,
                                  int heap_counter_prev,
                                  int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < count) {
        // First write ALL input_keys to avoid potential thread conflicts.
        buf_index_t buf_index =
                impl.buffer_accessor_.heap_[heap_counter_prev + tid];
        void* key = impl.buffer_accessor_.GetKeyPtr(buf_index);
        *static_cast<Key*>(key) = input_keys_templated[tid];
        output_buf_indices[tid] = buf_index;
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void InsertKernelPass1(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* input_keys,
                                  buf_index_t* output_buf_indices,
                                  bool* output_masks,
                                  int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = tid & 0x1F;

    if (tid - lane_id >= count) {
        return;
    }

    impl.node_mgr_impl_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;
    buf_index_t buf_index = 0;

    // Dummy for warp sync.
    Key key;
    if (tid < count) {
        lane_active = true;
        key = input_keys_templated[tid];
        buf_index = output_buf_indices[tid];
        bucket_id = impl.ComputeBucket(key);
    }

    // Index out-of-bound threads still have to run for warp synchronization.
    bool mask = impl.Insert(lane_active, lane_id, bucket_id, key, buf_index);

    if (tid < count) {
        output_masks[tid] = mask;
    }
}

template <typename Key, typename Hash, typename Eq, typename block_t>
__global__ void InsertKernelPass2(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                  const void* const* input_values_soa,
                                  buf_index_t* output_buf_indices,
                                  bool* output_masks,
                                  int64_t count,
                                  int64_t n_values) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < count) {
        buf_index_t buf_index = output_buf_indices[tid];

        if (output_masks[tid]) {
            for (int j = 0; j < n_values; ++j) {
                int64_t blocks_per_element =
                        impl.buffer_accessor_.value_blocks_per_element_[j];

                block_t* dst_value = static_cast<block_t*>(
                        impl.buffer_accessor_.GetValuePtr(buf_index, j));
                const block_t* src_value =
                        static_cast<const block_t*>(input_values_soa[j]) +
                        blocks_per_element * tid;
                for (int b = 0; b < blocks_per_element; ++b) {
                    dst_value[b] = src_value[b];
                }
            }
        } else {
            impl.buffer_accessor_.DeviceFree(buf_index);
        }
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void FindKernel(SlabHashBackendImpl<Key, Hash, Eq> impl,
                           const void* input_keys,
                           buf_index_t* output_buf_indices,
                           bool* output_masks,
                           int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // This warp is idle.
    if ((tid - lane_id) >= count) {
        return;
    }

    // Initialize the memory allocator on each warp.
    impl.node_mgr_impl_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;

    // Dummy for warp sync
    Key key;
    Pair<buf_index_t, bool> result;

    if (tid < count) {
        lane_active = true;
        key = input_keys_templated[tid];
        bucket_id = impl.ComputeBucket(key);
    }

    result = impl.Find(lane_active, lane_id, bucket_id, key);

    if (tid < count) {
        output_buf_indices[tid] = result.first;
        output_masks[tid] = result.second;
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void EraseKernelPass0(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                 const void* input_keys,
                                 buf_index_t* output_buf_indices,
                                 bool* output_masks,
                                 int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    if (tid - lane_id >= count) {
        return;
    }

    impl.node_mgr_impl_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;

    // Dummy for warp sync
    Key key;
    if (tid < count) {
        lane_active = true;
        key = input_keys_templated[tid];
        bucket_id = impl.ComputeBucket(key);
    }

    auto result = impl.Erase(lane_active, lane_id, bucket_id, key);

    if (tid < count) {
        output_buf_indices[tid] = result.first;
        output_masks[tid] = result.second;
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void EraseKernelPass1(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                 buf_index_t* output_buf_indices,
                                 bool* output_masks,
                                 int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < count && output_masks[tid]) {
        impl.buffer_accessor_.DeviceFree(output_buf_indices[tid]);
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void GetActiveIndicesKernel(SlabHashBackendImpl<Key, Hash, Eq> impl,
                                       buf_index_t* output_buf_indices,
                                       uint32_t* output_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // Assigning a warp per bucket.
    uint32_t bucket_id = tid >> 5;
    if (bucket_id >= impl.bucket_count_) {
        return;
    }

    impl.node_mgr_impl_.Init(tid, lane_id);

    uint32_t slab_entry = *impl.SlabEntryPtrFromHead(bucket_id, lane_id);
    bool is_active = slab_entry != kEmptyNodeAddr;

    if (is_active && ((1 << lane_id) & kNodePtrLanesMask)) {
        uint32_t index = atomicAdd(output_count, 1);
        output_buf_indices[index] = slab_entry;
    }

    uint32_t slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry,
                                    kNextSlabPtrLaneId, kWarpSize);

    // Count following nodes,
    while (slab_ptr != kEmptySlabAddr) {
        slab_entry = *impl.SlabEntryPtrFromNodes(slab_ptr, lane_id);
        is_active = (slab_entry != kEmptyNodeAddr);

        if (is_active && ((1 << lane_id) & kNodePtrLanesMask)) {
            uint32_t index = atomicAdd(output_count, 1);
            output_buf_indices[index] = slab_entry;
        }
        slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry, kNextSlabPtrLaneId,
                               kWarpSize);
    }
}

template <typename Key, typename Hash, typename Eq>
__global__ void CountElemsPerBucketKernel(
        SlabHashBackendImpl<Key, Hash, Eq> impl, int64_t* bucket_elem_counts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // Assigning a warp per bucket.
    uint32_t bucket_id = tid >> 5;
    if (bucket_id >= impl.bucket_count_) {
        return;
    }

    impl.node_mgr_impl_.Init(tid, lane_id);

    uint32_t count = 0;

    // Count head node.
    uint32_t slab_entry = *impl.SlabEntryPtrFromHead(bucket_id, lane_id);
    count += __popc(
            __ballot_sync(kNodePtrLanesMask, slab_entry != kEmptyNodeAddr));
    uint32_t slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry,
                                    kNextSlabPtrLaneId, kWarpSize);

    // Count following nodes.
    while (slab_ptr != kEmptySlabAddr) {
        slab_entry = *impl.SlabEntryPtrFromNodes(slab_ptr, lane_id);
        count += __popc(
                __ballot_sync(kNodePtrLanesMask, slab_entry != kEmptyNodeAddr));
        slab_ptr = __shfl_sync(kSyncLanesMask, slab_entry, kNextSlabPtrLaneId,
                               kWarpSize);
    }

    // Write back the results.
    if (lane_id == 0) {
        bucket_elem_counts[bucket_id] = count;
    }
}

}  // namespace core
}  // namespace open3d
