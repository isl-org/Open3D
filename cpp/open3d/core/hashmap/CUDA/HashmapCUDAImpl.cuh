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

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
CUDAHashmapImplContext<Hash, KeyEq>::CUDAHashmapImplContext()
    : bucket_count_(0), bucket_list_head_(nullptr) {}

template <typename Hash, typename KeyEq>
void CUDAHashmapImplContext<Hash, KeyEq>::Setup(
        size_t init_buckets,
        size_t init_capacity,
        size_t dsize_key,
        size_t dsize_value,
        const InternalNodeManagerContext& allocator_ctx,
        const InternalKvPairManagerContext& pair_allocator_ctx) {
    bucket_count_ = init_buckets;
    capacity_ = init_capacity;
    dsize_key_ = dsize_key;
    dsize_value_ = dsize_value;

    node_mgr_ctx_ = allocator_ctx;
    mem_mgr_ctx_ = pair_allocator_ctx;

    hash_fn_.key_size_in_int_ = dsize_key / sizeof(int);
    cmp_fn_.key_size_in_int_ = dsize_key / sizeof(int);
}

/// Device functions
template <typename Hash, typename KeyEq>
__device__ size_t
CUDAHashmapImplContext<Hash, KeyEq>::ComputeBucket(const void* key) const {
    return hash_fn_(key) % bucket_count_;
}

template <typename Hash, typename KeyEq>
__device__ void CUDAHashmapImplContext<Hash, KeyEq>::WarpSyncKey(
        const void* key_ptr, uint32_t lane_id, void* ret_key_ptr) {
    // REVIEW: can we directly use hash_fn_.key_size_in_int_? If yes, we can
    // apply this change to the rest of this file. Same with
    // cmp_fn_.key_size_in_int_.
    //
    // REVIEW: we should probably be more consistent with int v.s. int32_t v.s.
    // size_t. Here __shfl_sync works with 32-bit, so probably we should
    // use int32_t when we want to indicate this.
    const int chunks = dsize_key_ / sizeof(int);

    auto src_key_ptr = static_cast<const int*>(key_ptr);
    auto dst_key_ptr = static_cast<int*>(ret_key_ptr);
    // REVIEW: int instead of size_t?
    for (size_t i = 0; i < chunks; ++i) {
        dst_key_ptr[i] = __shfl_sync(ACTIVE_LANES_MASK, src_key_ptr[i], lane_id,
                                     WARP_WIDTH);
    }
}

template <typename Hash, typename KeyEq>
__device__ int32_t CUDAHashmapImplContext<Hash, KeyEq>::WarpFindKey(
        const void* key_ptr, uint32_t lane_id, ptr_t ptr) {
    bool is_lane_found =
            /* select key lanes */
            ((1 << lane_id) & PAIR_PTR_LANES_MASK)
            /* validate key addrs */
            && (ptr != EMPTY_PAIR_PTR)
            /* find keys in memory heap */
            && cmp_fn_(mem_mgr_ctx_.extract_iterator(ptr).first, key_ptr);

    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_found)) - 1;
}

template <typename Hash, typename KeyEq>
__device__ int32_t
CUDAHashmapImplContext<Hash, KeyEq>::WarpFindEmpty(ptr_t ptr) {
    bool is_lane_empty = (ptr == EMPTY_PAIR_PTR);
    return __ffs(__ballot_sync(PAIR_PTR_LANES_MASK, is_lane_empty)) - 1;
}

template <typename Hash, typename KeyEq>
__device__ ptr_t
CUDAHashmapImplContext<Hash, KeyEq>::AllocateSlab(uint32_t lane_id) {
    return node_mgr_ctx_.WarpAllocate(lane_id);
}

template <typename Hash, typename KeyEq>
__device__ __forceinline__ void CUDAHashmapImplContext<Hash, KeyEq>::FreeSlab(
        ptr_t slab_ptr) {
    node_mgr_ctx_.FreeUntouched(slab_ptr);
}

template <typename Hash, typename KeyEq>
__device__ Pair<ptr_t, bool> CUDAHashmapImplContext<Hash, KeyEq>::Find(
        bool to_search,
        uint32_t lane_id,
        uint32_t bucket_id,
        const void* query_key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = work_queue;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;

    ptr_t iterator = NULL_ITERATOR;
    bool mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_search))) {
        /** 0. Restart from linked list head if the last query is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        uint8_t src_key[MAX_KEY_BYTESIZE];
        WarpSyncKey(query_key, src_lane, src_key);

        // REVIEW: uint or unit?
        /* Each lane in the warp reads a uint in the slab in parallel */
        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** 1. Found in this slab, SUCCEED **/
        if (lane_found >= 0) {
            /* broadcast found value */
            ptr_t found_pair_internal_ptr = __shfl_sync(
                    ACTIVE_LANES_MASK, unit_data, lane_found, WARP_WIDTH);

            if (lane_id == src_lane) {
                to_search = false;

                /// Actually iterator_ptr
                iterator = found_pair_internal_ptr;
                mask = true;
            }
        }

        /** 2. Not found in this slab **/
        else {
            /* broadcast next slab: lane 31 reads 'next' */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** 2.1. Next slab is empty, ABORT **/
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                if (lane_id == src_lane) {
                    to_search = false;
                }
            }
            /** 2.2. Next slab exists, RESTART **/
            else {
                curr_slab_ptr = next_slab_ptr;
            }
        }

        prev_work_queue = work_queue;
    }

    return make_pair(iterator, mask);
}

// REVIEW: update comments: replacePair?
/*
 * Insert: ABORT if found
 * replacePair: REPLACE if found
 * WE DO NOT ALLOW DUPLICATE KEYSn
 */
template <typename Hash, typename KeyEq>
__device__ bool CUDAHashmapImplContext<Hash, KeyEq>::Insert(
        bool to_be_inserted,
        uint32_t lane_id,
        uint32_t bucket_id,
        const void* key,
        ptr_t iterator_ptr) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;
    uint8_t src_key[MAX_KEY_BYTESIZE];

    bool mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_inserted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        WarpSyncKey(key, src_lane, src_key);

        // REVIEW: uint or unit?
        /* Each lane in the warp reads a uint in the slab */
        uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);
        int32_t lane_empty = WarpFindEmpty(unit_data);

        /** Branch 1: key already existing, ABORT **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                /* free memory heap */
                to_be_inserted = false;
            }
        }

        /** Branch 2: empty slot available, try to insert **/
        else if (lane_empty >= 0) {
            if (lane_id == src_lane) {
                // TODO: check why we cannot put malloc here
                const uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_empty)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_empty);

                ptr_t old_iterator_ptr =
                        atomicCAS((unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR,
                                  iterator_ptr);

                // Remember to clean up in another pass
                /** Branch 2.1: SUCCEED **/
                if (old_iterator_ptr == EMPTY_PAIR_PTR) {
                    to_be_inserted = false;
                    mask = true;
                }
                /** Branch 2.2: failed: RESTART
                 *  In the consequent attempt,
                 *  > if the same key was inserted in this slot,
                 *    we fall back to Branch 1;
                 *  > if a different key was inserted,
                 *    we go to Branch 2 or 3.
                 * **/
            }
        }

        /** Branch 3: nothing found in this slab, goto next slab **/
        else {
            /* broadcast next slab */
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);

            /** Branch 3.1: next slab existing, RESTART this lane **/
            if (next_slab_ptr != EMPTY_SLAB_PTR) {
                curr_slab_ptr = next_slab_ptr;
            }

            /** Branch 3.2: next slab empty, try to allocate one **/
            else {
                ptr_t new_next_slab_ptr = AllocateSlab(lane_id);

                if (lane_id == NEXT_SLAB_PTR_LANE) {
                    const uint32_t* unit_data_ptr =
                            (curr_slab_ptr == HEAD_SLAB_PTR)
                                    ? get_unit_ptr_from_list_head(
                                              src_bucket, NEXT_SLAB_PTR_LANE)
                                    : get_unit_ptr_from_list_nodes(
                                              curr_slab_ptr,
                                              NEXT_SLAB_PTR_LANE);

                    ptr_t old_next_slab_ptr =
                            atomicCAS((unsigned int*)unit_data_ptr,
                                      EMPTY_SLAB_PTR, new_next_slab_ptr);

                    /** Branch 3.2.1: other thread allocated, RESTART lane
                     *  In the consequent attempt, goto Branch 2' **/
                    if (old_next_slab_ptr != EMPTY_SLAB_PTR) {
                        FreeSlab(new_next_slab_ptr);
                    }
                    /** Branch 3.2.2: this thread allocated, RESTART lane,
                     * 'goto Branch 2' **/
                }
            }
        }

        prev_work_queue = work_queue;
    }

    return mask;
}

template <typename Hash, typename KeyEq>
__device__ Pair<ptr_t, bool> CUDAHashmapImplContext<Hash, KeyEq>::Erase(
        bool to_be_deleted,
        uint32_t lane_id,
        uint32_t bucket_id,
        const void* key) {
    uint32_t work_queue = 0;
    uint32_t prev_work_queue = 0;
    uint32_t curr_slab_ptr = HEAD_SLAB_PTR;
    uint8_t src_key[MAX_KEY_BYTESIZE];

    ptr_t iterator_ptr = 0;
    bool mask = false;

    /** > Loop when we have active lanes **/
    while ((work_queue = __ballot_sync(ACTIVE_LANES_MASK, to_be_deleted))) {
        /** 0. Restart from linked list head if last insertion is finished
         * **/
        curr_slab_ptr =
                (prev_work_queue != work_queue) ? HEAD_SLAB_PTR : curr_slab_ptr;
        uint32_t src_lane = __ffs(work_queue) - 1;
        uint32_t src_bucket =
                __shfl_sync(ACTIVE_LANES_MASK, bucket_id, src_lane, WARP_WIDTH);

        WarpSyncKey(key, src_lane, src_key);

        const uint32_t unit_data =
                (curr_slab_ptr == HEAD_SLAB_PTR)
                        ? *(get_unit_ptr_from_list_head(src_bucket, lane_id))
                        : *(get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                         lane_id));

        int32_t lane_found = WarpFindKey(src_key, lane_id, unit_data);

        /** Branch 1: key found **/
        if (lane_found >= 0) {
            if (lane_id == src_lane) {
                uint32_t* unit_data_ptr =
                        (curr_slab_ptr == HEAD_SLAB_PTR)
                                ? get_unit_ptr_from_list_head(src_bucket,
                                                              lane_found)
                                : get_unit_ptr_from_list_nodes(curr_slab_ptr,
                                                               lane_found);

                uint32_t pair_to_delete = atomicExch(
                        (unsigned int*)unit_data_ptr, EMPTY_PAIR_PTR);
                mask = pair_to_delete != EMPTY_PAIR_PTR;
                iterator_ptr = pair_to_delete;
                /** Branch 1.2: other thread did the job, avoid double free
                 * **/
            }
        } else {  // no matching slot found:
            ptr_t next_slab_ptr = __shfl_sync(ACTIVE_LANES_MASK, unit_data,
                                              NEXT_SLAB_PTR_LANE, WARP_WIDTH);
            if (next_slab_ptr == EMPTY_SLAB_PTR) {
                // not found:
                if (lane_id == src_lane) {
                    to_be_deleted = false;
                }
            } else {
                curr_slab_ptr = next_slab_ptr;
            }
        }
        prev_work_queue = work_queue;
    }

    return make_pair(iterator_ptr, mask);
}

template <typename Hash, typename KeyEq>
__global__ void FindKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                           const void* keys,
                           iterator_t* iterators,
                           bool* masks,
                           size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    /* This warp is idle */
    if ((tid - lane_id) >= input_count) {
        return;
    }

    /* Initialize the memory allocator on each warp */
    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;

    // dummy
    uint8_t dummy_key[MAX_KEY_BYTESIZE];
    const void* key = reinterpret_cast<const void*>(dummy_key);
    Pair<ptr_t, bool> result;

    if (tid < input_count) {
        lane_active = true;
        key = static_cast<const uint8_t*>(keys) + tid * hash_ctx.dsize_key_;
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    result = hash_ctx.Find(lane_active, lane_id, bucket_id, key);

    if (tid < input_count) {
        iterators[tid] = hash_ctx.mem_mgr_ctx_.extract_iterator(result.first);
        masks[tid] = result.second;
    }
}

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* keys,
                                  ptr_t* iterator_ptrs,
                                  int heap_counter_prev,
                                  size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < input_count) {
        /** First write ALL keys to avoid potential thread conflicts **/
        // ptr_t iterator_ptr = hash_ctx.mem_mgr_ctx_.Allocate();
        // REVIEW: this is equivalent to extract_iterator_from_heap_index?
        ptr_t iterator_ptr =
                hash_ctx.mem_mgr_ctx_.heap_[heap_counter_prev + tid];
        iterator_t iterator =
                hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);

        auto dst_key_ptr = static_cast<int*>(iterator.first);
        auto src_key_ptr = static_cast<const int*>(keys) +
                           tid * hash_ctx.dsize_key_ / sizeof(int);
        // REVIEW: This happens many times, woult it help to do a macro? e.g.
        // #define MEMCPY_AS_INTS(dst, src, num_bytes)
        for (int i = 0; i < hash_ctx.dsize_key_ / sizeof(int); ++i) {
            dst_key_ptr[i] = src_key_ptr[i];
        }

        // if (input_count < 100) {
        //     int64_t key0 = *reinterpret_cast<int64_t*>(dst_key_ptr);
        //     int64_t key1 = *(reinterpret_cast<int64_t*>(dst_key_ptr) + 1);
        //     int64_t key2 = *(reinterpret_cast<int64_t*>(dst_key_ptr) + 2);
        //     printf("pass0 %d: %ld %ld %ld\n", tid, key0, key1, key2);
        // }

        iterator_ptrs[tid] = iterator_ptr;
    }
}

// REVIEW: rename parameters. To be consistent with the caller, masks ->
// output_masks.
template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* keys,
                                  ptr_t* iterator_ptrs,
                                  bool* masks,
                                  size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    // REVIEW: use WARP_SIZE instead of 32, or use & 0x1F to be consistent?
    uint32_t lane_id = tid % 32;

    // REVIEW: tid - lane_id >= input_coun.
    if ((tid - lane_id) >= input_count) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;
    ptr_t iterator_ptr = 0;

    // dummy
    uint8_t dummy_key[MAX_KEY_BYTESIZE];
    const void* key = reinterpret_cast<const void*>(dummy_key);

    if (tid < input_count) {
        lane_active = true;
        key = static_cast<const uint8_t*>(keys) + tid * hash_ctx.dsize_key_;
        iterator_ptr = iterator_ptrs[tid];
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    // REVIEW: maybe mention in comments the reason why `hash_ctx.Insert` has to
    // be outside of `if (tid < input_count). Is it for warp functions to work?
    bool mask =
            hash_ctx.Insert(lane_active, lane_id, bucket_id, key, iterator_ptr);

    if (tid < input_count) {
        masks[tid] = mask;

        // if (input_count < 100) {
        //     int64_t key0 = *reinterpret_cast<const int64_t*>(key);
        //     int64_t key1 = *(reinterpret_cast<const int64_t*>(key) + 1);
        //     int64_t key2 = *(reinterpret_cast<const int64_t*>(key) + 2);
        //     printf("pass1 %d->%d: %ld %ld %ld, %d\n", tid, iterator_ptr,
        //     key0,
        //            key1, key2, mask);
        // }
    }
}

// REVIEW: rename parameters to be consistent with the caller, iterators ->
// output_iterators; masks -> output_masks.
template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass2(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* values,
                                  ptr_t* iterator_ptrs,
                                  iterator_t* iterators,
                                  bool* masks,
                                  size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < input_count) {
        ptr_t iterator_ptr = iterator_ptrs[tid];

        if (masks[tid]) {
            iterator_t iterator =
                    hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);
            // Success: copy remaining values
            auto src_value_ptr = static_cast<const int*>(values) +
                                 tid * hash_ctx.dsize_value_ / sizeof(int);
            auto dst_value_ptr = static_cast<int*>(iterator.second);
            for (int i = 0; i < hash_ctx.dsize_value_ / sizeof(int); ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }

            if (iterators != nullptr) {
                iterators[tid] = iterator;
            }
        } else {
            hash_ctx.mem_mgr_ctx_.Free(iterator_ptr);

            if (iterators != nullptr) {
                iterators[tid] = iterator_t();
            }
        }
    }
}

// REVIEW: rename parameters to be consistent with the caller, iterators ->
// output_iterators; masks -> output_masks.
template <typename Hash, typename KeyEq>
__global__ void ActivateKernelPass2(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        ptr_t* iterator_ptrs,
        iterator_t* iterators,
        bool* masks,
        size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < input_count) {
        ptr_t iterator_ptr = iterator_ptrs[tid];

        if (masks[tid]) {
            iterator_t iterator =
                    hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);
            if (iterators != nullptr) {
                iterators[tid] = iterator;
            }

            // void* key = iterator.first;
            // int64_t key0 = *reinterpret_cast<const int64_t*>(key);
            // int64_t key1 = *(reinterpret_cast<const int64_t*>(key) + 1);
            // int64_t key2 = *(reinterpret_cast<const int64_t*>(key) + 2);
            // printf("pass2 %d->%d: %ld, %ld, %ld, %d, %p\n", tid,
            // iterator_ptr,
            //        key0, key1, key2, masks[tid], iterator.second);

        } else {
            // iterator_t iterator =
            //         hash_ctx.mem_mgr_ctx_.extract_iterator(iterator_ptr);
            // void* key = iterator.first;
            // int64_t key0 = *reinterpret_cast<const int64_t*>(key);
            // int64_t key1 = *(reinterpret_cast<const int64_t*>(key) + 1);
            // int64_t key2 = *(reinterpret_cast<const int64_t*>(key) + 2);

            // printf("pass2 else %d->%d: %ld, %ld, %ld, %d, %p\n", tid,
            //        iterator_ptr, key0, key1, key2, masks[tid],
            //        iterator.second);

            hash_ctx.mem_mgr_ctx_.Free(iterator_ptr);

            if (iterators != nullptr) {
                iterators[tid] = iterator_t();
            }
        }
    }
}

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 const void* keys,
                                 ptr_t* iterator_ptrs,
                                 bool* masks,
                                 size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // REVIEW: if (tid - lane_id >= input_count)
    if ((tid - lane_id) >= input_count) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    bool lane_active = false;
    uint32_t bucket_id = 0;

    uint8_t dummy_key[MAX_KEY_BYTESIZE];
    const void* key = reinterpret_cast<const void*>(dummy_key);

    if (tid < input_count) {
        lane_active = true;
        key = static_cast<const uint8_t*>(keys) + tid * hash_ctx.dsize_key_;
        bucket_id = hash_ctx.ComputeBucket(key);
    }

    auto result = hash_ctx.Erase(lane_active, lane_id, bucket_id, key);

    if (tid < input_count) {
        iterator_ptrs[tid] = result.first;
        masks[tid] = result.second;
    }
}

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 ptr_t* iterator_ptrs,
                                 bool* masks,
                                 size_t input_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < input_count && masks[tid]) {
        hash_ctx.mem_mgr_ctx_.Free(iterator_ptrs[tid]);
    }
}

template <typename Hash, typename KeyEq>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                   iterator_t* iterators,
                                   uint32_t* iterator_count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    // REVIEW: bucket_id seems more clear.
    uint32_t wid = tid >> 5;
    if (wid >= hash_ctx.bucket_count_) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint32_t src_unit_data =
            *hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    bool is_active = src_unit_data != EMPTY_PAIR_PTR;

    if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
        iterator_t iterator =
                hash_ctx.mem_mgr_ctx_.extract_iterator(src_unit_data);
        uint32_t index = atomicAdd(iterator_count, 1);
        iterators[index] = iterator;
    }

    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data = *hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        is_active = (src_unit_data != EMPTY_PAIR_PTR);

        if (is_active && ((1 << lane_id) & PAIR_PTR_LANES_MASK)) {
            iterator_t iterator =
                    hash_ctx.mem_mgr_ctx_.extract_iterator(src_unit_data);
            uint32_t index = atomicAdd(iterator_count, 1);
            iterators[index] = iterator;
        }
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }
}

/*
 * This kernel can be used to compute total number of elements within each
 * bucket. The final results per bucket is stored in d_count_result array
 */
template <typename Hash, typename KeyEq>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        size_t* bucket_elem_counts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = threadIdx.x & 0x1F;

    // assigning a warp per bucket
    // REVIEW: bucket_id seems more clear.
    uint32_t wid = tid >> 5;
    if (wid >= hash_ctx.bucket_count_) {
        return;
    }

    hash_ctx.node_mgr_ctx_.Init(tid, lane_id);

    uint32_t count = 0;

    // count head node
    uint32_t src_unit_data =
            *hash_ctx.get_unit_ptr_from_list_head(wid, lane_id);
    count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                  src_unit_data != EMPTY_PAIR_PTR));
    ptr_t next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data,
                             NEXT_SLAB_PTR_LANE, WARP_WIDTH);

    // count following nodes
    while (next != EMPTY_SLAB_PTR) {
        src_unit_data = *hash_ctx.get_unit_ptr_from_list_nodes(next, lane_id);
        count += __popc(__ballot_sync(PAIR_PTR_LANES_MASK,
                                      src_unit_data != EMPTY_PAIR_PTR));
        next = __shfl_sync(ACTIVE_LANES_MASK, src_unit_data, NEXT_SLAB_PTR_LANE,
                           WARP_WIDTH);
    }

    // write back the results:
    if (lane_id == 0) {
        bucket_elem_counts[wid] = count;
    }
}

__global__ void UnpackIteratorsKernel(const iterator_t* input_iterators,
                                      const bool* input_masks,
                                      void* output_keys,
                                      void* output_values,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      size_t iterator_count) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Valid queries
    if (tid < iterator_count && (input_masks == nullptr || input_masks[tid])) {
        if (output_keys != nullptr) {
            uint8_t* dst_key_ptr = (uint8_t*)output_keys + dsize_key * tid;
            uint8_t* src_key_ptr =
                    static_cast<uint8_t*>(input_iterators[tid].first);

            for (size_t i = 0; i < dsize_key; ++i) {
                dst_key_ptr[i] = src_key_ptr[i];
            }
        }

        if (output_values != nullptr) {
            uint8_t* dst_value_ptr =
                    (uint8_t*)output_values + dsize_value * tid;
            uint8_t* src_value_ptr =
                    static_cast<uint8_t*>(input_iterators[tid].second);

            for (size_t i = 0; i < dsize_value; ++i) {
                dst_value_ptr[i] = src_value_ptr[i];
            }
        }
    }
}

__global__ void AssignIteratorsKernel(iterator_t* input_iterators,
                                      const bool* input_masks,
                                      const void* input_values,
                                      size_t dsize_value,
                                      size_t iterator_count) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Valid queries
    if (tid < iterator_count && (input_masks == nullptr || input_masks[tid])) {
        uint8_t* src_value_ptr = (uint8_t*)input_values + dsize_value * tid;
        uint8_t* dst_value_ptr =
                static_cast<uint8_t*>(input_iterators[tid].second);

        // Byte-by-byte copy, can be improved
        for (size_t i = 0; i < dsize_value; ++i) {
            dst_value_ptr[i] = src_value_ptr[i];
        }
    }
}
}  // namespace core
}  // namespace open3d
