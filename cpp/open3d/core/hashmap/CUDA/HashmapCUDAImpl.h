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

#include "open3d/core/hashmap/CUDA/InternalNodeManager.h"
#include "open3d/core/hashmap/CUDA/KvPairsCUDA.cuh"
#include "open3d/core/hashmap/CUDA/Macros.h"
#include "open3d/core/hashmap/DeviceHashmap.h"

namespace open3d {
namespace core {

template <typename Hash, typename KeyEq>
class CUDAHashmapImplContext {
public:
    CUDAHashmapImplContext();

    __host__ void Setup(int64_t init_buckets,
                        int64_t init_capacity,
                        int64_t dsize_key,
                        int64_t dsize_value,
                        const InternalNodeManagerContext& node_mgr_ctx,
                        const CUDAKvPairsContext& kv_mgr_ctx);

    __device__ bool Insert(bool lane_active,
                           uint32_t lane_id,
                           uint32_t bucket_id,
                           const void* key_ptr,
                           addr_t iterator_addr);

    __device__ Pair<addr_t, bool> Find(bool lane_active,
                                       uint32_t lane_id,
                                       uint32_t bucket_id,
                                       const void* key_ptr);

    __device__ Pair<addr_t, bool> Erase(bool lane_active,
                                        uint32_t lane_id,
                                        uint32_t bucket_id,
                                        const void* key_ptr);

    __device__ void WarpSyncKey(const void* key_ptr,
                                uint32_t lane_id,
                                void* ret_key_ptr);
    __device__ int32_t WarpFindKey(const void* src_key_ptr,
                                   uint32_t lane_id,
                                   addr_t ptr);
    __device__ int32_t WarpFindEmpty(addr_t unit_data);

    // Hash function.
    __device__ int64_t ComputeBucket(const void* key_ptr) const;

    // Node manager.
    __device__ addr_t AllocateSlab(uint32_t lane_id);
    __device__ void FreeSlab(addr_t slab_ptr);

    // Helpers.
    __device__ addr_t* get_unit_ptr_from_list_nodes(addr_t slab_ptr,
                                                    uint32_t lane_id) {
        return node_mgr_ctx_.get_unit_ptr_from_slab(slab_ptr, lane_id);
    }
    __device__ addr_t* get_unit_ptr_from_list_head(uint32_t bucket_id,
                                                   uint32_t lane_id) {
        return reinterpret_cast<uint32_t*>(bucket_list_head_) +
               bucket_id * kWarpSize + lane_id;
    }

public:
    Hash hash_fn_;
    KeyEq cmp_fn_;

    int64_t bucket_count_;
    int64_t capacity_;
    int64_t dsize_key_;
    int64_t dsize_value_;

    Slab* bucket_list_head_;
    InternalNodeManagerContext node_mgr_ctx_;
    CUDAKvPairsContext kv_mgr_ctx_;
};

/// Kernels
template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* input_keys,
                                  addr_t* output_iterator_addrs,
                                  int heap_counter_prev,
                                  int64_t count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* input_keys,
                                  addr_t* input_iterator_addrs,
                                  bool* output_masks,
                                  int64_t count);

template <typename Hash, typename KeyEq>
__global__ void InsertKernelPass2(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                  const void* input_values,
                                  addr_t* input_iterator_addrs,
                                  iterator_t* output_iterators,
                                  bool* output_masks,
                                  int64_t count);

template <typename Hash, typename KeyEq>
__global__ void FindKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                           const void* input_keys,
                           iterator_t* output_iterators,
                           bool* output_masks,
                           int64_t count);

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass0(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 const void* input_keys,
                                 addr_t* output_iterator_addrs,
                                 bool* output_masks,
                                 int64_t count);

template <typename Hash, typename KeyEq>
__global__ void EraseKernelPass1(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                 addr_t* input_iterator_addrs,
                                 bool* output_masks,
                                 int64_t count);

template <typename Hash, typename KeyEq>
__global__ void GetIteratorsKernel(CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
                                   iterator_t* output_iterators,
                                   uint32_t* output_iterator_count);

template <typename Hash, typename KeyEq>
__global__ void CountElemsPerBucketKernel(
        CUDAHashmapImplContext<Hash, KeyEq> hash_ctx,
        int64_t* bucket_elem_counts);

__global__ void UnpackIteratorsKernel(const iterator_t* input_iterators,
                                      const bool* input_masks,
                                      void* output_keys,
                                      void* output_values,
                                      int64_t dsize_key,
                                      int64_t dsize_value,
                                      int64_t iterator_count);

__global__ void AssignIteratorsKernel(iterator_t* input_iterators,
                                      const bool* input_masks,
                                      const void* input_values,
                                      int64_t dsize_value,
                                      int64_t iterator_count);

}  // namespace core
}  // namespace open3d
