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

// Copyright 2019 Saman Ashkiani
// Rewritten by Wei Dong 2019 - 2020
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

#include <thrust/device_vector.h>

#include <cassert>
#include <memory>
#include <random>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/Macros.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
/// Internal Hashtable Node: (31 units and 1 next ptr) representation.
/// \member kv_pair_ptrs:
/// Each element is an internal ptr to a kv pair managed by the
/// InternalMemoryManager. Can be converted to a real ptr.
/// \member next_slab_ptr:
/// An internal ptr managed by InternalNodeManager.
class Slab {
public:
    ptr_t kv_pair_ptrs[WARP_WIDTH - 1];
    ptr_t next_slab_ptr;
};

// REVIEW: Update these to be consistent with Macros.h?
/// 32 super blocks (5 bit)
/// 256 memory blocks (8 bit) per super block
/// 1024 slabs (10 bit) per memory block
/// 32 pair ptrs (5 bit) per slab

/// Each warp is assigned to a memory block and simultaneously look for an empty
/// slab in 1024 candidates.
class InternalNodeManagerContext {
public:
    InternalNodeManagerContext()
        : super_blocks_(nullptr),
          hash_coef_(0),
          num_attempts_(0),
          memory_block_index_(0),
          super_block_index_(0) {}

    // REVIEW: this is not used, consider removing?
    InternalNodeManagerContext& operator=(
            const InternalNodeManagerContext& rhs) {
        super_blocks_ = rhs.super_blocks_;
        hash_coef_ = rhs.hash_coef_;
        super_block_index_ = 0;
        memory_block_index_ = 0;
        num_attempts_ = 0;
        return *this;
    }

    // REVIEW: Can the constructor only take (uint32_t* super_blocks, uint32_t
    // hash_coef) and merge Setup to the constructor?
    void Setup(uint32_t* super_blocks, uint32_t hash_coef) {
        super_blocks_ = super_blocks;
        hash_coef_ = hash_coef;
    }

    __device__ __forceinline__ uint32_t* get_unit_ptr_from_slab(
            const ptr_t& next_slab_ptr, const uint32_t& lane_id) {
        return super_blocks_ + addressDecoder(next_slab_ptr) + lane_id;
    }
    __device__ __forceinline__ uint32_t* get_ptr_for_bitmap(
            const uint32_t super_block_idx, const uint32_t bitmap_idx) {
        return super_blocks_ + super_block_idx * SUPER_BLOCK_SIZE_ + bitmap_idx;
    }

    // Objective: each warp selects its own memory_block warp allocator:
    __device__ void Init(uint32_t& tid, uint32_t& lane_id) {
        // hashing the memory block to be used:
        createMemBlockIndex(tid >> 5);

        // loading the assigned memory block:
        memory_block_bitmap_ =
                super_blocks_[super_block_index_ * SUPER_BLOCK_SIZE_ +
                              memory_block_index_ * BITMAP_SIZE_ + lane_id];
    }

    __device__ uint32_t WarpAllocate(const uint32_t& lane_id) {
        // tries and allocate a new memory units within the memory_block memory
        // block if it returns 0xFFFFFFFF, then there was not any empty memory
        // unit a new memory_block block should be chosen, and repeat again
        // allocated result:  5  bits: super_block_index
        //                    17 bits: memory block index
        //                    5  bits: memory unit index (hi-bits of 10bit)
        //                    5  bits: memory unit index (lo-bits of 10bit)
        int empty_lane = -1;
        uint32_t free_lane;
        uint32_t read_bitmap = memory_block_bitmap_;
        // REVIEW: replace these 0xFFFFFFFF with values from Macros.h?
        uint32_t allocated_result = 0xFFFFFFFF;
        // works as long as <31 bit are used in the allocated_result
        // in other words, if there are 32 super blocks and at most 64k blocks
        // per super block

        while (allocated_result == 0xFFFFFFFF) {
            empty_lane = __ffs(~memory_block_bitmap_) - 1;
            free_lane = __ballot_sync(0xFFFFFFFF, empty_lane >= 0);
            if (free_lane == 0) {
                // all bitmaps are full: need to be rehashed again:
                updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >>
                                    5);
                read_bitmap = memory_block_bitmap_;
                continue;
            }
            uint32_t src_lane = __ffs(free_lane) - 1;
            if (src_lane == lane_id) {
                read_bitmap = atomicCAS(
                        super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_ +
                                memory_block_index_ * BITMAP_SIZE_ + lane_id,
                        memory_block_bitmap_,
                        memory_block_bitmap_ | (1 << empty_lane));
                if (read_bitmap == memory_block_bitmap_) {
                    // successful attempt:
                    memory_block_bitmap_ |= (1 << empty_lane);
                    allocated_result = (super_block_index_
                                        << SUPER_BLOCK_BIT_OFFSET_ALLOC_) |
                                       (memory_block_index_
                                        << MEM_BLOCK_BIT_OFFSET_ALLOC_) |
                                       (lane_id << MEM_UNIT_BIT_OFFSET_ALLOC_) |
                                       empty_lane;
                } else {
                    // Not successful: updating the current bitmap
                    memory_block_bitmap_ = read_bitmap;
                }
            }
            // asking for the allocated result;
            allocated_result =
                    __shfl_sync(0xFFFFFFFF, allocated_result, src_lane);
        }
        return allocated_result;
    }

    // This function, frees a recently allocated memory unit by a single thread.
    // Since it is untouched, there shouldn't be any worries for the actual
    // memory contents to be reset again.
    __device__ void FreeUntouched(ptr_t ptr) {
        atomicAnd(super_blocks_ + getSuperBlockIndex(ptr) * SUPER_BLOCK_SIZE_ +
                          getMemBlockIndex(ptr) * BITMAP_SIZE_ +
                          (getMemUnitIndex(ptr) >> 5),
                  ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
    }

private:
    // =========
    // some helper inline address functions:
    // =========
    __device__ __host__ __forceinline__ uint32_t
    getSuperBlockIndex(ptr_t address) const {
        return address >> SUPER_BLOCK_BIT_OFFSET_ALLOC_;
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemBlockIndex(ptr_t address) const {
        return ((address >> MEM_BLOCK_BIT_OFFSET_ALLOC_) & 0x1FFFF);
    }
    __device__ __host__ __forceinline__ ptr_t
    getMemBlockAddress(ptr_t address) const {
        return (MEM_BLOCK_OFFSET_ +
                getMemBlockIndex(address) * MEM_BLOCK_SIZE_);
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemUnitIndex(ptr_t address) const {
        return address & 0x3FF;
    }
    __device__ __host__ __forceinline__ ptr_t getMemUnitAddress(ptr_t address) {
        return getMemUnitIndex(address) * MEM_UNIT_SIZE_;
    }

    // called at the beginning of the kernel:
    __device__ void createMemBlockIndex(uint32_t global_warp_id) {
        super_block_index_ = global_warp_id % NUM_SUPER_BLOCKS_;
        memory_block_index_ =
                (hash_coef_ * global_warp_id) >> (32 - LOG_NUM_MEM_BLOCKS_);
    }

    // called when the allocator fails to find an empty unit to allocate:
    __device__ void updateMemBlockIndex(uint32_t global_warp_id) {
        num_attempts_++;
        super_block_index_++;
        super_block_index_ = (super_block_index_ == NUM_SUPER_BLOCKS_)
                                     ? 0
                                     : super_block_index_;
        memory_block_index_ = (hash_coef_ * (global_warp_id + num_attempts_)) >>
                              (32 - LOG_NUM_MEM_BLOCKS_);
        // loading the assigned memory block:
        memory_block_bitmap_ =
                *((super_blocks_ + super_block_index_ * SUPER_BLOCK_SIZE_) +
                  memory_block_index_ * BITMAP_SIZE_ + (threadIdx.x & 0x1f));
    }

    __host__ __device__ ptr_t addressDecoder(ptr_t address_ptr_index) {
        return getSuperBlockIndex(address_ptr_index) * SUPER_BLOCK_SIZE_ +
               getMemBlockAddress(address_ptr_index) +
               getMemUnitIndex(address_ptr_index) * WARP_SIZE;
    }

    __host__ __device__ void print_address(ptr_t address_ptr_index) {
        printf("Super block Index: %d, Memory block index: %d, Memory unit "
               "index: "
               "%d\n",
               getSuperBlockIndex(address_ptr_index),
               getMemBlockIndex(address_ptr_index),
               getMemUnitIndex(address_ptr_index));
    }

private:
    // a pointer to each super-block
    uint32_t* super_blocks_;

    // hash_coef (register): used as (16 bits, 16 bits) for hashing
    uint32_t hash_coef_;  // a random 32-bit

    // memory_block (16 bits       + 5 bits) (memory block  + super block)
    uint32_t num_attempts_;
    uint32_t memory_block_index_;
    uint32_t memory_block_bitmap_;
    uint32_t super_block_index_;
};

__global__ void CountSlabsPerSuperblockKernel(
        InternalNodeManagerContext context, uint32_t* slabs_per_superblock);

/*
 * This class owns the memory for the allocator on the device
 */
class InternalNodeManager {
private:
    uint32_t* super_blocks_;

    // hash a warp id to a memory block index
    uint32_t hash_coef_;  // a random 32-bit

public:
    InternalNodeManagerContext gpu_context_;
    Device device_;

public:
    // REVIEW: the initialization list seems not useful, since the values are
    // overwritten in function body, except for device_.
    InternalNodeManager(const Device& device)
        : super_blocks_(nullptr), hash_coef_(0), device_(device) {
        // random coefficients for allocator's hash function
        std::mt19937 rng(time(0));
        hash_coef_ = rng();

        // In the light version, we put num_super_blocks super blocks within
        // a single array
        super_blocks_ = static_cast<uint32_t*>(MemoryManager::Malloc(
                SUPER_BLOCK_SIZE_ * NUM_SUPER_BLOCKS_ * sizeof(uint32_t),
                device_));

        OPEN3D_CUDA_CHECK(cudaMemset(
                super_blocks_, 0xFF,
                SUPER_BLOCK_SIZE_ * NUM_SUPER_BLOCKS_ * sizeof(uint32_t)));
        // printf("TOTAL ITERATORS: %ld\n", SUPER_BLOCK_SIZE_ *
        // NUM_SUPER_BLOCKS_);

        for (uint32_t i = 0; i < NUM_SUPER_BLOCKS_; i++) {
            // setting bitmaps into zeros:
            OPEN3D_CUDA_CHECK(
                    cudaMemset(super_blocks_ + i * SUPER_BLOCK_SIZE_, 0x00,
                               NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * BITMAP_SIZE_ *
                                       sizeof(uint32_t)));
        }

        // initializing the slab context:
        gpu_context_.Setup(super_blocks_, hash_coef_);
    }

    ~InternalNodeManager() { MemoryManager::Free(super_blocks_, device_); }

    std::vector<int> CountSlabsPerSuperblock() {
        const uint32_t num_super_blocks = NUM_SUPER_BLOCKS_;

        auto slabs_per_superblock_buffer =
                static_cast<uint32_t*>(MemoryManager::Malloc(
                        NUM_SUPER_BLOCKS_ * sizeof(uint32_t), device_));
        // REVIEW: Is this a copy? If yes, we can let thrust manage the memory
        // allocation directly.
        // e.g. thrust::device_vector<uint32_t> vec(num_super_blocks, 0);
        thrust::device_vector<uint32_t> slabs_per_superblock(
                slabs_per_superblock_buffer,
                slabs_per_superblock_buffer + num_super_blocks);
        thrust::fill(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     0);

        // counting total number of allocated memory units:
        // REVIEW: replace 128 and 32 with values from Macros.h?
        int blocksize = 128;
        int num_mem_units = NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
        int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
        CountSlabsPerSuperblockKernel<<<num_cuda_blocks, blocksize>>>(
                gpu_context_,
                thrust::raw_pointer_cast(slabs_per_superblock.data()));
        // REVIEW: do we need these after kernel call?
        // OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
        // OPEN3D_CUDA_CHECK(cudaGetLastError());
        std::vector<int> result(num_super_blocks);
        thrust::copy(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     result.begin());
        MemoryManager::Free(slabs_per_superblock_buffer, device_);

        return std::move(result);
    }
};

__global__ void CountSlabsPerSuperblockKernel(
        InternalNodeManagerContext context, uint32_t* slabs_per_superblock) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int num_bitmaps = NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    if (tid >= num_bitmaps) {
        return;
    }

    for (uint32_t i = 0; i < NUM_SUPER_BLOCKS_; i++) {
        uint32_t read_bitmap = *(context.get_ptr_for_bitmap(i, tid));
        atomicAdd(&slabs_per_superblock[i], __popc(read_bitmap));
    }
}
}  // namespace core
}  // namespace open3d
