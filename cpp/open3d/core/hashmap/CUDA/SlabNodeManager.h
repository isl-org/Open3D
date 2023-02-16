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

#include <memory>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/SlabMacros.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace core {

/// Internal Hashtable Node: (31 units and 1 next ptr) representation.
/// A slab is kWarpSize x kWarpSize bits, or kWarpSize 32-bit uints.
class Slab {
public:
    /// Each element is an internal ptr to a kv pair managed by the
    /// InternalMemoryManager. Can be converted to a real ptr.
    buf_index_t kv_pair_ptrs[kWarpSize - 1];
    /// An internal ptr managed by InternalNodeManager.
    buf_index_t next_slab_ptr;
};

class SlabNodeManagerImpl {
public:
    SlabNodeManagerImpl()
        : super_blocks_(nullptr),
          hash_coef_(0),
          num_attempts_(0),
          memory_block_index_(0),
          super_block_index_(0) {}

    __device__ __forceinline__ uint32_t* get_unit_ptr_from_slab(
            const buf_index_t& next_slab_ptr, const uint32_t& lane_id) {
        return super_blocks_ + addressDecoder(next_slab_ptr) + lane_id;
    }
    __device__ __forceinline__ uint32_t* get_ptr_for_bitmap(
            const uint32_t super_block_idx, const uint32_t bitmap_idx) {
        return super_blocks_ + super_block_idx * kUIntsPerSuperBlock +
               bitmap_idx;
    }

    // Objective: each warp selects its own memory_block warp allocator.
    __device__ void Init(uint32_t& tid, uint32_t& lane_id) {
        // Hashing the memory block to be used.
        createMemBlockIndex(tid >> 5);

        // Loading the assigned memory block.
        memory_block_bitmap_ =
                super_blocks_[super_block_index_ * kUIntsPerSuperBlock +
                              memory_block_index_ * kSlabsPerBlock + lane_id];
    }

    __device__ uint32_t WarpAllocate(const uint32_t& lane_id) {
        // Try and allocate a new memory units within the memory_block memory
        // block if it returns 0xFFFFFFFF, then there was not any empty memory
        // unit a new memory_block block should be chosen, and repeat again
        // allocated result:  5  bits: super_block_index
        //                    17 bits: memory block index
        //                    5  bits: memory unit index (hi-bits of 10bit)
        //                    5  bits: memory unit index (lo-bits of 10bit)
        int empty_lane = -1;
        uint32_t free_lane;
        uint32_t read_bitmap = memory_block_bitmap_;
        uint32_t allocated_result = kNotFoundFlag;
        // Works as long as <31 bit are used in the allocated_result
        // in other words, if there are 32 super blocks and at most 64k blocks
        // per super block.

        while (allocated_result == kNotFoundFlag) {
            empty_lane = __ffs(~memory_block_bitmap_) - 1;
            free_lane = __ballot_sync(kSyncLanesMask, empty_lane >= 0);
            if (free_lane == 0) {
                // all bitmaps are full: need to be rehashed again.
                updateMemBlockIndex((threadIdx.x + blockIdx.x * blockDim.x) >>
                                    5);
                read_bitmap = memory_block_bitmap_;
                continue;
            }
            uint32_t src_lane = __ffs(free_lane) - 1;
            if (src_lane == lane_id) {
                read_bitmap = atomicCAS(
                        super_blocks_ +
                                super_block_index_ * kUIntsPerSuperBlock +
                                memory_block_index_ * kSlabsPerBlock + lane_id,
                        memory_block_bitmap_,
                        memory_block_bitmap_ | (1 << empty_lane));
                if (read_bitmap == memory_block_bitmap_) {
                    // Successful attempt.
                    memory_block_bitmap_ |= (1 << empty_lane);
                    allocated_result =
                            (super_block_index_ << kSuperBlockMaskBits) |
                            (memory_block_index_ << kBlockMaskBits) |
                            (lane_id << kSlabMaskBits) | empty_lane;
                } else {
                    // Not successful: updating the current bitmap.
                    memory_block_bitmap_ = read_bitmap;
                }
            }
            // Asking for the allocated result.
            allocated_result =
                    __shfl_sync(kSyncLanesMask, allocated_result, src_lane);
        }
        return allocated_result;
    }

    // This function, frees a recently allocated memory unit by a single thread.
    // Since it is untouched, there shouldn't be any worries for the actual
    // memory contents to be reset again.
    __device__ void FreeUntouched(buf_index_t ptr) {
        atomicAnd(super_blocks_ +
                          getSuperBlockIndex(ptr) * kUIntsPerSuperBlock +
                          getMemBlockIndex(ptr) * kSlabsPerBlock +
                          (getMemUnitIndex(ptr) >> 5),
                  ~(1 << (getMemUnitIndex(ptr) & 0x1F)));
    }

private:
    __device__ __host__ __forceinline__ uint32_t
    getSuperBlockIndex(buf_index_t address) const {
        return address >> kSuperBlockMaskBits;
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemBlockIndex(buf_index_t address) const {
        return ((address >> kBlockMaskBits) & 0x1FFFF);
    }
    __device__ __host__ __forceinline__ buf_index_t
    getMemBlockAddress(buf_index_t address) const {
        return (kBitmapsPerSuperBlock +
                getMemBlockIndex(address) * kUIntsPerBlock);
    }
    __device__ __host__ __forceinline__ uint32_t
    getMemUnitIndex(buf_index_t address) const {
        return address & 0x3FF;
    }
    __device__ __host__ __forceinline__ buf_index_t
    getMemUnitAddress(buf_index_t address) {
        return getMemUnitIndex(address) * kWarpSize;
    }

    // Called at the beginning of the kernel.
    __device__ void createMemBlockIndex(uint32_t global_warp_id) {
        super_block_index_ = global_warp_id % kSuperBlocks;
        memory_block_index_ = (hash_coef_ * global_warp_id) >>
                              (32 - kBlocksPerSuperBlockInBits);
    }

    // Called when the allocator fails to find an empty unit to allocate.
    __device__ void updateMemBlockIndex(uint32_t global_warp_id) {
        num_attempts_++;
        super_block_index_++;
        super_block_index_ =
                (super_block_index_ == kSuperBlocks) ? 0 : super_block_index_;
        memory_block_index_ = (hash_coef_ * (global_warp_id + num_attempts_)) >>
                              (32 - kBlocksPerSuperBlockInBits);
        // Loading the assigned memory block.
        memory_block_bitmap_ =
                *((super_blocks_ + super_block_index_ * kUIntsPerSuperBlock) +
                  memory_block_index_ * kSlabsPerBlock + (threadIdx.x & 0x1f));
    }

    __host__ __device__ buf_index_t
    addressDecoder(buf_index_t address_ptr_index) {
        return getSuperBlockIndex(address_ptr_index) * kUIntsPerSuperBlock +
               getMemBlockAddress(address_ptr_index) +
               getMemUnitIndex(address_ptr_index) * kWarpSize;
    }

    __host__ __device__ void print_address(buf_index_t address_ptr_index) {
        printf("Super block Index: %d, Memory block index: %d, Memory unit "
               "index: "
               "%d\n",
               getSuperBlockIndex(address_ptr_index),
               getMemBlockIndex(address_ptr_index),
               getMemUnitIndex(address_ptr_index));
    }

public:
    /// A pointer to each super-block.
    uint32_t* super_blocks_;
    /// hash_coef (register): used as (16 bits, 16 bits) for hashing.
    uint32_t hash_coef_;  // A random 32-bit.

private:
    /// memory_block (16 bits + 5 bits) (memory block + super block).
    uint32_t num_attempts_;
    uint32_t memory_block_index_;
    uint32_t memory_block_bitmap_;
    uint32_t super_block_index_;
};

__global__ void CountSlabsPerSuperblockKernel(SlabNodeManagerImpl impl,
                                              uint32_t* slabs_per_superblock);

class SlabNodeManager {
public:
    SlabNodeManager(const Device& device) : device_(device) {
        /// Random coefficients for allocator's hash function.
        impl_.hash_coef_ = utility::random::RandUint32();

        /// In the light version, we put num_super_blocks super blocks within
        /// a single array.
        impl_.super_blocks_ = static_cast<uint32_t*>(MemoryManager::Malloc(
                kUIntsPerSuperBlock * kSuperBlocks * sizeof(uint32_t),
                device_));
        Reset();
    }

    ~SlabNodeManager() { MemoryManager::Free(impl_.super_blocks_, device_); }

    void Reset() {
        OPEN3D_CUDA_CHECK(cudaMemset(
                impl_.super_blocks_, 0xFF,
                kUIntsPerSuperBlock * kSuperBlocks * sizeof(uint32_t)));

        for (uint32_t i = 0; i < kSuperBlocks; i++) {
            // setting bitmaps into zeros:
            OPEN3D_CUDA_CHECK(cudaMemset(
                    impl_.super_blocks_ + i * kUIntsPerSuperBlock, 0x00,
                    kBlocksPerSuperBlock * kSlabsPerBlock * sizeof(uint32_t)));
        }
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());
    }

    std::vector<int> CountSlabsPerSuperblock() {
        const uint32_t num_super_blocks = kSuperBlocks;

        thrust::device_vector<uint32_t> slabs_per_superblock(kSuperBlocks);
        thrust::fill(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     0);

        // Counting total number of allocated memory units.
        int num_mem_units = kBlocksPerSuperBlock * 32;
        int num_cuda_blocks =
                (num_mem_units + kThreadsPerBlock - 1) / kThreadsPerBlock;
        CountSlabsPerSuperblockKernel<<<num_cuda_blocks, kThreadsPerBlock, 0,
                                        core::cuda::GetStream()>>>(
                impl_, thrust::raw_pointer_cast(slabs_per_superblock.data()));
        cuda::Synchronize();
        OPEN3D_CUDA_CHECK(cudaGetLastError());

        std::vector<int> result(num_super_blocks);
        thrust::copy(slabs_per_superblock.begin(), slabs_per_superblock.end(),
                     result.begin());

        return result;
    }

public:
    SlabNodeManagerImpl impl_;
    Device device_;
};
}  // namespace core
}  // namespace open3d
