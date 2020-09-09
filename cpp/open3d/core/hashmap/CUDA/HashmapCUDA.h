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

#include <cassert>
#include <memory>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/CUDA/HashmapCUDAImpl.h"
#include "open3d/core/hashmap/DeviceHashmap.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CUDAHashmap : public DeviceHashmap<Hash, KeyEq> {
public:
    ~CUDAHashmap();

    CUDAHashmap(size_t init_buckets,
                size_t init_capacity,
                size_t dsize_key,
                size_t dsize_value,
                const Device& device);

    void Rehash(size_t buckets) override;

    void Insert(const void* input_keys,
                const void* input_values,
                iterator_t* output_iterators,
                bool* output_masks,
                size_t count) override;

    void Activate(const void* input_keys,
                  iterator_t* output_iterators,
                  bool* output_masks,
                  size_t count) override;

    void Find(const void* input_keys,
              iterator_t* output_iterators,
              bool* output_masks,
              size_t count) override;

    void Erase(const void* input_keys,
               bool* output_masks,
               size_t count) override;

    size_t GetIterators(iterator_t* output_iterators) override;

    void UnpackIterators(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         size_t count) override;

    void AssignIterators(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         size_t count) override;

    std::vector<size_t> BucketSizes() const override;

    float LoadFactor() const override;

    size_t Size() const override;

protected:
    /// struct directly passed to kernels, cannot be a pointer
    CUDAHashmapImplContext<Hash, KeyEq> gpu_context_;

    std::shared_ptr<InternalKvPairManager> mem_mgr_;
    std::shared_ptr<InternalNodeManager> node_mgr_;

    void Allocate(size_t bucket_count, size_t capacity);

    void InsertImpl(const void* input_keys,
                    const void* input_values,
                    iterator_t* output_iterators,
                    bool* output_masks,
                    size_t count);

    void ActivateImpl(const void* input_keys,
                      iterator_t* output_iterators,
                      bool* output_masks,
                      size_t count);

    void FindImpl(const void* input_keys,
                  iterator_t* output_iterators,
                  bool* output_masks,
                  size_t count);

    void EraseImpl(const void* input_keys, bool* output_masks, size_t count);
};

/// Interface
template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::CUDAHashmap(size_t init_buckets,
                                      size_t init_capacity,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      const Device& device)
    : DeviceHashmap<Hash, KeyEq>(
              init_buckets, init_capacity, dsize_key, dsize_value, device) {
    Allocate(init_buckets, init_capacity);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Allocate(size_t bucket_count, size_t capacity) {
    this->bucket_count_ = bucket_count;
    this->capacity_ = capacity;

    mem_mgr_ = std::make_shared<InternalKvPairManager>(
            this->capacity_, this->dsize_key_, this->dsize_value_,
            this->device_);

    // Memory for hash table linked list nodes
    node_mgr_ = std::make_shared<InternalNodeManager>(this->device_);
    gpu_context_.Setup(this->bucket_count_, this->capacity_, this->dsize_key_,
                       this->dsize_value_, node_mgr_->gpu_context_,
                       mem_mgr_->gpu_context_);

    // Memory for hash table
    gpu_context_.bucket_list_head_ = static_cast<Slab*>(MemoryManager::Malloc(
            sizeof(Slab) * this->bucket_count_, this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(gpu_context_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * this->bucket_count_));

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::~CUDAHashmap() {
    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::Size() const {
    return *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                      const void* input_values,
                                      iterator_t* output_iterators,
                                      bool* output_masks,
                                      size_t count) {
    bool extern_masks = (output_masks != nullptr);
    if (!extern_masks) {
        output_masks = static_cast<bool*>(
                MemoryManager::Malloc(count * sizeof(bool), this->device_));
    }

    // Check capacity and rehash if in need
    int capacity = Size();
    uint32_t new_capacity = capacity + count;

    if (new_capacity > this->capacity_) {
        float avg_ratio = float(this->capacity_) / float(this->bucket_count_);
        uint32_t exp_buckets = uint32_t(std::ceil(new_capacity / avg_ratio));

        // At least increase by a factor of 2
        Rehash(std::max(this->bucket_count_ * 2, exp_buckets));
    }

    InsertImpl(input_keys, input_values, output_iterators, output_masks, count);

    if (!extern_masks) {
        MemoryManager::Free(output_masks, this->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::InsertImpl(const void* input_keys,
                                          const void* input_values,
                                          iterator_t* output_iterators,
                                          bool* output_masks,
                                          size_t count) {
    bool extern_masks = (output_masks != nullptr);
    if (!extern_masks) {
        output_masks = static_cast<bool*>(
                MemoryManager::Malloc(count * sizeof(bool), this->device_));
    }
    auto iterator_ptrs = static_cast<ptr_t*>(
            MemoryManager::Malloc(sizeof(ptr_t) * count, this->device_));
    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;

    int heap_counter =
            *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
    *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_) =
            heap_counter + count;
    InsertKernelPass0<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, iterator_ptrs, heap_counter, count);
    InsertKernelPass1<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, iterator_ptrs, output_masks, count);
    InsertKernelPass2<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_values, iterator_ptrs, output_iterators,
            output_masks, count);
    if (!extern_masks) {
        MemoryManager::Free(output_masks, this->device_);
    }
    MemoryManager::Free(iterator_ptrs, this->device_);

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Activate(const void* input_keys,
                                        iterator_t* output_iterators,
                                        bool* output_masks,
                                        size_t count) {
    static int rehash_count = 0;

    bool extern_masks = (output_masks != nullptr);
    if (!extern_masks) {
        output_masks = static_cast<bool*>(
                MemoryManager::Malloc(count * sizeof(bool), this->device_));
    }

    // Check capacity
    int capacity = Size();
    uint32_t new_capacity = capacity + count;
    utility::LogInfo("Rehash count = {}, capacity = {}, new_capacity = {}",
                     count, this->capacity_, new_capacity);
    if (new_capacity > this->capacity_) {
        float avg_ratio = float(this->capacity_) / float(this->bucket_count_);
        uint32_t exp_buckets = uint32_t(std::ceil(new_capacity / avg_ratio));

        // At least increase by a factor of 2
        Rehash(std::max(this->bucket_count_ * 2, exp_buckets));
        rehash_count++;
    }

    ActivateImpl(input_keys, output_iterators, output_masks, count);

    if (!extern_masks) {
        MemoryManager::Free(output_masks, this->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::ActivateImpl(const void* input_keys,
                                            iterator_t* output_iterators,
                                            bool* output_masks,
                                            size_t count) {
    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    auto iterator_ptrs = static_cast<ptr_t*>(
            MemoryManager::Malloc(sizeof(ptr_t) * count, this->device_));

    int heap_counter =
            *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
    *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_) =
            heap_counter + count;
    InsertKernelPass0<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, iterator_ptrs, heap_counter, count);

    InsertKernelPass1<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, iterator_ptrs, output_masks, count);

    ActivateKernelPass2<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, iterator_ptrs, output_iterators, output_masks, count);

    MemoryManager::Free(iterator_ptrs, this->device_);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                    iterator_t* output_iterators,
                                    bool* output_masks,
                                    size_t count) {
    bool extern_masks = (output_masks != nullptr);
    if (!extern_masks) {
        output_masks = static_cast<bool*>(
                MemoryManager::Malloc(count * sizeof(bool), this->device_));
    }

    FindImpl(input_keys, output_iterators, output_masks, count);

    if (!extern_masks) {
        MemoryManager::Free(output_masks, this->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::FindImpl(const void* input_keys,
                                        iterator_t* output_iterators,
                                        bool* output_masks,
                                        size_t count) {
    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));

    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    FindKernel<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, output_iterators, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    // thrust::device_vector<void*> all_iterators_device(
    //         (void**)output_iterators, (void**)output_iterators + count * 2);
    // for (int i = 0; i < count * 2; ++i) {
    //     void* iterator = all_iterators_device[i];
    //     std::cout << iterator << "\n";
    // }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                     bool* output_masks,
                                     size_t count) {
    bool extern_masks = (output_masks != nullptr);
    if (!extern_masks) {
        output_masks = (bool*)MemoryManager::Malloc(count * sizeof(bool),
                                                    this->device_);
    }

    EraseImpl(input_keys, output_masks, count);

    if (!extern_masks) {
        MemoryManager::Free(output_masks, this->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::EraseImpl(const void* input_keys,
                                         bool* output_masks,
                                         size_t count) {
    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;

    auto iterator_ptrs = static_cast<ptr_t*>(
            MemoryManager::Malloc(sizeof(ptr_t) * count, this->device_));

    EraseKernelPass0<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, (uint8_t*)input_keys, iterator_ptrs, output_masks,
            count);
    EraseKernelPass1<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, iterator_ptrs,
                                                 output_masks, count);

    MemoryManager::Free(iterator_ptrs, this->device_);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.bucket_count_ * WARP_SIZE + blocksize - 1) /
            blocksize;

    uint32_t* iterator_count =
            (uint32_t*)MemoryManager::Malloc(sizeof(uint32_t), this->device_);
    cudaMemset(iterator_count, 0, sizeof(uint32_t));

    GetIteratorsKernel<<<num_blocks, blocksize>>>(
            gpu_context_, output_iterators, iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    uint32_t ret;
    MemoryManager::Memcpy(&ret, Device("CPU:0"), iterator_count, this->device_,
                          sizeof(uint32_t));

    // uint32_t total_count = ret;
    // thrust::device_vector<iterator_t> all_iterators_device(
    //         output_iterators, output_iterators + total_count);

    // // 2. Verbose check: every iterator should be observable in gt// for
    // (size_t i = 0; i < total_count; ++i) {
    //     iterator_t iterator = all_iterators_device[i];
    //     int64_t key0 = *(thrust::device_ptr<int64_t>(
    //             reinterpret_cast<int64_t*>(iterator.first)));
    //     int64_t key1 = *(thrust::device_ptr<int64_t>(
    //             reinterpret_cast<int64_t*>(iterator.first) + 1));
    //     int64_t key2 = *(thrust::device_ptr<int64_t>(
    //             reinterpret_cast<int64_t*>(iterator.first) + 2));
    //     std::cout << key0 << ", " << key1 << ", " << key2 << "->"
    //               << iterator.second << "\n";
    // }

    return ret;
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::UnpackIterators(
        const iterator_t* input_iterators,
        const bool* input_masks,
        void* output_keys,
        void* output_values,
        size_t iterator_count) {
    if (iterator_count == 0) return;

    const size_t num_threads = 32;
    const size_t num_blocks = (iterator_count + num_threads - 1) / num_threads;

    UnpackIteratorsKernel<<<num_blocks, num_threads>>>(
            input_iterators, input_masks, output_keys, output_values,
            this->dsize_key_, this->dsize_value_, iterator_count);

    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::AssignIterators(iterator_t* input_iterators,
                                               const bool* input_masks,
                                               const void* input_values,
                                               size_t iterator_count) {
    if (iterator_count == 0) return;
    const size_t num_threads = 32;
    const size_t num_blocks = (iterator_count + num_threads - 1) / num_threads;

    AssignIteratorsKernel<<<num_blocks, num_threads>>>(
            input_iterators, input_masks, input_values, this->dsize_value_,
            iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Rehash(size_t buckets) {
    utility::LogInfo("Rehashing: {} -> {}", this->bucket_count_, buckets);
    // TODO: add a size operator instead of rough estimation
    auto output_iterators = (iterator_t*)MemoryManager::Malloc(
            sizeof(iterator_t) * this->capacity_, this->device_);
    uint32_t iterator_count = GetIterators(output_iterators);

    void* output_keys;
    void* output_values;
    if (iterator_count > 0) {
        output_keys = MemoryManager::Malloc(this->dsize_key_ * iterator_count,
                                            this->device_);
        output_values = MemoryManager::Malloc(
                this->dsize_value_ * iterator_count, this->device_);

        UnpackIterators(output_iterators, /* masks = */ nullptr, output_keys,
                        output_values, iterator_count);
    }

    float avg_ratio = float(this->capacity_) / float(this->bucket_count_);
    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
    Allocate(buckets, uint32_t(std::ceil(buckets * avg_ratio)));

    /// Insert back
    if (iterator_count > 0) {
        auto output_masks = (bool*)MemoryManager::Malloc(
                sizeof(bool) * iterator_count, this->device_);

        Insert(output_keys, output_values, output_iterators, output_masks,
               iterator_count);

        MemoryManager::Free(output_keys, this->device_);
        MemoryManager::Free(output_values, this->device_);
        MemoryManager::Free(output_masks, this->device_);
    }

    MemoryManager::Free(output_iterators, this->device_);
}

/// Bucket-related utilities
/// Return number of elems per bucket
template <typename Hash, typename KeyEq>
std::vector<size_t> CUDAHashmap<Hash, KeyEq>::BucketSizes() const {
    auto elems_per_bucket_buffer = static_cast<size_t*>(MemoryManager::Malloc(
            gpu_context_.bucket_count_ * sizeof(size_t), this->device_));

    thrust::device_vector<size_t> elems_per_bucket(
            elems_per_bucket_buffer,
            elems_per_bucket_buffer + gpu_context_.bucket_count_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.capacity_ + blocksize - 1) / blocksize;
    CountElemsPerBucketKernel<<<num_blocks, blocksize>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()));

    std::vector<size_t> result(gpu_context_.bucket_count_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    MemoryManager::Free(elems_per_bucket_buffer, this->device_);
    return std::move(result);
}

/// Return size / bucket_count
template <typename Hash, typename KeyEq>
float CUDAHashmap<Hash, KeyEq>::LoadFactor() const {
    auto elems_per_bucket = BucketSizes();
    int total_elems_stored = std::accumulate(elems_per_bucket.begin(),
                                             elems_per_bucket.end(), 0);

    node_mgr_->gpu_context_ = gpu_context_.node_mgr_ctx_;

    /// Unrelated factor for now
    // auto slabs_per_bucket = node_mgr_->CountSlabsPerSuperblock();
    // int total_slabs_stored =
    //         std::accumulate(slabs_per_bucket.begin(),
    //         slabs_per_bucket.end(),
    //                         gpu_context_.bucket_count_);

    float load_factor =
            float(total_elems_stored) / float(elems_per_bucket.size());

    return load_factor;
}

}  // namespace core
}  // namespace open3d
