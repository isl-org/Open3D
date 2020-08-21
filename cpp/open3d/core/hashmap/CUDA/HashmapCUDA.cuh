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

#include "open3d/core/hashmap/CUDA/HashmapCUDA.h"
#include "open3d/core/hashmap/CUDA/HashmapCUDAImpl.cuh"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace core {
/// Interface
template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::CUDAHashmap(size_t init_buckets,
                                      size_t init_capacity,
                                      size_t dsize_key,
                                      size_t dsize_value,
                                      Device device)
    : Hashmap<Hash, KeyEq>(
              init_buckets, init_capacity, dsize_key, dsize_value, device) {
    utility::Timer timer;
    timer.Start();

    mem_mgr_ = std::make_shared<InternalKvPairManager>(init_capacity, dsize_key,
                                                       dsize_value, device);

    node_mgr_ = std::make_shared<InternalNodeManager>(device);
    gpu_context_.Setup(init_buckets, init_capacity, dsize_key, dsize_value,
                       node_mgr_->gpu_context_, mem_mgr_->gpu_context_);

    gpu_context_.bucket_list_head_ = static_cast<Slab*>(
            MemoryManager::Malloc(sizeof(Slab) * init_buckets, device));
    OPEN3D_CUDA_CHECK(cudaMemset(gpu_context_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * init_buckets));
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    timer.Stop();
    utility::LogInfo("[HashmapCUDA] constructor {}", timer.GetDuration());
}

template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::~CUDAHashmap() {
    utility::Timer timer;
    timer.Start();
    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
    timer.Stop();
    utility::LogInfo("[HashmapCUDA] destructor takes {}", timer.GetDuration());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                      const void* input_values,
                                      iterator_t* output_iterators,
                                      bool* output_masks,
                                      size_t count) {
    utility::Timer timer;
    timer.Start();
    bool extern_alloc = (output_masks != nullptr);
    if (!extern_alloc) {
        output_masks = (bool*)MemoryManager::Malloc(count * sizeof(bool),
                                                    this->device_);
    }
    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;

    auto iterator_ptrs =
            (ptr_t*)MemoryManager::Malloc(sizeof(ptr_t) * count, this->device_);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    timer.Stop();
    utility::LogDebug("[HashmapCUDA] Preparation takes {}",
                      timer.GetDuration());

    // Batch allocate
    // int index = atomicAdd(heap_counter_, 1);
    // assert(index < max_capacity_);
    // return heap_[index];

    /// Batch allocate iterators
    timer.Start();
    int heap_counter =
            *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_);
    *thrust::device_ptr<int>(gpu_context_.mem_mgr_ctx_.heap_counter_) =
            heap_counter + count;
    int iterator_heap_index0 = *thrust::device_ptr<uint32_t>(
            gpu_context_.mem_mgr_ctx_.heap_ + heap_counter);

    InsertKernelPass0<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, input_keys,
                                                  iterator_ptrs,
                                                  iterator_heap_index0, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    timer.Stop();
    utility::LogDebug("[HashmapCUDA] Pass0 takes {}", timer.GetDuration());

    timer.Start();
    InsertKernelPass1<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_keys, iterator_ptrs, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    timer.Stop();
    utility::LogDebug("[HashmapCUDA] Pass1 takes {}", timer.GetDuration());

    timer.Start();
    InsertKernelPass2<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, input_values, iterator_ptrs, output_iterators,
            output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    timer.Stop();
    utility::LogDebug("[HashmapCUDA] Pass2 takes {}", timer.GetDuration());

    timer.Start();
    MemoryManager::Free(iterator_ptrs, this->device_);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    if (!extern_alloc) {
        MemoryManager::Free(output_masks, this->device_);
    }
    timer.Stop();
    utility::LogDebug("[HashmapCUDA] Free takes {}", timer.GetDuration());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                    iterator_t* output_iterators,
                                    bool* output_masks,
                                    size_t count) {
    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));

    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    FindKernel<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, (uint8_t*)input_keys,
                                           output_iterators, output_masks,
                                           count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                     bool* output_masks,
                                     size_t count) {
    bool extern_alloc = (output_masks != nullptr);
    if (!extern_alloc) {
        output_masks = (bool*)MemoryManager::Malloc(count * sizeof(bool),
                                                    this->device_);
    }

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    const uint32_t num_blocks = (count + BLOCKSIZE_ - 1) / BLOCKSIZE_;

    auto iterator_ptrs =
            (ptr_t*)MemoryManager::Malloc(sizeof(ptr_t) * count, this->device_);

    EraseKernelPass0<<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, (uint8_t*)input_keys, iterator_ptrs, output_masks,
            count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    EraseKernelPass1<<<num_blocks, BLOCKSIZE_>>>(gpu_context_, iterator_ptrs,
                                                 output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(iterator_ptrs, this->device_);

    if (!extern_alloc) {
        MemoryManager::Free(output_masks, this->device_);
    }
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    const uint32_t blocksize = 128;
    const uint32_t num_blocks =
            (gpu_context_.bucket_count_ * 32 + blocksize - 1) / blocksize;

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
    // TODO: add a size operator instead of rough estimation
    auto output_iterators = (iterator_t*)MemoryManager::Malloc(
            sizeof(iterator_t) * this->capacity_, this->device_);
    uint32_t iterator_count = GetIterators(output_iterators);

    auto output_keys = MemoryManager::Malloc(this->dsize_key_ * iterator_count,
                                             this->device_);
    auto output_values = MemoryManager::Malloc(
            this->dsize_value_ * iterator_count, this->device_);

    UnpackIterators(output_iterators, /* masks = */ nullptr, output_keys,
                    output_values, iterator_count);

    this->bucket_count_ = buckets;
    this->capacity_ = buckets * kDefaultElemsPerBucket;

    mem_mgr_ = std::make_shared<InternalKvPairManager>(
            this->capacity_, this->dsize_key_, this->dsize_value_,
            this->device_);
    node_mgr_ = std::make_shared<InternalNodeManager>(this->device_);
    gpu_context_.Setup(this->bucket_count_, this->capacity_, this->dsize_key_,
                       this->dsize_value_, node_mgr_->gpu_context_,
                       mem_mgr_->gpu_context_);

    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
    gpu_context_.bucket_list_head_ = static_cast<Slab*>(
            MemoryManager::Malloc(sizeof(Slab) * buckets, this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(gpu_context_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * buckets));

    /// Insert back
    auto output_masks = (bool*)MemoryManager::Malloc(
            sizeof(bool) * iterator_count, this->device_);
    Insert(output_keys, output_values, output_iterators, output_masks,
           iterator_count);

    MemoryManager::Free(output_iterators, this->device_);
    MemoryManager::Free(output_keys, this->device_);
    MemoryManager::Free(output_values, this->device_);
    MemoryManager::Free(output_masks, this->device_);
}

template <typename Hash, typename KeyEq>
size_t CUDAHashmap<Hash, KeyEq>::Size() {
    return mem_mgr_->heap_counter();
}

/// Bucket-related utilitiesx
/// Return number of elems per bucket
template <typename Hash, typename KeyEq>
std::vector<size_t> CUDAHashmap<Hash, KeyEq>::BucketSizes() {
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
float CUDAHashmap<Hash, KeyEq>::LoadFactor() {
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

template <typename Hash, typename KeyEq>
std::shared_ptr<CUDAHashmap<Hash, KeyEq>> CreateCUDAHashmap(
        size_t init_buckets,
        size_t init_capacity,
        size_t dsize_key,
        size_t dsize_value,
        Device device) {
    return std::make_shared<CUDAHashmap<Hash, KeyEq>>(
            init_buckets, init_capacity, dsize_key, dsize_value, device);
}
}  // namespace core
}  // namespace open3d
