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
    CUDAHashmap(int64_t init_buckets,
                int64_t init_capacity,
                int64_t dsize_key,
                int64_t dsize_value,
                const Device& device);

    ~CUDAHashmap();

    void Rehash(int64_t buckets) override;

    void Insert(const void* input_keys,
                const void* input_values,
                iterator_t* output_iterators,
                bool* output_masks,
                int64_t count) override;

    void Activate(const void* input_keys,
                  iterator_t* output_iterators,
                  bool* output_masks,
                  int64_t count) override;

    void Find(const void* input_keys,
              iterator_t* output_iterators,
              bool* output_masks,
              int64_t count) override;

    void Erase(const void* input_keys,
               bool* output_masks,
               int64_t count) override;

    int64_t GetIterators(iterator_t* output_iterators) override;

    void UnpackIterators(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         int64_t count) override;

    void AssignIterators(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         int64_t count) override;

    std::vector<int64_t> BucketSizes() const override;

    float LoadFactor() const override;

    int64_t Size() const override;

    Tensor GetKeyBlobAsTensor(const SizeVector& shape, Dtype dtype) override;
    Tensor GetValueBlobAsTensor(const SizeVector& shape, Dtype dtype) override;

protected:
    /// The struct is directly passed to kernels by value, so cannot be a shared
    /// pointer.
    CUDAHashmapImplContext<Hash, KeyEq> gpu_context_;

    std::shared_ptr<CUDAKvPairs> kv_pairs_;
    std::shared_ptr<InternalNodeManager> node_mgr_;

    /// Rehash, Insert, Activate all call InsertImpl. It will be clean to
    /// separate this implementation and avoid shared checks.
    void InsertImpl(const void* input_keys,
                    const void* input_values,
                    iterator_t* output_iterators,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t bucket_count, int64_t capacity);
};

template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::CUDAHashmap(int64_t init_buckets,
                                      int64_t init_capacity,
                                      int64_t dsize_key,
                                      int64_t dsize_value,
                                      const Device& device)
    : DeviceHashmap<Hash, KeyEq>(
              init_buckets, init_capacity, dsize_key, dsize_value, device) {
    Allocate(init_buckets, init_capacity);
}

template <typename Hash, typename KeyEq>
CUDAHashmap<Hash, KeyEq>::~CUDAHashmap() {
    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Rehash(int64_t buckets) {
    int64_t iterator_count = Size();

    void* output_keys = nullptr;
    void* output_values = nullptr;
    iterator_t* output_iterators = nullptr;
    bool* output_masks = nullptr;

    if (iterator_count > 0) {
        output_keys = MemoryManager::Malloc(this->dsize_key_ * iterator_count,
                                            this->device_);
        output_values = MemoryManager::Malloc(
                this->dsize_value_ * iterator_count, this->device_);
        output_iterators = static_cast<iterator_t*>(MemoryManager::Malloc(
                sizeof(iterator_t) * iterator_count, this->device_));
        output_masks = static_cast<bool*>(MemoryManager::Malloc(
                sizeof(bool) * iterator_count, this->device_));

        GetIterators(output_iterators);
        UnpackIterators(output_iterators, /* masks = */ nullptr, output_keys,
                        output_values, iterator_count);
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(this->bucket_count_);
    MemoryManager::Free(gpu_context_.bucket_list_head_, this->device_);
    Allocate(buckets, int64_t(std::ceil(buckets * avg_capacity_per_bucket)));

    if (iterator_count > 0) {
        InsertImpl(output_keys, output_values, output_iterators, output_masks,
                   iterator_count);

        MemoryManager::Free(output_keys, this->device_);
        MemoryManager::Free(output_values, this->device_);
        MemoryManager::Free(output_masks, this->device_);
        MemoryManager::Free(output_iterators, this->device_);
    }
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                      const void* input_values,
                                      iterator_t* output_iterators,
                                      bool* output_masks,
                                      int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        int64_t expected_buckets = std::max(
                this->bucket_count_ * 2,
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }

    InsertImpl(input_keys, input_values, output_iterators, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Activate(const void* input_keys,
                                        iterator_t* output_iterators,
                                        bool* output_masks,
                                        int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        int64_t expected_buckets = std::max(
                this->bucket_count_ * 2,
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }

    InsertImpl(input_keys, nullptr, output_iterators, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                    iterator_t* output_iterators,
                                    bool* output_masks,
                                    int64_t count) {
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    FindKernel<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, input_keys, output_iterators, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                     bool* output_masks,
                                     int64_t count) {
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    auto iterator_addrs = static_cast<addr_t*>(
            MemoryManager::Malloc(sizeof(addr_t) * count, this->device_));

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    EraseKernelPass0<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, input_keys, iterator_addrs, output_masks, count);
    EraseKernelPass1<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, iterator_addrs, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(iterator_addrs, this->device_);
}

template <typename Hash, typename KeyEq>
int64_t CUDAHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    uint32_t* iterator_count = static_cast<uint32_t*>(
            MemoryManager::Malloc(sizeof(uint32_t), this->device_));
    cudaMemset(iterator_count, 0, sizeof(uint32_t));

    const int64_t num_blocks =
            (gpu_context_.bucket_count_ * kWarpSize + kThreadsPerBlock - 1) /
            kThreadsPerBlock;
    GetIteratorsKernel<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, output_iterators, iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    uint32_t ret;
    MemoryManager::MemcpyToHost(&ret, iterator_count, this->device_,
                                sizeof(uint32_t));
    MemoryManager::Free(iterator_count, this->device_);

    return static_cast<int64_t>(ret);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::UnpackIterators(
        const iterator_t* input_iterators,
        const bool* input_masks,
        void* output_keys,
        void* output_values,
        int64_t iterator_count) {
    if (iterator_count == 0) return;

    const int64_t num_blocks =
            (iterator_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    UnpackIteratorsKernel<<<num_blocks, kThreadsPerBlock>>>(
            input_iterators, input_masks, output_keys, output_values,
            this->dsize_key_, this->dsize_value_, iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::AssignIterators(iterator_t* input_iterators,
                                               const bool* input_masks,
                                               const void* input_values,
                                               int64_t iterator_count) {
    if (iterator_count == 0) return;

    const int64_t num_blocks =
            (iterator_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    AssignIteratorsKernel<<<num_blocks, kThreadsPerBlock>>>(
            input_iterators, input_masks, input_values, this->dsize_value_,
            iterator_count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Hash, typename KeyEq>
std::vector<int64_t> CUDAHashmap<Hash, KeyEq>::BucketSizes() const {
    thrust::device_vector<int64_t> elems_per_bucket(gpu_context_.bucket_count_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const int64_t num_blocks =
            (gpu_context_.capacity_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
    CountElemsPerBucketKernel<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, thrust::raw_pointer_cast(elems_per_bucket.data()));
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    std::vector<int64_t> result(gpu_context_.bucket_count_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    return std::move(result);
}

template <typename Hash, typename KeyEq>
float CUDAHashmap<Hash, KeyEq>::LoadFactor() const {
    return float(Size()) / float(this->bucket_count_);
}

template <typename Hash, typename KeyEq>
int64_t CUDAHashmap<Hash, KeyEq>::Size() const {
    return *thrust::device_ptr<int>(gpu_context_.kv_mgr_ctx_.heap_counter_);
}

template <typename Hash, typename KeyEq>
Tensor CUDAHashmap<Hash, KeyEq>::GetKeyBlobAsTensor(const SizeVector& shape,
                                                    Dtype dtype) {
    if (dtype.ByteSize() * shape.NumElements() !=
        this->capacity_ * this->dsize_key_) {
        utility::LogError(
                "[CUDAHashmap] Tensor shape and dtype mismatch with key blob "
                "size");
    }
    return Tensor(shape, Tensor::DefaultStrides(shape),
                  kv_pairs_->GetKeyBlob()->GetDataPtr(), dtype,
                  kv_pairs_->GetKeyBlob());
}

template <typename Hash, typename KeyEq>
Tensor CUDAHashmap<Hash, KeyEq>::GetValueBlobAsTensor(const SizeVector& shape,
                                                      Dtype dtype) {
    if (dtype.ByteSize() * shape.NumElements() !=
        this->capacity_ * this->dsize_value_) {
        utility::LogError(
                "[CUDAHashmap] Tensor shape and dtype mismatch with value blob "
                "size: {} vs {}");
    }
    return Tensor(shape, Tensor::DefaultStrides(shape),
                  kv_pairs_->GetValueBlob()->GetDataPtr(), dtype,
                  kv_pairs_->GetValueBlob());
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::InsertImpl(const void* input_keys,
                                          const void* input_values,
                                          iterator_t* output_iterators,
                                          bool* output_masks,
                                          int64_t count) {
    if (count == 0) return;
    auto iterator_addrs = static_cast<addr_t*>(
            MemoryManager::Malloc(sizeof(addr_t) * count, this->device_));

    /// Increase heap_counter to pre-allocate potential memory increment and
    /// avoid atomicAdd in kernel.
    int prev_heap_counter =
            *thrust::device_ptr<int>(gpu_context_.kv_mgr_ctx_.heap_counter_);
    *thrust::device_ptr<int>(gpu_context_.kv_mgr_ctx_.heap_counter_) =
            prev_heap_counter + count;

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    InsertKernelPass0<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, input_keys, iterator_addrs, prev_heap_counter, count);
    InsertKernelPass1<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, input_keys, iterator_addrs, output_masks, count);
    InsertKernelPass2<<<num_blocks, kThreadsPerBlock>>>(
            gpu_context_, input_values, iterator_addrs, output_iterators,
            output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(iterator_addrs, this->device_);
}

template <typename Hash, typename KeyEq>
void CUDAHashmap<Hash, KeyEq>::Allocate(int64_t bucket_count,
                                        int64_t capacity) {
    this->bucket_count_ = bucket_count;
    this->capacity_ = capacity;

    // Allocate buffer for key-values.
    kv_pairs_ =
            std::make_shared<CUDAKvPairs>(this->capacity_, this->dsize_key_,
                                          this->dsize_value_, this->device_);

    // Allocate buffer for linked list nodes.
    node_mgr_ = std::make_shared<InternalNodeManager>(this->device_);

    // Allocate linked list heads.
    gpu_context_.bucket_list_head_ = static_cast<Slab*>(MemoryManager::Malloc(
            sizeof(Slab) * this->bucket_count_, this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(gpu_context_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * this->bucket_count_));

    gpu_context_.Setup(this->bucket_count_, this->capacity_, this->dsize_key_,
                       this->dsize_value_, node_mgr_->gpu_context_,
                       kv_pairs_->GetContext());
}

}  // namespace core
}  // namespace open3d
