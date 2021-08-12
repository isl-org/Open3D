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

#pragma once

#include <memory>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/hashmap/CUDA/SlabHashmapImpl.h"
#include "open3d/core/hashmap/DeviceHashmap.h"

namespace open3d {
namespace core {
template <typename Key, typename Hash>
class SlabHashmap : public DeviceHashmap {
public:
    SlabHashmap(int64_t init_capacity,
                int64_t dsize_key,
                std::vector<int64_t> dsize_values,
                const Device& device);

    ~SlabHashmap();

    void Rehash(int64_t buckets) override;

    void Insert(const void* input_keys,
                const std::vector<const void*> input_values,
                addr_t* output_addrs,
                bool* output_masks,
                int64_t count) override;

    void Activate(const void* input_keys,
                  addr_t* output_addrs,
                  bool* output_masks,
                  int64_t count) override;

    void Find(const void* input_keys,
              addr_t* output_addrs,
              bool* output_masks,
              int64_t count) override;

    void Erase(const void* input_keys,
               bool* output_masks,
               int64_t count) override;

    int64_t GetActiveIndices(addr_t* output_indices) override;
    void Clear() override;

    int64_t Size() const override;
    int64_t GetBucketCount() const override;
    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

    SlabHashmapImpl<Key, Hash> GetImpl() { return impl_; }

protected:
    /// The struct is directly passed to kernels by value, so cannot be a
    /// shared pointer.
    SlabHashmapImpl<Key, Hash> impl_;

    CUDAHashmapBufferAccessor buffer_accessor_;
    std::shared_ptr<SlabNodeManager> node_mgr_;

    /// Rehash, Insert, Activate all call InsertImpl. It will be clean to
    /// separate this implementation and avoid shared checks.
    void InsertImpl(const void* input_keys,
                    const std::vector<const void*> input_values,
                    addr_t* output_addrs,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t bucket_count, int64_t capacity);
    void Free();

    int64_t bucket_count_;
};

template <typename Key, typename Hash>
SlabHashmap<Key, Hash>::SlabHashmap(int64_t init_capacity,
                                    int64_t dsize_key,
                                    std::vector<int64_t> dsize_values,
                                    const Device& device)
    : DeviceHashmap(init_capacity, dsize_key, dsize_values, device) {
    int64_t init_buckets = init_capacity * 2;
    Allocate(init_buckets, init_capacity);
}

template <typename Key, typename Hash>
SlabHashmap<Key, Hash>::~SlabHashmap() {
    Free();
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Rehash(int64_t buckets) {
    int64_t iterator_count = Size();

    Tensor active_keys;
    std::vector<Tensor> active_values;

    if (iterator_count > 0) {
        Tensor active_addrs =
                Tensor({iterator_count}, core::Int32, this->device_);
        GetActiveIndices(static_cast<addr_t*>(active_addrs.GetDataPtr()));

        Tensor active_indices = active_addrs.To(core::Int64);

        active_keys = this->buffer_->GetKeyBuffer().IndexGet({active_indices});
        auto value_buffers = this->GetValueBuffers();
        for (auto& value_buffer : value_buffers) {
            active_values.emplace_back(value_buffer.IndexGet({active_indices}));
        }
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(this->bucket_count_);

    Free();

    Allocate(buckets,
             std::max(int64_t(std::ceil(buckets * avg_capacity_per_bucket)),
                      active_keys.GetLength()));

    if (iterator_count > 0) {
        Tensor output_addrs({iterator_count}, core::Int32, this->device_);
        Tensor output_masks({iterator_count}, core::Bool, this->device_);

        std::vector<const void*> active_value_ptrs;
        for (auto& active_value : active_values) {
            active_value_ptrs.push_back(active_value.GetDataPtr());
        }
        InsertImpl(active_keys.GetDataPtr(), active_value_ptrs,
                   static_cast<addr_t*>(output_addrs.GetDataPtr()),
                   output_masks.GetDataPtr<bool>(), iterator_count);
    }
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Insert(const void* input_keys,
                                    std::vector<const void*> input_values,
                                    addr_t* output_addrs,
                                    bool* output_masks,
                                    int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        int64_t expected_buckets = std::max(
                int64_t(this->bucket_count_ * 2),
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }

    InsertImpl(input_keys, input_values, output_addrs, output_masks, count);
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Activate(const void* input_keys,
                                      addr_t* output_addrs,
                                      bool* output_masks,
                                      int64_t count) {
    std::vector<const void*> null_values;
    Insert(input_keys, null_values, output_addrs, output_masks, count);
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Find(const void* input_keys,
                                  addr_t* output_addrs,
                                  bool* output_masks,
                                  int64_t count) {
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    FindKernel<<<num_blocks, kThreadsPerBlock, 0, core::cuda::GetStream()>>>(
            impl_, input_keys, output_addrs, output_masks, count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Erase(const void* input_keys,
                                   bool* output_masks,
                                   int64_t count) {
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    auto iterator_addrs = static_cast<addr_t*>(
            MemoryManager::Malloc(sizeof(addr_t) * count, this->device_));

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    EraseKernelPass0<<<num_blocks, kThreadsPerBlock, 0,
                       core::cuda::GetStream()>>>(
            impl_, input_keys, iterator_addrs, output_masks, count);
    EraseKernelPass1<<<num_blocks, kThreadsPerBlock, 0,
                       core::cuda::GetStream()>>>(impl_, iterator_addrs,
                                                  output_masks, count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(iterator_addrs, this->device_);
}

template <typename Key, typename Hash>
int64_t SlabHashmap<Key, Hash>::GetActiveIndices(addr_t* output_addrs) {
    uint32_t* iterator_count = static_cast<uint32_t*>(
            MemoryManager::Malloc(sizeof(uint32_t), this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(iterator_count, 0, sizeof(uint32_t)));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    const int64_t num_blocks =
            (impl_.bucket_count_ * kWarpSize + kThreadsPerBlock - 1) /
            kThreadsPerBlock;
    GetActiveIndicesKernel<<<num_blocks, kThreadsPerBlock, 0,
                             core::cuda::GetStream()>>>(impl_, output_addrs,
                                                        iterator_count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    uint32_t ret;
    MemoryManager::MemcpyToHost(&ret, iterator_count, this->device_,
                                sizeof(uint32_t));
    MemoryManager::Free(iterator_count, this->device_);

    return static_cast<int64_t>(ret);
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Clear() {
    // Clear the heap
    buffer_accessor_.Reset(this->device_);

    // Clear the linked list heads
    OPEN3D_CUDA_CHECK(cudaMemset(impl_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * this->bucket_count_));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    // Clear the linked list nodes
    node_mgr_->Reset();
}

template <typename Key, typename Hash>
int64_t SlabHashmap<Key, Hash>::Size() const {
    return buffer_accessor_.HeapCounter(this->device_);
}

template <typename Key, typename Hash>
int64_t SlabHashmap<Key, Hash>::GetBucketCount() const {
    return bucket_count_;
}

template <typename Key, typename Hash>
std::vector<int64_t> SlabHashmap<Key, Hash>::BucketSizes() const {
    thrust::device_vector<int64_t> elems_per_bucket(impl_.bucket_count_);
    thrust::fill(elems_per_bucket.begin(), elems_per_bucket.end(), 0);

    const int64_t num_blocks =
            (impl_.buffer_accessor_.capacity_ + kThreadsPerBlock - 1) /
            kThreadsPerBlock;
    CountElemsPerBucketKernel<<<num_blocks, kThreadsPerBlock, 0,
                                core::cuda::GetStream()>>>(
            impl_, thrust::raw_pointer_cast(elems_per_bucket.data()));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    std::vector<int64_t> result(impl_.bucket_count_);
    thrust::copy(elems_per_bucket.begin(), elems_per_bucket.end(),
                 result.begin());
    return result;
}

template <typename Key, typename Hash>
float SlabHashmap<Key, Hash>::LoadFactor() const {
    return float(Size()) / float(this->bucket_count_);
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::InsertImpl(
        const void* input_keys,
        std::vector<const void*> input_values_host,
        addr_t* output_addrs,
        bool* output_masks,
        int64_t count) {
    if (count == 0) return;

    /// Increase heap_counter to pre-allocate potential memory increment and
    /// avoid atomicAdd in kernel.
    int prev_heap_counter = buffer_accessor_.HeapCounter(this->device_);
    *thrust::device_ptr<int>(impl_.buffer_accessor_.heap_counter_) =
            prev_heap_counter + count;

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    InsertKernelPass0<<<num_blocks, kThreadsPerBlock, 0,
                        core::cuda::GetStream()>>>(
            impl_, input_keys, output_addrs, prev_heap_counter, count);
    InsertKernelPass1<<<num_blocks, kThreadsPerBlock, 0,
                        core::cuda::GetStream()>>>(
            impl_, input_keys, output_addrs, output_masks, count);

    thrust::device_vector<const void*> input_values(input_values_host.begin(),
                                                    input_values_host.end());
    int64_t n_values = input_values.size() == impl_.buffer_accessor_.n_values_
                               ? impl_.buffer_accessor_.n_values_
                               : 0;
    // https://stackoverflow.com/a/37998941
    const void* const* input_values_ptr =
            thrust::raw_pointer_cast(input_values.data());
    InsertKernelPass2<<<num_blocks, kThreadsPerBlock, 0,
                        core::cuda::GetStream()>>>(impl_, input_values_ptr,
                                                   output_addrs, output_masks,
                                                   count, n_values);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Allocate(int64_t bucket_count, int64_t capacity) {
    this->bucket_count_ = bucket_count;
    this->capacity_ = capacity;

    // Allocate buffer for key values.
    this->buffer_ =
            std::make_shared<HashmapBuffer>(this->capacity_, this->dsize_key_,
                                            this->dsize_values_, this->device_);
    buffer_accessor_.HostAllocate(this->device_);
    buffer_accessor_.Setup(this->capacity_, this->dsize_key_,
                           this->dsize_values_, this->buffer_->GetKeyBuffer(),
                           this->buffer_->GetValueBuffers(),
                           this->buffer_->GetIndexHeap());
    buffer_accessor_.Reset(this->device_);

    // Allocate buffer for linked list nodes.
    node_mgr_ = std::make_shared<SlabNodeManager>(this->device_);

    // Allocate linked list heads.
    impl_.bucket_list_head_ = static_cast<Slab*>(MemoryManager::Malloc(
            sizeof(Slab) * this->bucket_count_, this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(impl_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * this->bucket_count_));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    impl_.Setup(this->bucket_count_, node_mgr_->impl_, buffer_accessor_);
}

template <typename Key, typename Hash>
void SlabHashmap<Key, Hash>::Free() {
    buffer_accessor_.HostFree(this->device_);
    buffer_accessor_.Shutdown(this->device_);
    MemoryManager::Free(impl_.bucket_list_head_, this->device_);
}
}  // namespace core
}  // namespace open3d
