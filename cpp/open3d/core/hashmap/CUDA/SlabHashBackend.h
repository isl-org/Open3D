// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/hashmap/CUDA/SlabHashBackendImpl.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"

namespace open3d {
namespace core {
template <typename Key, typename Hash, typename Eq>
class SlabHashBackend : public DeviceHashBackend {
public:
    SlabHashBackend(int64_t init_capacity,
                    int64_t key_dsize,
                    const std::vector<int64_t>& value_dsizes,
                    const Device& device);

    ~SlabHashBackend();

    void Reserve(int64_t capacity) override;

    void Insert(const void* input_keys,
                const std::vector<const void*>& input_values_soa,
                buf_index_t* output_buf_indices,
                bool* output_masks,
                int64_t count) override;

    void Find(const void* input_keys,
              buf_index_t* output_buf_indices,
              bool* output_masks,
              int64_t count) override;

    void Erase(const void* input_keys,
               bool* output_masks,
               int64_t count) override;

    int64_t GetActiveIndices(buf_index_t* output_indices) override;
    void Clear() override;

    int64_t Size() const override;
    int64_t GetBucketCount() const override;
    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

    SlabHashBackendImpl<Key, Hash, Eq> GetImpl() { return impl_; }

    void Allocate(int64_t capacity) override;
    void Free() override;

protected:
    /// The struct is directly passed to kernels by value, so cannot be a
    /// shared pointer.
    SlabHashBackendImpl<Key, Hash, Eq> impl_;

    CUDAHashBackendBufferAccessor buffer_accessor_;
    std::shared_ptr<SlabNodeManager> node_mgr_;

    int64_t bucket_count_;
};

template <typename Key, typename Hash, typename Eq>
SlabHashBackend<Key, Hash, Eq>::SlabHashBackend(
        int64_t init_capacity,
        int64_t key_dsize,
        const std::vector<int64_t>& value_dsizes,
        const Device& device)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device) {
    CUDAScopedDevice scoped_device(this->device_);
    Allocate(init_capacity);
}

template <typename Key, typename Hash, typename Eq>
SlabHashBackend<Key, Hash, Eq>::~SlabHashBackend() {
    CUDAScopedDevice scoped_device(this->device_);
    Free();
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Reserve(int64_t capacity) {
    CUDAScopedDevice scoped_device(this->device_);
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Find(const void* input_keys,
                                          buf_index_t* output_buf_indices,
                                          bool* output_masks,
                                          int64_t count) {
    CUDAScopedDevice scoped_device(this->device_);
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    FindKernel<<<num_blocks, kThreadsPerBlock, 0, core::cuda::GetStream()>>>(
            impl_, input_keys, output_buf_indices, output_masks, count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Erase(const void* input_keys,
                                           bool* output_masks,
                                           int64_t count) {
    CUDAScopedDevice scoped_device(this->device_);
    if (count == 0) return;

    OPEN3D_CUDA_CHECK(cudaMemset(output_masks, 0, sizeof(bool) * count));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
    auto buf_indices = static_cast<buf_index_t*>(
            MemoryManager::Malloc(sizeof(buf_index_t) * count, this->device_));

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    EraseKernelPass0<<<num_blocks, kThreadsPerBlock, 0,
                       core::cuda::GetStream()>>>(
            impl_, input_keys, buf_indices, output_masks, count);
    EraseKernelPass1<<<num_blocks, kThreadsPerBlock, 0,
                       core::cuda::GetStream()>>>(impl_, buf_indices,
                                                  output_masks, count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    MemoryManager::Free(buf_indices, this->device_);
}

template <typename Key, typename Hash, typename Eq>
int64_t SlabHashBackend<Key, Hash, Eq>::GetActiveIndices(
        buf_index_t* output_buf_indices) {
    CUDAScopedDevice scoped_device(this->device_);
    uint32_t* count = static_cast<uint32_t*>(
            MemoryManager::Malloc(sizeof(uint32_t), this->device_));
    OPEN3D_CUDA_CHECK(cudaMemset(count, 0, sizeof(uint32_t)));

    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    const int64_t num_blocks =
            (impl_.bucket_count_ * kWarpSize + kThreadsPerBlock - 1) /
            kThreadsPerBlock;
    GetActiveIndicesKernel<<<num_blocks, kThreadsPerBlock, 0,
                             core::cuda::GetStream()>>>(
            impl_, output_buf_indices, count);
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    uint32_t ret;
    MemoryManager::MemcpyToHost(&ret, count, this->device_, sizeof(uint32_t));
    MemoryManager::Free(count, this->device_);

    return static_cast<int64_t>(ret);
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Clear() {
    CUDAScopedDevice scoped_device(this->device_);
    // Clear the heap
    this->buffer_->ResetHeap();

    // Clear the linked list heads
    OPEN3D_CUDA_CHECK(cudaMemset(impl_.bucket_list_head_, 0xFF,
                                 sizeof(Slab) * this->bucket_count_));
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());

    // Clear the linked list nodes
    node_mgr_->Reset();
}

template <typename Key, typename Hash, typename Eq>
int64_t SlabHashBackend<Key, Hash, Eq>::Size() const {
    CUDAScopedDevice scoped_device(this->device_);
    return this->buffer_->GetHeapTopIndex();
}

template <typename Key, typename Hash, typename Eq>
int64_t SlabHashBackend<Key, Hash, Eq>::GetBucketCount() const {
    CUDAScopedDevice scoped_device(this->device_);
    return bucket_count_;
}

template <typename Key, typename Hash, typename Eq>
std::vector<int64_t> SlabHashBackend<Key, Hash, Eq>::BucketSizes() const {
    CUDAScopedDevice scoped_device(this->device_);
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

template <typename Key, typename Hash, typename Eq>
float SlabHashBackend<Key, Hash, Eq>::LoadFactor() const {
    CUDAScopedDevice scoped_device(this->device_);
    return float(Size()) / float(this->bucket_count_);
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Insert(
        const void* input_keys,
        const std::vector<const void*>& input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count) {
    CUDAScopedDevice scoped_device(this->device_);
    if (count == 0) return;

    /// Increase heap_top to pre-allocate potential memory increment and
    /// avoid atomicAdd in kernel.
    int prev_heap_top = this->buffer_->GetHeapTopIndex();
    *thrust::device_ptr<int>(impl_.buffer_accessor_.heap_top_) =
            prev_heap_top + count;

    const int64_t num_blocks =
            (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    InsertKernelPass0<<<num_blocks, kThreadsPerBlock, 0,
                        core::cuda::GetStream()>>>(
            impl_, input_keys, output_buf_indices, prev_heap_top, count);
    InsertKernelPass1<<<num_blocks, kThreadsPerBlock, 0,
                        core::cuda::GetStream()>>>(
            impl_, input_keys, output_buf_indices, output_masks, count);

    thrust::device_vector<const void*> input_values_soa_device(
            input_values_soa.begin(), input_values_soa.end());

    int64_t n_values = input_values_soa.size();
    const void* const* ptr_input_values_soa =
            thrust::raw_pointer_cast(input_values_soa_device.data());
    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T(
            impl_.buffer_accessor_.common_block_size_, [&]() {
                InsertKernelPass2<Key, Hash, Eq, block_t>
                        <<<num_blocks, kThreadsPerBlock, 0,
                           core::cuda::GetStream()>>>(
                                impl_, ptr_input_values_soa, output_buf_indices,
                                output_masks, count, n_values);
            });
    cuda::Synchronize();
    OPEN3D_CUDA_CHECK(cudaGetLastError());
}

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Allocate(int64_t capacity) {
    CUDAScopedDevice scoped_device(this->device_);
    this->bucket_count_ = capacity * 2;
    this->capacity_ = capacity;

    // Allocate buffer for key values.
    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);
    buffer_accessor_.Setup(*this->buffer_);

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

template <typename Key, typename Hash, typename Eq>
void SlabHashBackend<Key, Hash, Eq>::Free() {
    CUDAScopedDevice scoped_device(this->device_);
    buffer_accessor_.Shutdown(this->device_);
    MemoryManager::Free(impl_.bucket_list_head_, this->device_);
}
}  // namespace core
}  // namespace open3d
