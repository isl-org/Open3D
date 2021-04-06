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

#pragma once

#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <limits>
#include <stdgpu/unordered_map.cuh>  // stdgpu::unordered_map
#include <unordered_map>

#include "open3d/core/hashmap/CUDA/CUDAHashmapBufferAccessor.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/kernel/CUDALauncher.cuh"

namespace open3d {
namespace core {
template <typename Key, typename Hash>
class StdGPUHashmap : public DeviceHashmap {
public:
    StdGPUHashmap(int64_t init_capacity,
                  int64_t dsize_key,
                  int64_t dsize_value,
                  const Device& device);
    ~StdGPUHashmap();

    void Rehash(int64_t buckets) override;

    void Insert(const void* input_keys,
                const void* input_values,
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

    int64_t Size() const override;

    int64_t GetBucketCount() const override;
    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

    stdgpu::unordered_map<Key, addr_t, Hash> GetImpl() const { return impl_; }

protected:
    // Use reference, since the structure itself is implicitly handled as a
    // pointer directly by stdgpu.
    stdgpu::unordered_map<Key, addr_t, Hash> impl_;

    CUDAHashmapBufferAccessor buffer_accessor_;

    void InsertImpl(const void* input_keys,
                    const void* input_values,
                    addr_t* output_addrs,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t capacity);
    void Free();
};

template <typename Key, typename Hash>
StdGPUHashmap<Key, Hash>::StdGPUHashmap(int64_t init_capacity,
                                        int64_t dsize_key,
                                        int64_t dsize_value,
                                        const Device& device)
    : DeviceHashmap(init_capacity, dsize_key, dsize_value, device) {
    Allocate(init_capacity);
}

template <typename Key, typename Hash>
StdGPUHashmap<Key, Hash>::~StdGPUHashmap() {
    Free();
}

template <typename Key, typename Hash>
int64_t StdGPUHashmap<Key, Hash>::Size() const {
    return impl_.size();
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Insert(const void* input_keys,
                                      const void* input_values,
                                      addr_t* output_addrs,
                                      bool* output_masks,
                                      int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        int64_t bucket_count = GetBucketCount();
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(bucket_count);
        int64_t expected_buckets = std::max(
                bucket_count * 2,
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }
    InsertImpl(input_keys, input_values, output_addrs, output_masks, count);
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Activate(const void* input_keys,
                                        addr_t* output_addrs,
                                        bool* output_masks,
                                        int64_t count) {
    Insert(input_keys, nullptr, output_addrs, output_masks, count);
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash>
__global__ void STDGPUFindKernel(stdgpu::unordered_map<Key, addr_t, Hash> map,
                                 CUDAHashmapBufferAccessor buffer_accessor,
                                 const Key* input_keys,
                                 addr_t* output_addrs,
                                 bool* output_masks,
                                 int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    auto iter = map.find(key);
    bool flag = (iter != map.end());
    output_masks[tid] = flag;
    output_addrs[tid] = flag ? iter->second : 0;
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Find(const void* input_keys,
                                    addr_t* output_addrs,
                                    bool* output_masks,
                                    int64_t count) {
    int threads = 32;
    int blocks = (count + threads - 1) / threads;

    STDGPUFindKernel<<<blocks, threads>>>(impl_, buffer_accessor_,
                                          static_cast<const Key*>(input_keys),
                                          output_addrs, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash>
__global__ void STDGPUEraseKernel(stdgpu::unordered_map<Key, addr_t, Hash> map,
                                  CUDAHashmapBufferAccessor buffer_accessor,
                                  const Key* input_keys,
                                  addr_t* output_addrs,
                                  bool* output_masks,
                                  int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    if (output_masks[tid]) {
        output_masks[tid] = map.erase(key);
        if (output_masks[tid]) {
            buffer_accessor.DeviceFree(output_addrs[tid]);
        }
    }
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Erase(const void* input_keys,
                                     bool* output_masks,
                                     int64_t count) {
    stdgpu::index_t threads = 32;
    stdgpu::index_t blocks = (count + threads - 1) / threads;

    // Erase has to go in two passes -- find the iterator, then erase and free
    // Not frequently used, may not be fully optimized due to the tricky
    // iterator change in the erase operation
    core::Tensor toutput_addrs =
            core::Tensor({count}, Dtype::Int32, this->device_);
    addr_t* output_addrs = static_cast<addr_t*>(toutput_addrs.GetDataPtr());

    STDGPUFindKernel<<<blocks, threads>>>(impl_, buffer_accessor_,
                                          static_cast<const Key*>(input_keys),
                                          output_addrs, output_masks, count);
    STDGPUEraseKernel<<<blocks, threads>>>(impl_, buffer_accessor_,
                                           static_cast<const Key*>(input_keys),
                                           output_addrs, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Key>
struct ValueExtractor {
    OPEN3D_HOST_DEVICE addr_t
    operator()(const thrust::pair<Key, addr_t>& x) const {
        return x.second;
    }
};

template <typename Key, typename Hash>
int64_t StdGPUHashmap<Key, Hash>::GetActiveIndices(addr_t* output_indices) {
    auto range = impl_.device_range();

    thrust::transform(range.begin(), range.end(), output_indices,
                      ValueExtractor<Key>());

    return impl_.size();
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Rehash(int64_t buckets) {
    int64_t iterator_count = Size();

    Tensor active_keys;
    Tensor active_values;

    if (iterator_count > 0) {
        Tensor active_addrs({iterator_count}, Dtype::Int32, this->device_);
        GetActiveIndices(static_cast<addr_t*>(active_addrs.GetDataPtr()));

        Tensor active_indices = active_addrs.To(Dtype::Int64);
        active_keys = this->GetKeyBuffer().IndexGet({active_indices});
        active_values = this->GetValueBuffer().IndexGet({active_indices});
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(GetBucketCount());

    int64_t new_capacity =
            int64_t(std::ceil(buckets * avg_capacity_per_bucket));
    Allocate(new_capacity);

    if (iterator_count > 0) {
        Tensor output_addrs({iterator_count}, Dtype::Int32, this->device_);
        Tensor output_masks({iterator_count}, Dtype::Bool, this->device_);

        InsertImpl(active_keys.GetDataPtr(), active_values.GetDataPtr(),
                   static_cast<addr_t*>(output_addrs.GetDataPtr()),
                   output_masks.GetDataPtr<bool>(), iterator_count);
    }
}

template <typename Key, typename Hash>
int64_t StdGPUHashmap<Key, Hash>::GetBucketCount() const {
    return impl_.bucket_count();
}

template <typename Key, typename Hash>
std::vector<int64_t> StdGPUHashmap<Key, Hash>::BucketSizes() const {
    utility::LogError("Unimplemented");
}

template <typename Key, typename Hash>
float StdGPUHashmap<Key, Hash>::LoadFactor() const {
    return impl_.load_factor();
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash>
__global__ void STDGPUInsertKernel(stdgpu::unordered_map<Key, addr_t, Hash> map,
                                   CUDAHashmapBufferAccessor buffer_accessor,
                                   const Key* input_keys,
                                   const void* input_values,
                                   int64_t dsize_value,
                                   addr_t* output_addrs,
                                   bool* output_masks,
                                   int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    output_addrs[tid] = 0;
    output_masks[tid] = false;

    // First apply 'try insert' with a dummy index
    auto res = map.emplace(key, 0);

    // If success, change the iterator and provide the actual index
    if (res.second) {
        addr_t dst_kv_addr = buffer_accessor.DeviceAllocate();
        auto dst_kv_iter = buffer_accessor.ExtractIterator(dst_kv_addr);

        // Copy templated key to buffer (duplicate)
        // TODO: hack stdgpu inside and take out the buffer directly
        *static_cast<Key*>(dst_kv_iter.first) = key;

        // Copy/reset non-templated value in buffer
        uint8_t* dst_value = static_cast<uint8_t*>(dst_kv_iter.second);
        if (input_values != nullptr) {
            const uint8_t* src_value =
                    static_cast<const uint8_t*>(input_values) +
                    dsize_value * tid;
            for (int byte = 0; byte < dsize_value; ++byte) {
                dst_value[byte] = src_value[byte];
            }
        }

        // Update from the dummy index
        res.first->second = dst_kv_addr;

        // Write to return variables
        output_addrs[tid] = dst_kv_addr;
        output_masks[tid] = true;
    }
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::InsertImpl(const void* input_keys,
                                          const void* input_values,
                                          addr_t* output_addrs,
                                          bool* output_masks,
                                          int64_t count) {
    stdgpu::index_t threads = 32;
    stdgpu::index_t blocks = (count + threads - 1) / threads;

    STDGPUInsertKernel<<<blocks, threads>>>(impl_, buffer_accessor_,
                                            static_cast<const Key*>(input_keys),
                                            input_values, this->dsize_value_,
                                            output_addrs, output_masks, count);
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;

    // Allocate buffer for key values.
    this->buffer_ =
            std::make_shared<HashmapBuffer>(this->capacity_, this->dsize_key_,
                                            this->dsize_value_, this->device_);

    buffer_accessor_.HostAllocate(this->device_);
    buffer_accessor_.Setup(this->capacity_, this->dsize_key_,
                           this->dsize_value_, this->buffer_->GetKeyBuffer(),
                           this->buffer_->GetValueBuffer(),
                           this->buffer_->GetHeap());
    buffer_accessor_.Reset(this->device_);

    impl_ = stdgpu::unordered_map<Key, addr_t, Hash>::createDeviceObject(
            this->capacity_);
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Free() {
    // Buffer is automatically handled by the smart pointer.

    buffer_accessor_.HostFree(this->device_);

    stdgpu::unordered_map<Key, addr_t, Hash>::destroyDeviceObject(impl_);
}
}  // namespace core
}  // namespace open3d
