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

#include <stdgpu/memory.h>
#include <thrust/transform.h>

#include <stdgpu/unordered_map.cuh>
#include <type_traits>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/StdAllocator.h"
#include "open3d/core/hashmap/CUDA/CUDAHashmapBufferAccessor.h"
#include "open3d/core/hashmap/DeviceHashmap.h"

namespace open3d {
namespace core {

/// Class satisfying the Allocator requirements defined by the C++ standard.
/// This bridge makes the MemoryManager interface accessible to all classes
/// and containers in stdgpu that use the standard Allocator interface.
///
/// This allows to allocate (potentially cached) GPU memory in stdgpu.
template <typename T>
class StdGPUAllocator {
public:
    /// T.
    using value_type = T;

    /// Default constructor.
    StdGPUAllocator() = default;

    /// Constructor from device.
    explicit StdGPUAllocator(const Device& device) : std_allocator_(device) {}

    /// Default copy constructor.
    StdGPUAllocator(const StdGPUAllocator&) = default;

    /// Default copy assignment operator.
    StdGPUAllocator& operator=(const StdGPUAllocator&) = default;

    /// Default move constructor.
    StdGPUAllocator(StdGPUAllocator&&) = default;

    /// Default move assignment operator.
    StdGPUAllocator& operator=(StdGPUAllocator&&) = default;

    /// Rebind copy constructor.
    template <typename U>
    StdGPUAllocator(const StdGPUAllocator<U>& other)
        : std_allocator_(other.std_allocator_) {}

    /// Allocates memory of size \p n.
    T* allocate(std::size_t n) {
        if (GetDevice().GetType() != Device::DeviceType::CUDA) {
            utility::LogError("Unsupported device.");
        }

        T* p = std_allocator_.allocate(n);
        stdgpu::register_memory(p, n, stdgpu::dynamic_memory_type::device);
        return p;
    }

    /// Deallocates memory from pointer \p p of size \p n .
    void deallocate(T* p, std::size_t n) {
        if (GetDevice().GetType() != Device::DeviceType::CUDA) {
            utility::LogError("Unsupported device.");
        }

        stdgpu::deregister_memory(p, n, stdgpu::dynamic_memory_type::device);
        std_allocator_.deallocate(p, n);
    }

    /// Returns true if the instances are equal, false otherwise.
    bool operator==(const StdGPUAllocator& other) {
        return std_allocator_ == other.std_allocator_;
    }

    /// Returns true if the instances are not equal, false otherwise.
    bool operator!=(const StdGPUAllocator& other) { return !operator==(other); }

    /// Returns the device on which memory is allocated.
    Device GetDevice() const { return std_allocator_.GetDevice(); }

private:
    // Allow access in rebind constructor.
    template <typename T2>
    friend class StdGPUAllocator;

    StdAllocator<T> std_allocator_;
};

// These typedefs must be defined outside of StdGPUHashmap to make them
// accessible in raw CUDA kernels.
template <typename Key>
using InternalStdGPUHashmapAllocator =
        StdGPUAllocator<thrust::pair<const Key, addr_t>>;

template <typename Key, typename Hash>
using InternalStdGPUHashmap =
        stdgpu::unordered_map<Key,
                              addr_t,
                              Hash,
                              stdgpu::equal_to<Key>,
                              InternalStdGPUHashmapAllocator<Key>>;

template <typename Key, typename Hash>
class StdGPUHashmap : public DeviceHashmap {
public:
    StdGPUHashmap(int64_t init_capacity,
                  int64_t dsize_key,
                  std::vector<int64_t> dsize_values,
                  const Device& device);
    ~StdGPUHashmap();

    void Rehash(int64_t buckets) override;

    void Insert(const void* input_keys,
                std::vector<const void*> input_values,
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

    InternalStdGPUHashmap<Key, Hash> GetImpl() const { return impl_; }

protected:
    // Use reference, since the structure itself is implicitly handled as a
    // pointer directly by stdgpu.
    InternalStdGPUHashmap<Key, Hash> impl_;

    CUDAHashmapBufferAccessor buffer_accessor_;

    void InsertImpl(const void* input_keys,
                    std::vector<const void*> input_values,
                    addr_t* output_addrs,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t capacity);
    void Free();
};

template <typename Key, typename Hash>
StdGPUHashmap<Key, Hash>::StdGPUHashmap(int64_t init_capacity,
                                        int64_t dsize_key,
                                        std::vector<int64_t> dsize_values,
                                        const Device& device)
    : DeviceHashmap(init_capacity, dsize_key, dsize_values, device) {
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
                                      std::vector<const void*> input_values,
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
    std::vector<const void*> null_values;
    Insert(input_keys, null_values, output_addrs, output_masks, count);
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash>
__global__ void STDGPUFindKernel(InternalStdGPUHashmap<Key, Hash> map,
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
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    STDGPUFindKernel<<<blocks, threads, 0, core::cuda::GetStream()>>>(
            impl_, buffer_accessor_, static_cast<const Key*>(input_keys),
            output_addrs, output_masks, count);
    cuda::Synchronize(this->device_);
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash>
__global__ void STDGPUEraseKernel(InternalStdGPUHashmap<Key, Hash> map,
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
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    core::Tensor toutput_addrs =
            core::Tensor({count}, core::Int32, this->device_);
    addr_t* output_addrs = static_cast<addr_t*>(toutput_addrs.GetDataPtr());

    STDGPUEraseKernel<<<blocks, threads, 0, core::cuda::GetStream()>>>(
            impl_, buffer_accessor_, static_cast<const Key*>(input_keys),
            output_addrs, output_masks, count);
    cuda::Synchronize(this->device_);
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
void StdGPUHashmap<Key, Hash>::Clear() {
    impl_.clear();
    buffer_accessor_.Reset(this->device_);
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Rehash(int64_t buckets) {
    int64_t iterator_count = Size();

    Tensor active_keys;
    std::vector<Tensor> active_values;

    if (iterator_count > 0) {
        Tensor active_addrs({iterator_count}, core::Int32, this->device_);
        GetActiveIndices(static_cast<addr_t*>(active_addrs.GetDataPtr()));

        Tensor active_indices = active_addrs.To(core::Int64);
        active_keys = this->buffer_->GetKeyBuffer().IndexGet({active_indices});
        auto value_buffers = this->GetValueBuffers();
        for (auto& value_buffer : value_buffers) {
            active_values.emplace_back(value_buffer.IndexGet({active_indices}));
        }
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(GetBucketCount());

    Free();
    int64_t new_capacity =
            int64_t(std::ceil(buckets * avg_capacity_per_bucket));
    Allocate(new_capacity);

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
__global__ void STDGPUInsertKernel(InternalStdGPUHashmap<Key, Hash> map,
                                   CUDAHashmapBufferAccessor buffer_accessor,
                                   const Key* input_keys,
                                   const void* const* input_values,
                                   addr_t* output_addrs,
                                   bool* output_masks,
                                   int64_t count,
                                   int64_t n_values) {
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
        auto key_ptr = buffer_accessor.GetKeyPtr(dst_kv_addr);

        // Copy templated key to buffer (duplicate)
        // TODO: hack stdgpu inside and take out the buffer directly
        *static_cast<Key*>(key_ptr) = key;

        // Copy/reset non-templated value in buffer
        for (int j = 0; j < n_values; ++j) {
            const int64_t dsize_value = buffer_accessor.dsize_values_[j];
            uint8_t* dst_value = static_cast<uint8_t*>(
                    buffer_accessor.GetValuePtr(dst_kv_addr, j));
            const uint8_t* src_value =
                    static_cast<const uint8_t*>(input_values[j]) +
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
void StdGPUHashmap<Key, Hash>::InsertImpl(
        const void* input_keys,
        std::vector<const void*> input_values_host,
        addr_t* output_addrs,
        bool* output_masks,
        int64_t count) {
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    thrust::device_vector<const void*> input_values(input_values_host.begin(),
                                                    input_values_host.end());
    int64_t n_values = input_values.size() == buffer_accessor_.n_values_
                               ? buffer_accessor_.n_values_
                               : 0;
    // std::cout << "n_values = " << n_values << "\n";
    // https://stackoverflow.com/a/37998941
    const void* const* input_values_ptr =
            thrust::raw_pointer_cast(input_values.data());

    STDGPUInsertKernel<<<blocks, threads, 0, core::cuda::GetStream()>>>(
            impl_, buffer_accessor_, static_cast<const Key*>(input_keys),
            input_values_ptr, output_addrs, output_masks, count, n_values);
    cuda::Synchronize(this->device_);
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Allocate(int64_t capacity) {
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

    // stdgpu initializes on the default stream. Set the current stream to
    // ensure correct behavior.
    {
        CUDAScopedStream scoped_stream(cuda::GetDefaultStream());

        impl_ = InternalStdGPUHashmap<Key, Hash>::createDeviceObject(
                this->capacity_,
                InternalStdGPUHashmapAllocator<Key>(this->device_));
        cuda::Synchronize(this->device_);
    }
}

template <typename Key, typename Hash>
void StdGPUHashmap<Key, Hash>::Free() {
    // Buffer is automatically handled by the smart pointer.

    buffer_accessor_.HostFree(this->device_);
    buffer_accessor_.Shutdown(this->device_);

    // stdgpu initializes on the default stream. Set the current stream to
    // ensure correct behavior.
    {
        CUDAScopedStream scoped_stream(cuda::GetDefaultStream());

        InternalStdGPUHashmap<Key, Hash>::destroyDeviceObject(impl_);
    }
}
}  // namespace core
}  // namespace open3d
