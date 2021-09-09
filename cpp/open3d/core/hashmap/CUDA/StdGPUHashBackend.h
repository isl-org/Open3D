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
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <stdgpu/unordered_map.cuh>
#include <type_traits>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/StdAllocator.h"
#include "open3d/core/hashmap/CUDA/CUDAHashBackendBufferAccessor.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"

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

// These typedefs must be defined outside of StdGPUHashBackend to make them
// accessible in raw CUDA kernels.
template <typename Key>
using InternalStdGPUHashBackendAllocator =
        StdGPUAllocator<thrust::pair<const Key, buf_index_t>>;

template <typename Key, typename Hash, typename Eq>
using InternalStdGPUHashBackend =
        stdgpu::unordered_map<Key,
                              buf_index_t,
                              Hash,
                              Eq,
                              InternalStdGPUHashBackendAllocator<Key>>;

template <typename Key, typename Hash, typename Eq>
class StdGPUHashBackend : public DeviceHashBackend {
public:
    StdGPUHashBackend(int64_t init_capacity,
                      int64_t key_dsize,
                      const std::vector<int64_t>& value_dsizes,
                      const Device& device);
    ~StdGPUHashBackend();

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

    InternalStdGPUHashBackend<Key, Hash, Eq> GetImpl() const { return impl_; }

    void Allocate(int64_t capacity);
    void Free();

protected:
    // Use reference, since the structure itself is implicitly handled as a
    // pointer directly by stdgpu.
    InternalStdGPUHashBackend<Key, Hash, Eq> impl_;

    CUDAHashBackendBufferAccessor buffer_accessor_;
};

template <typename Key, typename Hash, typename Eq>
StdGPUHashBackend<Key, Hash, Eq>::StdGPUHashBackend(
        int64_t init_capacity,
        int64_t key_dsize,
        const std::vector<int64_t>& value_dsizes,
        const Device& device)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device) {
    Allocate(init_capacity);
}

template <typename Key, typename Hash, typename Eq>
StdGPUHashBackend<Key, Hash, Eq>::~StdGPUHashBackend() {
    Free();
}

template <typename Key, typename Hash, typename Eq>
int64_t StdGPUHashBackend<Key, Hash, Eq>::Size() const {
    return impl_.size();
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash, typename Eq>
__global__ void STDGPUFindKernel(InternalStdGPUHashBackend<Key, Hash, Eq> map,
                                 CUDAHashBackendBufferAccessor buffer_accessor,
                                 const Key* input_keys,
                                 buf_index_t* output_buf_indices,
                                 bool* output_masks,
                                 int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    auto iter = map.find(key);
    bool flag = (iter != map.end());
    output_masks[tid] = flag;
    output_buf_indices[tid] = flag ? iter->second : 0;
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Find(const void* input_keys,
                                            buf_index_t* output_buf_indices,
                                            bool* output_masks,
                                            int64_t count) {
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    STDGPUFindKernel<<<blocks, threads, 0, core::cuda::GetStream()>>>(
            impl_, buffer_accessor_, static_cast<const Key*>(input_keys),
            output_buf_indices, output_masks, count);
    cuda::Synchronize(this->device_);
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash, typename Eq>
__global__ void STDGPUEraseKernel(InternalStdGPUHashBackend<Key, Hash, Eq> map,
                                  CUDAHashBackendBufferAccessor buffer_accessor,
                                  const Key* input_keys,
                                  buf_index_t* output_buf_indices,
                                  bool* output_masks,
                                  int64_t count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    auto iter = map.find(key);
    bool flag = (iter != map.end());
    output_masks[tid] = flag;
    output_buf_indices[tid] = flag ? iter->second : 0;

    if (output_masks[tid]) {
        output_masks[tid] = map.erase(key);
        if (output_masks[tid]) {
            buffer_accessor.DeviceFree(output_buf_indices[tid]);
        }
    }
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Erase(const void* input_keys,
                                             bool* output_masks,
                                             int64_t count) {
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    core::Tensor toutput_buf_indices =
            core::Tensor({count}, core::Int32, this->device_);
    buf_index_t* output_buf_indices =
            static_cast<buf_index_t*>(toutput_buf_indices.GetDataPtr());

    STDGPUEraseKernel<<<blocks, threads, 0, core::cuda::GetStream()>>>(
            impl_, buffer_accessor_, static_cast<const Key*>(input_keys),
            output_buf_indices, output_masks, count);
    cuda::Synchronize(this->device_);
}

template <typename Key>
struct ValueExtractor {
    OPEN3D_HOST_DEVICE buf_index_t
    operator()(const thrust::pair<Key, buf_index_t>& x) const {
        return x.second;
    }
};

template <typename Key, typename Hash, typename Eq>
int64_t StdGPUHashBackend<Key, Hash, Eq>::GetActiveIndices(
        buf_index_t* output_indices) {
    auto range = impl_.device_range();

    thrust::transform(range.begin(), range.end(), output_indices,
                      ValueExtractor<Key>());

    return impl_.size();
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Clear() {
    impl_.clear();
    this->buffer_->ResetHeap();
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Reserve(int64_t capacity) {}

template <typename Key, typename Hash, typename Eq>
int64_t StdGPUHashBackend<Key, Hash, Eq>::GetBucketCount() const {
    return impl_.bucket_count();
}

template <typename Key, typename Hash, typename Eq>
std::vector<int64_t> StdGPUHashBackend<Key, Hash, Eq>::BucketSizes() const {
    utility::LogError("Unimplemented");
}

template <typename Key, typename Hash, typename Eq>
float StdGPUHashBackend<Key, Hash, Eq>::LoadFactor() const {
    return impl_.load_factor();
}

// Need an explicit kernel for non-const access to map
template <typename Key, typename Hash, typename Eq, typename block_t>
__global__ void STDGPUInsertKernel(
        InternalStdGPUHashBackend<Key, Hash, Eq> map,
        CUDAHashBackendBufferAccessor buffer_accessor,
        const Key* input_keys,
        const void* const* input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count,
        int64_t n_values) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = input_keys[tid];
    output_buf_indices[tid] = 0;
    output_masks[tid] = false;

    // First apply 'try insert' with a dummy index
    auto res = map.emplace(key, 0);

    // If success, change the iterator and provide the actual index
    if (res.second) {
        buf_index_t buf_index = buffer_accessor.DeviceAllocate();
        auto key_ptr = buffer_accessor.GetKeyPtr(buf_index);

        // Copy templated key to buffer (duplicate)
        // TODO: hack stdgpu inside and take out the buffer directly
        *static_cast<Key*>(key_ptr) = key;

        // Copy/reset non-templated value in buffer
        for (int j = 0; j < n_values; ++j) {
            const int64_t blocks_per_element =
                    buffer_accessor.value_blocks_per_element_[j];

            block_t* dst_value = static_cast<block_t*>(
                    buffer_accessor.GetValuePtr(buf_index, j));
            const block_t* src_value =
                    static_cast<const block_t*>(input_values_soa[j]) +
                    blocks_per_element * tid;
            for (int b = 0; b < blocks_per_element; ++b) {
                dst_value[b] = src_value[b];
            }
        }

        // Update from the dummy index
        res.first->second = buf_index;

        // Write to return variables
        output_buf_indices[tid] = buf_index;
        output_masks[tid] = true;
    }
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Insert(
        const void* input_keys,
        const std::vector<const void*>& input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count) {
    uint32_t threads = 128;
    uint32_t blocks = (count + threads - 1) / threads;

    thrust::device_vector<const void*> input_values_soa_device(
            input_values_soa.begin(), input_values_soa.end());

    int64_t n_values = input_values_soa.size();
    const void* const* ptr_input_values_soa =
            thrust::raw_pointer_cast(input_values_soa_device.data());

    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T(
            buffer_accessor_.common_block_size_, [&]() {
                STDGPUInsertKernel<Key, Hash, Eq, block_t>
                        <<<blocks, threads, 0, core::cuda::GetStream()>>>(
                                impl_, buffer_accessor_,
                                static_cast<const Key*>(input_keys),
                                ptr_input_values_soa, output_buf_indices,
                                output_masks, count, n_values);
            });
    cuda::Synchronize(this->device_);
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;

    // Allocate buffer for key values.
    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);
    buffer_accessor_.Setup(*this->buffer_);

    // stdgpu initializes on the default stream. Set the current stream to
    // ensure correct behavior.
    {
        CUDAScopedStream scoped_stream(cuda::GetDefaultStream());

        impl_ = InternalStdGPUHashBackend<Key, Hash, Eq>::createDeviceObject(
                this->capacity_,
                InternalStdGPUHashBackendAllocator<Key>(this->device_));
        cuda::Synchronize(this->device_);
    }
}

template <typename Key, typename Hash, typename Eq>
void StdGPUHashBackend<Key, Hash, Eq>::Free() {
    // Buffer is automatically handled by the smart pointer.
    buffer_accessor_.Shutdown(this->device_);

    // stdgpu initializes on the default stream. Set the current stream to
    // ensure correct behavior.
    {
        CUDAScopedStream scoped_stream(cuda::GetDefaultStream());

        InternalStdGPUHashBackend<Key, Hash, Eq>::destroyDeviceObject(impl_);
    }
}
}  // namespace core
}  // namespace open3d
