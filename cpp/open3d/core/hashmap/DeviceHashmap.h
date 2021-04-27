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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashmapBuffer.h"

namespace open3d {
namespace core {

enum class HashmapBackend;

class DeviceHashmap {
public:
    /// Comprehensive constructor for the developer.
    DeviceHashmap(int64_t init_capacity,
                  int64_t dsize_key,
                  int64_t dsize_value,
                  const Device& device)
        : capacity_(init_capacity),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device) {}
    virtual ~DeviceHashmap() {}

    /// Rehash expects a lot of extra memory space at runtime,
    /// since it consists of
    /// 1) dumping all key value pairs to a buffer
    /// 2) creating a new hash table
    /// 3) parallel inserting dumped key value pairs
    /// 4) deallocating old hash table
    virtual void Rehash(int64_t buckets) = 0;

    /// Parallel insert contiguous arrays of keys and values.
    virtual void Insert(const void* input_keys,
                        const void* input_values,
                        addr_t* output_iterators,
                        bool* output_masks,
                        int64_t count) = 0;

    /// Parallel activate contiguous arrays of keys without copying values.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    virtual void Activate(const void* input_keys,
                          addr_t* output_iterators,
                          bool* output_masks,
                          int64_t count) = 0;

    /// Parallel find a contiguous array of keys.
    virtual void Find(const void* input_keys,
                      addr_t* output_iterators,
                      bool* output_masks,
                      int64_t count) = 0;

    /// Parallel erase a contiguous array of keys.
    virtual void Erase(const void* input_keys,
                       bool* output_masks,
                       int64_t count) = 0;

    /// Parallel collect all iterators in the hash table
    virtual int64_t GetActiveIndices(addr_t* output_indices) = 0;

    /// Clear stored map without reallocating memory.
    virtual void Clear() = 0;

    virtual int64_t Size() const = 0;
    virtual int64_t GetBucketCount() const = 0;
    virtual float LoadFactor() const = 0;

    int64_t GetCapacity() const { return capacity_; }
    int64_t GetKeyBytesize() const { return dsize_key_; }
    int64_t GetValueBytesize() const { return dsize_value_; }
    Device GetDevice() const { return device_; }

    Tensor& GetKeyBuffer() { return buffer_->GetKeyBuffer(); }
    Tensor& GetValueBuffer() { return buffer_->GetValueBuffer(); }

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    virtual std::vector<int64_t> BucketSizes() const = 0;

public:
    int64_t capacity_;
    int64_t dsize_key_;
    int64_t dsize_value_;

    Device device_;

    std::shared_ptr<HashmapBuffer> buffer_;
};

/// Factory functions:
/// - Default constructor switch is in DeviceHashmap.cpp
/// - Default CPU constructor is in CPU/CreateCPUHashmap.cpp
/// - Default CUDA constructor is in CUDA/CreateCUDAHashmap.cu
std::shared_ptr<DeviceHashmap> CreateDeviceHashmap(
        int64_t init_capacity,
        const Dtype& dtype_key,
        const Dtype& dtype_value,
        const SizeVector& element_shape_key,
        const SizeVector& element_shape_value,
        const Device& device,
        const HashmapBackend& backend);

std::shared_ptr<DeviceHashmap> CreateCPUHashmap(
        int64_t init_capacity,
        const Dtype& dtype_key,
        const Dtype& dtype_value,
        const SizeVector& element_shape_key,
        const SizeVector& element_shape_value,
        const Device& device,
        const HashmapBackend& backend);

std::shared_ptr<DeviceHashmap> CreateCUDAHashmap(
        int64_t init_capacity,
        const Dtype& dtype_key,
        const Dtype& dtype_value,
        const SizeVector& element_shape_key,
        const SizeVector& element_shape_value,
        const Device& device,
        const HashmapBackend& backend);

}  // namespace core
}  // namespace open3d
