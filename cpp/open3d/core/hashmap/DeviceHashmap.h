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

#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Traits.h"
namespace open3d {
namespace core {

struct DefaultHash {
    // Default constructor is required, since we need a struct instead of its
    // pointer as a member in a hash table for CUDA kernel launches.
    // Must set key_size_ before calling operator(), otherwise the behavior will
    // be undefined.
    DefaultHash() {}
    DefaultHash(int64_t key_size) : key_size_in_int_(key_size / sizeof(int)) {
        if (key_size_in_int_ == 0) {
            utility::LogError(
                    "[DefaultHash] Only support keys whose byte size is "
                    "multiples of sizeof(int)");
        }
    }

    uint64_t OPEN3D_HOST_DEVICE operator()(const void* key_ptr) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        auto cast_key_ptr = static_cast<const int*>(key_ptr);
        for (int64_t i = 0; i < key_size_in_int_; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    int64_t key_size_in_int_;
};

struct DefaultKeyEq {
    // Default constructor is required, since we need a struct instead of its
    // pointer as a member in a hash table for CUDA kernel launches.
    // Must set key_size_ before calling operator(), otherwise the behavior will
    // be undefined.
    DefaultKeyEq() {}
    DefaultKeyEq(int64_t key_size) : key_size_in_int_(key_size / sizeof(int)) {}

    bool OPEN3D_HOST_DEVICE operator()(const void* lhs, const void* rhs) const {
        if (lhs == nullptr || rhs == nullptr) {
            return false;
        }

        auto lhs_key_ptr = static_cast<const int*>(lhs);
        auto rhs_key_ptr = static_cast<const int*>(rhs);

        bool is_eq = true;
        for (int64_t i = 0; i < key_size_in_int_; ++i) {
            is_eq = is_eq && (lhs_key_ptr[i] == rhs_key_ptr[i]);
        }
        return is_eq;
    }

    int64_t key_size_in_int_;
};

/// Base class: shared interface
template <typename Hash, typename KeyEq>
class DeviceHashmap {
public:
    /// Comprehensive constructor for the developer.
    DeviceHashmap(int64_t init_buckets,
                  int64_t init_capacity,
                  int64_t dsize_key,
                  int64_t dsize_value,
                  const Device& device)
        : bucket_count_(init_buckets),
          capacity_(init_capacity),
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
                        iterator_t* output_iterators,
                        bool* output_masks,
                        int64_t count) = 0;

    /// Parallel activate contiguous arrays of keys without copying values.
    /// Specifically useful for large value elements (e.g., a tensor), where we
    /// can do in-place management after activation.
    virtual void Activate(const void* input_keys,
                          iterator_t* output_iterators,
                          bool* output_masks,
                          int64_t count) = 0;

    /// Parallel find a contiguous array of keys.
    virtual void Find(const void* input_keys,
                      iterator_t* output_iterators,
                      bool* output_masks,
                      int64_t count) = 0;

    /// Parallel erase a contiguous array of keys.
    virtual void Erase(const void* input_keys,
                       bool* output_masks,
                       int64_t count) = 0;

    /// Parallel collect all iterators in the hash table
    virtual int64_t GetIterators(iterator_t* output_iterators) = 0;

    /// Parallel unpack iterators to contiguous arrays of keys and/or values.
    virtual void UnpackIterators(const iterator_t* input_iterators,
                                 const bool* input_masks,
                                 void* output_keys,
                                 void* output_values,
                                 int64_t count) = 0;

    /// Parallel assign iterators in-place with associated values.
    virtual void AssignIterators(iterator_t* input_iterators,
                                 const bool* input_masks,
                                 const void* input_values,
                                 int64_t count) = 0;

    virtual int64_t Size() const = 0;

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    virtual std::vector<int64_t> BucketSizes() const = 0;

    /// Return size / bucket_count.
    virtual float LoadFactor() const = 0;

    virtual Tensor GetKeyBlobAsTensor(const SizeVector& shape, Dtype dtype) = 0;
    virtual Tensor GetValueBlobAsTensor(const SizeVector& shape,
                                        Dtype dtype) = 0;

    int64_t GetBucketCount() const { return bucket_count_; }
    int64_t GetCapacity() const { return capacity_; }
    int64_t GetKeyBytesize() const { return dsize_key_; }
    int64_t GetValueBytesize() const { return dsize_value_; }
    Device GetDevice() const { return device_; }

public:
    int64_t bucket_count_;
    int64_t capacity_;
    int64_t dsize_key_;
    int64_t dsize_value_;
    Device device_;

    float avg_capacity_bucket_ratio() {
        return float(capacity_) / float(bucket_count_);
    }
};

/// Factory functions:
/// - Default constructor switch is in DeviceHashmap.cpp
/// - Default CPU constructor is in CPU/DefaultHashmapCPU.cpp
/// - Default CUDA constructor is in CUDA/DefaultHashmapCUDA.cu

/// - Template constructor switch is in TemplateHashmap.h
/// - Template CPU constructor is in CPU/TemplateHashmapCPU.hpp
/// - Template CUDA constructor is in CUDA/TemplateHashmapCUDA.cuh
typedef DeviceHashmap<DefaultHash, DefaultKeyEq> DefaultDeviceHashmap;

std::shared_ptr<DefaultDeviceHashmap> CreateDefaultDeviceHashmap(
        int64_t init_buckets,
        int64_t init_capacity,
        int64_t dsize_key,
        int64_t dsize_value,
        const Device& device);

std::shared_ptr<DefaultDeviceHashmap> CreateDefaultCPUHashmap(
        int64_t init_buckets,
        int64_t init_capacity,
        int64_t dsize_key,
        int64_t dsize_value,
        const Device& device);

std::shared_ptr<DefaultDeviceHashmap> CreateDefaultCUDAHashmap(
        int64_t init_buckets,
        int64_t init_capacity,
        int64_t dsize_key,
        int64_t dsize_value,
        const Device& device);

}  // namespace core
}  // namespace open3d
