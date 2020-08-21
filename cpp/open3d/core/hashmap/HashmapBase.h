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
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
constexpr size_t kDefaultElemsPerBucket = 4;

struct DefaultHash {
    // Default constructor makes compiler happy. Undefined behavior, must set
    // key_size_ before calling operator().
    DefaultHash() {}
    DefaultHash(size_t key_size) : key_size_in_int_(key_size / sizeof(int)) {}

    uint64_t OPEN3D_HOST_DEVICE operator()(const void* key_ptr) const {
        uint64_t hash = UINT64_C(14695981039346656037);

        auto cast_key_ptr = static_cast<const int*>(key_ptr);
        for (size_t i = 0; i < key_size_in_int_; ++i) {
            hash ^= cast_key_ptr[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    size_t key_size_in_int_;
};

struct DefaultKeyEq {
    // Default constructor makes compiler happy. Undefined behavior, must set
    // key_size_ before calling operator().
    DefaultKeyEq() {}
    DefaultKeyEq(size_t key_size) : key_size_in_int_(key_size / sizeof(int)) {}

    bool OPEN3D_HOST_DEVICE operator()(const void* lhs, const void* rhs) const {
        if (lhs == nullptr || rhs == nullptr) {
            return false;
        }

        auto lhs_key_ptr = static_cast<const int*>(lhs);
        auto rhs_key_ptr = static_cast<const int*>(rhs);

        bool is_eq = true;
        for (size_t i = 0; i < key_size_in_int_; ++i) {
            is_eq = is_eq && (lhs_key_ptr[i] == rhs_key_ptr[i]);
        }
        return is_eq;
    }

    size_t key_size_in_int_;
};

/// Base class: shared interface
template <typename Hash = DefaultHash, typename KeyEq = DefaultKeyEq>
class Hashmap {
public:
    /// Comprehensive constructor for the developer.
    /// The developer knows all the parameter settings.
    Hashmap(size_t init_buckets,
            size_t init_capacity,
            size_t dsize_key,
            size_t dsize_value,
            Device device)
        : bucket_count_(init_buckets),
          capacity_(init_capacity),
          dsize_key_(dsize_key),
          dsize_value_(dsize_value),
          device_(device){};
    virtual ~Hashmap(){};

    /// Rehash expects extra memory space at runtime, since it consists of
    /// 1) dumping all key value pairs to a buffer
    /// 2) create a new hash table
    /// 3) parallel insert dumped key value pairs
    /// 4) deallocate old hash table
    virtual void Rehash(size_t buckets) = 0;

    /// Parallel insert contiguous arrays of keys and values.
    /// Output iterators and masks can be nullptrs if return iterators are not
    /// to be processed.
    virtual void Insert(const void* input_keys,
                        const void* input_values,
                        iterator_t* output_iterators,
                        bool* output_masks,
                        size_t count) = 0;

    /// Parallel find a contiguous array of keys.
    /// Output iterators and masks CANNOT be nullptrs as we have to interpret
    /// them.
    virtual void Find(const void* input_keys,
                      iterator_t* output_iterators,
                      bool* output_masks,
                      size_t count) = 0;

    /// Parallel erase a contiguous array of keys.
    /// Output masks can be a nullptr if return results are not to be processed.
    virtual void Erase(const void* input_keys,
                       bool* output_masks,
                       size_t count) = 0;

    /// Parallel collect all iterators in the hash table
    virtual size_t GetIterators(iterator_t* output_iterators) = 0;

    /// Parallel unpack iterators to contiguous arrays of keys and/or values.
    /// Output keys and values can be nullptrs if they are not to be
    /// processed/stored.
    virtual void UnpackIterators(const iterator_t* input_iterators,
                                 const bool* input_masks,
                                 void* output_keys,
                                 void* output_values,
                                 size_t count) = 0;

    /// Parallel assign iterators in-place with associated values.
    /// Note: users should manage the key-value correspondences around
    /// iterators.
    virtual void AssignIterators(iterator_t* input_iterators,
                                 const bool* input_masks,
                                 const void* input_values,
                                 size_t count) = 0;

    /// Return number of elems per bucket.
    /// High performance not required, so directly returns a vector.
    virtual std::vector<size_t> BucketSizes() = 0;

    /// Return size / bucket_count.
    virtual float LoadFactor() = 0;

public:
    uint32_t bucket_count_;
    uint32_t capacity_;
    uint32_t dsize_key_;
    uint32_t dsize_value_;

public:
    Device device_;
};

/// Low level factory for customized functions
/// User-friendly interface: just roughly estimate capacity, we handle
/// bucket_count.
template <typename Hash, typename KeyEq>
std::shared_ptr<Hashmap<Hash, KeyEq>> CreateHashmap(size_t init_capacity,
                                                    size_t dsize_key,
                                                    size_t dsize_value,
                                                    Device device);

/// Comprehensive interface
template <typename Hash, typename KeyEq>
std::shared_ptr<Hashmap<Hash, KeyEq>> CreateHashmap(size_t init_buckets,
                                                    size_t init_capacity,
                                                    size_t dsize_key,
                                                    size_t dsize_value,
                                                    Device device);

/// High level factory for default functions
/// Factory interface for non-templated Default hashmap -- to be instantiated in
/// implementations
typedef Hashmap<DefaultHash, DefaultKeyEq> DefaultHashmap;

/// User-friendly interface: just roughly estimate capacity, we handle
/// bucket_count.
std::shared_ptr<DefaultHashmap> CreateDefaultHashmap(size_t init_capacity,
                                                     size_t dsize_key,
                                                     size_t dsize_value,
                                                     Device device);

/// Comprehensive interface
std::shared_ptr<DefaultHashmap> CreateDefaultHashmap(size_t init_buckets,
                                                     size_t init_capacity,
                                                     size_t dsize_key,
                                                     size_t dsize_value,
                                                     Device device);

namespace _factory {
std::shared_ptr<DefaultHashmap> CreateDefaultCPUHashmap(size_t init_buckets,
                                                        size_t init_capacity,
                                                        size_t dsize_key,
                                                        size_t dsize_value,
                                                        Device device);

std::shared_ptr<DefaultHashmap> CreateDefaultCUDAHashmap(size_t init_buckets,
                                                         size_t init_capacity,
                                                         size_t dsize_key,
                                                         size_t dsize_value,
                                                         Device device);
}  // namespace _factory
}  // namespace core
}  // namespace open3d
