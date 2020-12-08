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

#include <unordered_map>

#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CPUHashmap : public DeviceHashmap<Hash, KeyEq> {
public:
    ~CPUHashmap();

    CPUHashmap(size_t init_buckets,
               size_t init_capacity,
               size_t dsize_key,
               size_t dsize_value,
               const Device& device);

    void Rehash(size_t buckets) override;

    void Insert(const void* input_keys,
                const void* input_values,
                iterator_t* output_iterators,
                bool* output_masks,
                size_t count) override;

    void Activate(const void* input_keys,
                  iterator_t* output_iterators,
                  bool* output_masks,
                  size_t count) override;

    void Find(const void* input_keys,
              iterator_t* output_iterators,
              bool* output_masks,
              size_t count) override;

    void Erase(const void* input_keys,
               bool* output_masks,
               size_t count) override;

    size_t GetIterators(iterator_t* output_iterators) override;

    void UnpackIterators(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         size_t count) override;

    void AssignIterators(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         size_t count) override;

    std::vector<size_t> BucketSizes() const override;
    float LoadFactor() const override;

    size_t Size() const override;

private:
    std::shared_ptr<std::unordered_map<void*, void*, Hash, KeyEq>> impl_;

    // Valid kv_pairs.
    std::vector<iterator_t> kv_pairs_;
};

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::CPUHashmap(size_t init_buckets,
                                    size_t init_capacity,
                                    size_t dsize_key,
                                    size_t dsize_value,
                                    const Device& device)
    : DeviceHashmap<Hash, KeyEq>(
              init_buckets,
              init_capacity,  /// Dummy for std unordered_map, reserved for.
                              /// other hashmaps.
              dsize_key,
              dsize_value,
              device) {
    impl_ = std::make_shared<std::unordered_map<void*, void*, Hash, KeyEq>>(
            init_buckets, Hash(dsize_key), KeyEq(dsize_key));
}

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::~CPUHashmap() {
    for (auto kv_pair : kv_pairs_) {
        MemoryManager::Free(kv_pair.first, this->device_);
        MemoryManager::Free(kv_pair.second, this->device_);
    }
    impl_->clear();
}

template <typename Hash, typename KeyEq>
size_t CPUHashmap<Hash, KeyEq>::Size() const {
    return impl_->size();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                     const void* input_values,
                                     iterator_t* output_iterators,
                                     bool* output_masks,
                                     size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* src_key =
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i;
        const uint8_t* src_value = static_cast<const uint8_t*>(input_values) +
                                   this->dsize_value_ * i;

        // Manually copy before insert.
        uint8_t* dst_key = static_cast<uint8_t*>(
                MemoryManager::Malloc(this->dsize_key_, this->device_));
        uint8_t* dst_value = static_cast<uint8_t*>(
                MemoryManager::Malloc(this->dsize_value_, this->device_));

        MemoryManager::Memcpy(dst_key, this->device_, src_key, this->device_,
                              this->dsize_key_);
        MemoryManager::Memcpy(dst_value, this->device_, src_value,
                              this->device_, this->dsize_value_);

        // Try insertion.
        auto res = impl_->insert({dst_key, dst_value});

        // Handle memory.
        if (res.second) {
            output_iterators[i] = iterator_t(dst_key, dst_value);
            output_masks[i] = true;
        } else {
            MemoryManager::Free(dst_key, this->device_);
            MemoryManager::Free(dst_value, this->device_);
            output_iterators[i] = iterator_t();
            output_masks[i] = false;
        }
    }
    this->capacity_ = impl_->size();
    this->bucket_count_ = impl_->bucket_count();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Activate(const void* input_keys,
                                       iterator_t* output_iterators,
                                       bool* output_masks,
                                       size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* src_key =
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i;

        // Manually copy before insert
        uint8_t* dst_key = static_cast<uint8_t*>(
                MemoryManager::Malloc(this->dsize_key_, this->device_));
        uint8_t* dummy_value = static_cast<uint8_t*>(
                MemoryManager::Malloc(this->dsize_value_, this->device_));
        memset(dummy_value, 0, this->dsize_value_);

        MemoryManager::Memcpy(dst_key, this->device_, src_key, this->device_,
                              this->dsize_key_);

        // Try insertion.
        auto res = impl_->insert({dst_key, dummy_value});

        // Handle memory.
        if (res.second) {
            output_iterators[i] = iterator_t(dst_key, dummy_value);
            output_masks[i] = true;
        } else {
            MemoryManager::Free(dst_key, this->device_);
            MemoryManager::Free(dummy_value, this->device_);
            output_iterators[i] = iterator_t();
            output_masks[i] = false;
        }
    }
    this->capacity_ = impl_->size();
    this->bucket_count_ = impl_->bucket_count();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                   iterator_t* output_iterators,
                                   bool* output_masks,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        auto iter = impl_->find(key);
        if (iter == impl_->end()) {
            output_iterators[i] = iterator_t();
            output_masks[i] = false;
        } else {
            output_iterators[i] = iterator_t(iter->first, iter->second);
            output_masks[i] = true;
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                    bool* output_masks,
                                    size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        size_t erased = impl_->erase(key);
        output_masks[i] = erased > 0;
    }
}

template <typename Hash, typename KeyEq>
size_t CPUHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    size_t count = impl_->size();

    size_t i = 0;
    for (auto iter = impl_->begin(); iter != impl_->end(); ++iter, ++i) {
        output_iterators[i] = iterator_t(iter->first, iter->second);
    }

    return count;
}

void UnpackIteratorsStep(const iterator_t* input_iterators,
                         const bool* input_masks,
                         void* output_keys,
                         void* output_values,
                         const Device& device,
                         size_t dsize_key,
                         size_t dsize_value,
                         size_t tid) {
    // Valid queries.
    if (input_masks == nullptr || input_masks[tid]) {
        if (output_keys != nullptr) {
            uint8_t* dst_key_ptr =
                    static_cast<uint8_t*>(output_keys) + dsize_key * tid;
            uint8_t* src_key_ptr =
                    static_cast<uint8_t*>(input_iterators[tid].first);
            MemoryManager::Memcpy(dst_key_ptr, device, src_key_ptr, device,
                                  dsize_key);
        }

        if (output_values != nullptr) {
            uint8_t* dst_value_ptr =
                    static_cast<uint8_t*>(output_values) + dsize_value * tid;
            uint8_t* src_value_ptr =
                    static_cast<uint8_t*>(input_iterators[tid].second);
            MemoryManager::Memcpy(dst_value_ptr, device, src_value_ptr, device,
                                  dsize_value);
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::UnpackIterators(const iterator_t* input_iterators,
                                              const bool* input_masks,
                                              void* output_keys,
                                              void* output_values,
                                              size_t iterator_count) {
    for (size_t i = 0; i < iterator_count; ++i) {
        UnpackIteratorsStep(input_iterators, input_masks, output_keys,
                            output_values, this->device_, this->dsize_key_,
                            this->dsize_value_, i);
    }
}

void AssignIteratorsStep(iterator_t* input_iterators,
                         const bool* input_masks,
                         const void* input_values,
                         const Device& device,
                         size_t dsize_value,
                         size_t tid) {
    // Valid queries.
    if (input_masks == nullptr || input_masks[tid]) {
        const uint8_t* src_value_ptr =
                static_cast<const uint8_t*>(input_values) + dsize_value * tid;
        uint8_t* dst_value_ptr =
                static_cast<uint8_t*>(input_iterators[tid].second);
        MemoryManager::Memcpy(dst_value_ptr, device, src_value_ptr, device,
                              dsize_value);
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::AssignIterators(iterator_t* input_iterators,
                                              const bool* input_masks,
                                              const void* input_values,
                                              size_t iterator_count) {
    for (size_t i = 0; i < iterator_count; ++i) {
        AssignIteratorsStep(input_iterators, input_masks, input_values,
                            this->device_, this->dsize_value_, i);
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Rehash(size_t buckets) {
    impl_->rehash(buckets);
}

template <typename Hash, typename KeyEq>
std::vector<size_t> CPUHashmap<Hash, KeyEq>::BucketSizes() const {
    size_t bucket_count = impl_->bucket_count();
    std::vector<size_t> ret;
    for (size_t i = 0; i < bucket_count; ++i) {
        ret.push_back(impl_->bucket_size(i));
    }
    return ret;
}

template <typename Hash, typename KeyEq>
float CPUHashmap<Hash, KeyEq>::LoadFactor() const {
    return impl_->load_factor();
}

}  // namespace core
}  // namespace open3d
