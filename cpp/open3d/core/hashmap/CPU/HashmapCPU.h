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

#include <tbb/concurrent_unordered_map.h>

#include <unordered_map>

#include "open3d/core/hashmap/CPU/KvPairsCPU.hpp"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Traits.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CPUHashmap : public DeviceHashmap<Hash, KeyEq> {
public:
    CPUHashmap(size_t init_buckets,
               size_t init_capacity,
               size_t dsize_key,
               size_t dsize_value,
               const Device& device);

    ~CPUHashmap();

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

protected:
    std::shared_ptr<tbb::concurrent_unordered_map<void*, addr_t, Hash, KeyEq>>
            impl_;
    std::shared_ptr<CPUKvPairs> kv_pairs_;

    void InsertImpl(const void* input_keys,
                    const void* input_values,
                    iterator_t* output_iterators,
                    bool* output_masks,
                    size_t count);
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
    impl_ = std::make_shared<
            tbb::concurrent_unordered_map<void*, addr_t, Hash, KeyEq>>(
            init_buckets, Hash(this->dsize_key_), KeyEq(this->dsize_key_));
    kv_pairs_ = std::make_shared<CPUKvPairs>(this->capacity_, this->dsize_key_,
                                             this->dsize_value_, this->device_);
}

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::~CPUHashmap() {
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
    size_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        size_t expected_buckets =
                std::max(this->bucket_count_ * 2,
                         size_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }
    InsertImpl(input_keys, input_values, output_iterators, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Activate(const void* input_keys,
                                       iterator_t* output_iterators,
                                       bool* output_masks,
                                       size_t count) {
    size_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        size_t expected_buckets =
                std::max(this->bucket_count_ * 2,
                         size_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }
    InsertImpl(input_keys, nullptr, output_iterators, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                   iterator_t* output_iterators,
                                   bool* output_masks,
                                   size_t count) {
    auto kv_pairs_ctx = kv_pairs_->GetContext();
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        auto iter = impl_->find(key);
        if (iter == impl_->end()) {
            output_iterators[i] = iterator_t();
            output_masks[i] = false;
        } else {
            output_iterators[i] = kv_pairs_ctx->extract_iterator(iter->second);
            output_masks[i] = true;
        }
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                    bool* output_masks,
                                    size_t count) {
    auto kv_pairs_ctx = kv_pairs_->GetContext();
    for (size_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        auto iter = impl_->find(key);
        if (iter == impl_->end()) {
            output_masks[i] = false;
        } else {
            kv_pairs_ctx->Free(iter->second);
            impl_->unsafe_erase(iter);
            output_masks[i] = true;
        }
    }
    this->bucket_count_ = impl_->unsafe_bucket_count();
}

template <typename Hash, typename KeyEq>
size_t CPUHashmap<Hash, KeyEq>::GetIterators(iterator_t* output_iterators) {
    auto kv_pairs_ctx = kv_pairs_->GetContext();

    size_t count = impl_->size();
    size_t i = 0;
    for (auto iter = impl_->begin(); iter != impl_->end(); ++iter, ++i) {
        output_iterators[i] = kv_pairs_ctx->extract_iterator(iter->second);
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
#pragma omp parallel for
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
#pragma omp parallel for
    for (size_t i = 0; i < iterator_count; ++i) {
        AssignIteratorsStep(input_iterators, input_masks, input_values,
                            this->device_, this->dsize_value_, i);
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Rehash(size_t buckets) {
    size_t iterator_count = Size();

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

    this->capacity_ = size_t(std::ceil(buckets * avg_capacity_per_bucket));
    impl_ = std::make_shared<
            tbb::concurrent_unordered_map<void*, addr_t, Hash, KeyEq>>(
            buckets, Hash(this->dsize_key_), KeyEq(this->dsize_key_));
    kv_pairs_ = std::make_shared<CPUKvPairs>(this->capacity_, this->dsize_key_,
                                             this->dsize_value_, this->device_);

    if (iterator_count > 0) {
        InsertImpl(output_keys, output_values, output_iterators, output_masks,
                   iterator_count);

        MemoryManager::Free(output_keys, this->device_);
        MemoryManager::Free(output_values, this->device_);
        MemoryManager::Free(output_masks, this->device_);
        MemoryManager::Free(output_iterators, this->device_);
    }

    impl_->rehash(buckets);
    this->bucket_count_ = impl_->unsafe_bucket_count();
}

template <typename Hash, typename KeyEq>
std::vector<size_t> CPUHashmap<Hash, KeyEq>::BucketSizes() const {
    size_t bucket_count = impl_->unsafe_bucket_count();
    std::vector<size_t> ret;
    for (size_t i = 0; i < bucket_count; ++i) {
        ret.push_back(impl_->unsafe_bucket_size(i));
    }
    return ret;
}

template <typename Hash, typename KeyEq>
float CPUHashmap<Hash, KeyEq>::LoadFactor() const {
    return impl_->load_factor();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::InsertImpl(const void* input_keys,
                                         const void* input_values,
                                         iterator_t* output_iterators,
                                         bool* output_masks,
                                         size_t count) {
    auto kv_pairs_ctx = kv_pairs_->GetContext();
    std::vector<addr_t> output_addrs(count);

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* src_key =
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i;

        addr_t dst_kv_addr = kv_pairs_ctx->Allocate();
        iterator_t dst_kv_iter = kv_pairs_ctx->extract_iterator(dst_kv_addr);

        uint8_t* dst_key = static_cast<uint8_t*>(dst_kv_iter.first);
        uint8_t* dst_value = static_cast<uint8_t*>(dst_kv_iter.second);
        std::memcpy(dst_key, src_key, this->dsize_key_);

        if (input_values != nullptr) {
            const uint8_t* src_value =
                    static_cast<const uint8_t*>(input_values) +
                    this->dsize_value_ * i;
            std::memcpy(dst_value, src_value, this->dsize_value_);
        } else {
            std::memset(dst_value, 0, this->dsize_value_);
        }

        // Try insertion.
        auto res = impl_->insert({dst_key, dst_kv_addr});

        output_masks[i] = res.second;
        output_addrs[i] = dst_kv_addr;
        output_iterators[i] = dst_kv_iter;
    }

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        if (!output_masks[i]) {
            kv_pairs_ctx->Free(output_addrs[i]);
            output_iterators[i] = iterator_t();
        }
    }

    this->bucket_count_ = impl_->unsafe_bucket_count();
}

}  // namespace core
}  // namespace open3d
