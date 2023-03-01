// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <limits>
#include <unordered_map>

#include "open3d/core/hashmap/CPU/CPUHashBackendBufferAccessor.hpp"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
template <typename Key, typename Hash, typename Eq>
class TBBHashBackend : public DeviceHashBackend {
public:
    TBBHashBackend(int64_t init_capacity,
                   int64_t key_dsize,
                   const std::vector<int64_t>& value_dsizes,
                   const Device& device);
    ~TBBHashBackend();

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

    std::shared_ptr<tbb::concurrent_unordered_map<Key, buf_index_t, Hash, Eq>>
    GetImpl() const {
        return impl_;
    }

    void Allocate(int64_t capacity) override;
    void Free() override{};

protected:
    std::shared_ptr<tbb::concurrent_unordered_map<Key, buf_index_t, Hash, Eq>>
            impl_;

    std::shared_ptr<CPUHashBackendBufferAccessor> buffer_accessor_;
};

template <typename Key, typename Hash, typename Eq>
TBBHashBackend<Key, Hash, Eq>::TBBHashBackend(
        int64_t init_capacity,
        int64_t key_dsize,
        const std::vector<int64_t>& value_dsizes,
        const Device& device)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device) {
    Allocate(init_capacity);
}

template <typename Key, typename Hash, typename Eq>
TBBHashBackend<Key, Hash, Eq>::~TBBHashBackend() {}

template <typename Key, typename Hash, typename Eq>
int64_t TBBHashBackend<Key, Hash, Eq>::Size() const {
    return impl_->size();
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Find(const void* input_keys,
                                         buf_index_t* output_buf_indices,
                                         bool* output_masks,
                                         int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);

#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < count; ++i) {
        const Key& key = input_keys_templated[i];

        auto iter = impl_->find(key);
        bool flag = (iter != impl_->end());
        output_masks[i] = flag;
        output_buf_indices[i] = flag ? iter->second : 0;
    }
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Erase(const void* input_keys,
                                          bool* output_masks,
                                          int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);

    for (int64_t i = 0; i < count; ++i) {
        const Key& key = input_keys_templated[i];

        auto iter = impl_->find(key);
        bool flag = (iter != impl_->end());
        output_masks[i] = flag;
        if (flag) {
            buffer_accessor_->DeviceFree(iter->second);
            impl_->unsafe_erase(iter);
        }
    }
}

template <typename Key, typename Hash, typename Eq>
int64_t TBBHashBackend<Key, Hash, Eq>::GetActiveIndices(
        buf_index_t* output_buf_indices) {
    int64_t count = impl_->size();
    int64_t i = 0;
    for (auto iter = impl_->begin(); iter != impl_->end(); ++iter, ++i) {
        output_buf_indices[i] = static_cast<int64_t>(iter->second);
    }

    return count;
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Clear() {
    impl_->clear();
    this->buffer_->ResetHeap();
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Reserve(int64_t capacity) {
    impl_->rehash(std::ceil(capacity / impl_->max_load_factor()));
}

template <typename Key, typename Hash, typename Eq>
int64_t TBBHashBackend<Key, Hash, Eq>::GetBucketCount() const {
    return impl_->unsafe_bucket_count();
}

template <typename Key, typename Hash, typename Eq>
std::vector<int64_t> TBBHashBackend<Key, Hash, Eq>::BucketSizes() const {
    int64_t bucket_count = impl_->unsafe_bucket_count();
    std::vector<int64_t> ret;
    for (int64_t i = 0; i < bucket_count; ++i) {
        ret.push_back(impl_->unsafe_bucket_size(i));
    }
    return ret;
}

template <typename Key, typename Hash, typename Eq>
float TBBHashBackend<Key, Hash, Eq>::LoadFactor() const {
    return impl_->load_factor();
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Insert(
        const void* input_keys,
        const std::vector<const void*>& input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);

    size_t n_values = input_values_soa.size();

#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < count; ++i) {
        output_buf_indices[i] = 0;
        output_masks[i] = false;

        const Key& key = input_keys_templated[i];

        // Try to insert a dummy buffer index.
        auto res = impl_->insert({key, 0});

        // Lazy copy key value pair to buffer only if succeeded
        if (res.second) {
            buf_index_t buf_index = buffer_accessor_->DeviceAllocate();
            void* key_ptr = buffer_accessor_->GetKeyPtr(buf_index);

            // Copy templated key to buffer
            *static_cast<Key*>(key_ptr) = key;

            // Copy/reset non-templated value in buffer
            for (size_t j = 0; j < n_values; ++j) {
                uint8_t* dst_value = static_cast<uint8_t*>(
                        buffer_accessor_->GetValuePtr(buf_index, j));

                const uint8_t* src_value =
                        static_cast<const uint8_t*>(input_values_soa[j]) +
                        this->value_dsizes_[j] * i;
                std::memcpy(dst_value, src_value, this->value_dsizes_[j]);
            }

            // Update from dummy 0
            res.first->second = buf_index;

            // Write to return variables
            output_buf_indices[i] = buf_index;
            output_masks[i] = true;
        }
    }
}

template <typename Key, typename Hash, typename Eq>
void TBBHashBackend<Key, Hash, Eq>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;

    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);

    buffer_accessor_ =
            std::make_shared<CPUHashBackendBufferAccessor>(*this->buffer_);

    impl_ = std::make_shared<
            tbb::concurrent_unordered_map<Key, buf_index_t, Hash, Eq>>(
            capacity, Hash(), Eq());
}

}  // namespace core
}  // namespace open3d
