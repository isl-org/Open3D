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

#include <tbb/concurrent_unordered_map.h>

#include <limits>
#include <unordered_map>

#include "open3d/core/hashmap/CPU/CPUHashBackendBufferAccessor.hpp"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
template <typename Key, typename Hash>
class TBBHashBackend : public DeviceHashBackend {
public:
    TBBHashBackend(int64_t init_capacity,
                   int64_t key_dsize,
                   std::vector<int64_t> value_dsizes,
                   const Device& device);
    ~TBBHashBackend();

    void Rehash(int64_t buckets) override;

    void Insert(const void* input_keys,
                std::vector<const void*> input_values_soa,
                buf_index_t* output_buf_indices,
                bool* output_masks,
                int64_t count) override;

    void Activate(const void* input_keys,
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

    std::shared_ptr<tbb::concurrent_unordered_map<Key, buf_index_t, Hash>>
    GetImpl() const {
        return impl_;
    }

protected:
    std::shared_ptr<tbb::concurrent_unordered_map<Key, buf_index_t, Hash>>
            impl_;

    std::shared_ptr<CPUHashBackendBufferAccessor> buffer_accessor_;

    void InsertImpl(const void* input_keys,
                    std::vector<const void*> input_values_soa,
                    buf_index_t* output_buf_indices,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t capacity);
};

template <typename Key, typename Hash>
TBBHashBackend<Key, Hash>::TBBHashBackend(int64_t init_capacity,
                                          int64_t key_dsize,
                                          std::vector<int64_t> value_dsizes,
                                          const Device& device)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device) {
    Allocate(init_capacity);
}

template <typename Key, typename Hash>
TBBHashBackend<Key, Hash>::~TBBHashBackend() {}

template <typename Key, typename Hash>
int64_t TBBHashBackend<Key, Hash>::Size() const {
    return impl_->size();
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Insert(
        const void* input_keys,
        const std::vector<const void*> input_values_soa,
        buf_index_t* output_buf_indices,
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
    InsertImpl(input_keys, input_values_soa, output_buf_indices, output_masks,
               count);
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Activate(const void* input_keys,
                                         buf_index_t* output_buf_indices,
                                         bool* output_masks,
                                         int64_t count) {
    std::vector<const void*> null_values;
    Insert(input_keys, null_values, output_buf_indices, output_masks, count);
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Find(const void* input_keys,
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

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Erase(const void* input_keys,
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

template <typename Key, typename Hash>
int64_t TBBHashBackend<Key, Hash>::GetActiveIndices(
        buf_index_t* output_buf_indices) {
    int64_t count = impl_->size();
    int64_t i = 0;
    for (auto iter = impl_->begin(); iter != impl_->end(); ++iter, ++i) {
        output_buf_indices[i] = static_cast<int64_t>(iter->second);
    }

    return count;
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Clear() {
    impl_->clear();
    this->buffer_->ResetHeap();
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Rehash(int64_t buckets) {
    int64_t count = Size();

    Tensor active_keys;
    std::vector<Tensor> active_values;

    if (count > 0) {
        Tensor active_buf_indices({count}, core::Int32, this->device_);
        GetActiveIndices(
                static_cast<buf_index_t*>(active_buf_indices.GetDataPtr()));

        Tensor active_indices = active_buf_indices.To(core::Int64);
        active_keys = this->GetKeyBuffer().IndexGet({active_indices});

        auto value_buffers = this->GetValueBuffers();
        for (auto& value_buffer : value_buffers) {
            active_values.emplace_back(value_buffer.IndexGet({active_indices}));
        }
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(GetBucketCount());
    int64_t new_capacity =
            int64_t(std::ceil(buckets * avg_capacity_per_bucket));

    Allocate(new_capacity);

    if (count > 0) {
        Tensor output_buf_indices({count}, core::Int32, this->device_);
        Tensor output_masks({count}, core::Bool, this->device_);

        std::vector<const void*> active_value_ptrs;
        for (auto& active_value : active_values) {
            active_value_ptrs.push_back(active_value.GetDataPtr());
        }
        InsertImpl(active_keys.GetDataPtr(), active_value_ptrs,
                   static_cast<buf_index_t*>(output_buf_indices.GetDataPtr()),
                   output_masks.GetDataPtr<bool>(), count);
    }

    impl_->rehash(buckets);
}

template <typename Key, typename Hash>
int64_t TBBHashBackend<Key, Hash>::GetBucketCount() const {
    return impl_->unsafe_bucket_count();
}

template <typename Key, typename Hash>
std::vector<int64_t> TBBHashBackend<Key, Hash>::BucketSizes() const {
    int64_t bucket_count = impl_->unsafe_bucket_count();
    std::vector<int64_t> ret;
    for (int64_t i = 0; i < bucket_count; ++i) {
        ret.push_back(impl_->unsafe_bucket_size(i));
    }
    return ret;
}

template <typename Key, typename Hash>
float TBBHashBackend<Key, Hash>::LoadFactor() const {
    return impl_->load_factor();
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::InsertImpl(
        const void* input_keys,
        const std::vector<const void*> input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count) {
    const Key* input_keys_templated = static_cast<const Key*>(input_keys);

    size_t n_values = value_dsizes_.size();

    bool assign = (input_values_soa.size() == n_values);
    if (input_values_soa.size() != n_values && input_values_soa.size() != 0) {
        utility::LogWarning(
                "Input values mismatch with actual stored values, fall back to "
                "activate/reset instead of insertion.");
    }

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

                if (assign) {
                    const uint8_t* src_value =
                            static_cast<const uint8_t*>(input_values_soa[j]) +
                            this->value_dsizes_[j] * i;
                    std::memcpy(dst_value, src_value, this->value_dsizes_[j]);
                } else {
                    std::memset(dst_value, 0, this->value_dsizes_[j]);
                }
            }

            // Update from dummy 0
            res.first->second = buf_index;

            // Write to return variables
            output_buf_indices[i] = buf_index;
            output_masks[i] = true;
        }
    }
}

template <typename Key, typename Hash>
void TBBHashBackend<Key, Hash>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;

    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);

    buffer_accessor_ =
            std::make_shared<CPUHashBackendBufferAccessor>(*this->buffer_);

    impl_ = std::make_shared<
            tbb::concurrent_unordered_map<Key, buf_index_t, Hash>>(capacity,
                                                                   Hash());
}

}  // namespace core
}  // namespace open3d
