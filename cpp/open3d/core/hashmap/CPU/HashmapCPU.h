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

#include "open3d/core/hashmap/CPU/HashmapBufferCPU.hpp"
#include "open3d/core/hashmap/DeviceHashmap.h"

namespace open3d {
namespace core {
template <typename Hash, typename KeyEq>
class CPUHashmap : public DeviceHashmap<Hash, KeyEq> {
public:
    CPUHashmap(int64_t init_buckets,
               int64_t init_capacity,
               int64_t dsize_key,
               int64_t dsize_value,
               const Device& device);

    ~CPUHashmap();

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

    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

protected:
    std::shared_ptr<tbb::concurrent_unordered_map<void*, addr_t, Hash, KeyEq>>
            impl_;

    std::shared_ptr<CPUHashmapBufferContext> buffer_ctx_;

    void InsertImpl(const void* input_keys,
                    const void* input_values,
                    addr_t* output_addrs,
                    bool* output_masks,
                    int64_t count);

    void Allocate(int64_t capacity, int64_t buckets);
};

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::CPUHashmap(int64_t init_buckets,
                                    int64_t init_capacity,
                                    int64_t dsize_key,
                                    int64_t dsize_value,
                                    const Device& device)
    : DeviceHashmap<Hash, KeyEq>(
              init_buckets,
              init_capacity,  /// Dummy for std unordered_map, reserved for.
                              /// other hashmaps.
              dsize_key,
              dsize_value,
              device) {
    Allocate(init_capacity, init_buckets);
}

template <typename Hash, typename KeyEq>
CPUHashmap<Hash, KeyEq>::~CPUHashmap() {}

template <typename Hash, typename KeyEq>
int64_t CPUHashmap<Hash, KeyEq>::Size() const {
    return impl_->size();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Insert(const void* input_keys,
                                     const void* input_values,
                                     addr_t* output_addrs,
                                     bool* output_masks,
                                     int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        int64_t expected_buckets = std::max(
                this->bucket_count_ * 2,
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }
    InsertImpl(input_keys, input_values, output_addrs, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Activate(const void* input_keys,
                                       addr_t* output_addrs,
                                       bool* output_masks,
                                       int64_t count) {
    int64_t new_size = Size() + count;
    if (new_size > this->capacity_) {
        float avg_capacity_per_bucket =
                float(this->capacity_) / float(this->bucket_count_);
        int64_t expected_buckets = std::max(
                this->bucket_count_ * 2,
                int64_t(std::ceil(new_size / avg_capacity_per_bucket)));
        Rehash(expected_buckets);
    }
    InsertImpl(input_keys, nullptr, output_addrs, output_masks, count);
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Find(const void* input_keys,
                                   addr_t* output_addrs,
                                   bool* output_masks,
                                   int64_t count) {
#pragma omp parallel for
    for (int64_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        auto iter = impl_->find(key);
        bool flag = (iter != impl_->end());
        output_masks[i] = flag;
        output_addrs[i] = flag ? iter->second : 0;
    }
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Erase(const void* input_keys,
                                    bool* output_masks,
                                    int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        uint8_t* key = const_cast<uint8_t*>(
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i);

        auto iter = impl_->find(key);
        bool flag = (iter != impl_->end());
        output_masks[i] = flag;
        if (flag) {
            buffer_ctx_->DeviceFree(iter->second);
            impl_->unsafe_erase(iter);
        }
    }
    this->bucket_count_ = impl_->unsafe_bucket_count();
}

template <typename Hash, typename KeyEq>
int64_t CPUHashmap<Hash, KeyEq>::GetActiveIndices(addr_t* output_indices) {
    int64_t count = impl_->size();
    int64_t i = 0;
    for (auto iter = impl_->begin(); iter != impl_->end(); ++iter, ++i) {
        output_indices[i] = static_cast<int64_t>(iter->second);
    }

    return count;
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Rehash(int64_t buckets) {
    int64_t iterator_count = Size();

    Tensor active_keys;
    Tensor active_values;

    if (iterator_count > 0) {
        Tensor active_addrs({iterator_count}, Dtype::Int32, this->device_);
        GetActiveIndices(active_addrs.GetDataPtr<addr_t>());

        Tensor active_indices = active_addrs.To(Dtype::Int64);
        active_keys = this->GetKeyBuffer().IndexGet({active_indices});
        active_values = this->GetValueBuffer().IndexGet({active_indices});
    }

    float avg_capacity_per_bucket =
            float(this->capacity_) / float(this->bucket_count_);

    int64_t new_capacity =
            int64_t(std::ceil(buckets * avg_capacity_per_bucket));
    Allocate(new_capacity, buckets);

    if (iterator_count > 0) {
        Tensor output_addrs({iterator_count}, Dtype::Int32, this->device_);
        Tensor output_masks({iterator_count}, Dtype::Bool, this->device_);

        InsertImpl(active_keys.GetDataPtr(), active_values.GetDataPtr(),
                   output_addrs.GetDataPtr<addr_t>(),
                   output_masks.GetDataPtr<bool>(), iterator_count);
    }

    impl_->rehash(buckets);
    this->bucket_count_ = impl_->unsafe_bucket_count();
}

template <typename Hash, typename KeyEq>
std::vector<int64_t> CPUHashmap<Hash, KeyEq>::BucketSizes() const {
    int64_t bucket_count = impl_->unsafe_bucket_count();
    std::vector<int64_t> ret;
    for (int64_t i = 0; i < bucket_count; ++i) {
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
                                         addr_t* output_addrs,
                                         bool* output_masks,
                                         int64_t count) {
#pragma omp parallel for
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t* src_key =
                static_cast<const uint8_t*>(input_keys) + this->dsize_key_ * i;

        addr_t dst_kv_addr = buffer_ctx_->DeviceAllocate();
        auto dst_kv_iter = buffer_ctx_->ExtractIterator(dst_kv_addr);

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

        output_addrs[i] = dst_kv_addr;
        output_masks[i] = res.second;
    }

#pragma omp parallel for
    for (int64_t i = 0; i < count; ++i) {
        if (!output_masks[i]) {
            buffer_ctx_->DeviceFree(output_addrs[i]);
        }
    }

    this->bucket_count_ = impl_->unsafe_bucket_count();
}

template <typename Hash, typename KeyEq>
void CPUHashmap<Hash, KeyEq>::Allocate(int64_t capacity, int64_t buckets) {
    this->capacity_ = capacity;

    this->buffer_ =
            std::make_shared<HashmapBuffer>(this->capacity_, this->dsize_key_,
                                            this->dsize_value_, this->device_);

    buffer_ctx_ = std::make_shared<CPUHashmapBufferContext>(
            this->capacity_, this->dsize_key_, this->dsize_value_,
            this->buffer_->GetKeyBuffer(), this->buffer_->GetValueBuffer(),
            this->buffer_->GetHeap());
    buffer_ctx_->Reset();

    impl_ = std::make_shared<
            tbb::concurrent_unordered_map<void*, addr_t, Hash, KeyEq>>(
            buckets, Hash(this->dsize_key_), KeyEq(this->dsize_key_));
}

}  // namespace core
}  // namespace open3d
