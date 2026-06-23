// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/SYCL/SYCLHashBackendBufferAccessor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

// Per-slot states for the open-addressing table. EMPTY must be 0 so that a
// zero-initialized (memset) state array starts fully empty.
//
//   EMPTY    -> never used (probe stops here on find)
//   LOCKED   -> a writer is claiming the slot and writing its key/value
//   OCCUPIED -> slot holds a fully published key/value
//   DELETED  -> tombstone; probe continues, slot is reusable by insert
enum SYCLHashSlotState : int32_t {
    kSYCLSlotEmpty = 0,
    kSYCLSlotLocked = 1,
    kSYCLSlotOccupied = 2,
    kSYCLSlotDeleted = 3,
};

/// A native SYCL device hash backend implementing DeviceHashBackend.
///
/// Design: a lock-free open-addressing hash table with linear probing and
/// tombstone deletion. The table has bucket_count = 2 * capacity slots so it is
/// never more than half full (at most `capacity` live keys), guaranteeing that
/// probing always terminates. The table stores only a per-slot state and a
/// buffer index; actual keys/values live in the shared HashBackendBuffer and
/// are compared by dereferencing the buffer via the buffer index.
///
/// Concurrency: slot ownership is claimed with a single compare-exchange on the
/// state word (EMPTY/DELETED -> LOCKED). The key/value and buffer index are
/// written before the slot is published (LOCKED -> OCCUPIED) with release
/// ordering, so a concurrent probe that observes OCCUPIED (with acquire
/// ordering) sees a consistent entry. A probe that observes LOCKED waits until
/// the slot is published; this is required for correct de-duplication of
/// identical keys inserted concurrently and assumes the device provides
/// independent forward progress (true for current Intel GPUs under SYCL 2020).
template <typename Key, typename Hash, typename Eq>
class SYCLHashBackend : public DeviceHashBackend {
public:
    SYCLHashBackend(int64_t init_capacity,
                    int64_t key_dsize,
                    const std::vector<int64_t>& value_dsizes,
                    const Device& device);
    ~SYCLHashBackend();

    void Reserve(int64_t capacity) override {}

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

    void Allocate(int64_t capacity) override;
    void Free() override;

protected:
    SYCLHashBackendBufferAccessor buffer_accessor_;

    // Open-addressing table arrays (USM device memory).
    int32_t* slot_state_ = nullptr;       /* [bucket_count_] */
    buf_index_t* slot_buf_index_ = nullptr; /* [bucket_count_] */
    int64_t bucket_count_ = 0;

    sycl::queue queue_;
};

template <typename Key, typename Hash, typename Eq>
SYCLHashBackend<Key, Hash, Eq>::SYCLHashBackend(
        int64_t init_capacity,
        int64_t key_dsize,
        const std::vector<int64_t>& value_dsizes,
        const Device& device)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device),
      queue_(sy::SYCLContext::GetInstance().GetDefaultQueue(device)) {
    Allocate(init_capacity);
}

template <typename Key, typename Hash, typename Eq>
SYCLHashBackend<Key, Hash, Eq>::~SYCLHashBackend() {
    Free();
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::Size() const {
    return this->buffer_->GetHeapTopIndex();
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::GetBucketCount() const {
    return bucket_count_;
}

template <typename Key, typename Hash, typename Eq>
std::vector<int64_t> SYCLHashBackend<Key, Hash, Eq>::BucketSizes() const {
    utility::LogError("Unimplemented");
}

template <typename Key, typename Hash, typename Eq>
float SYCLHashBackend<Key, Hash, Eq>::LoadFactor() const {
    return float(Size()) / float(bucket_count_);
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Insert(
        const void* input_keys,
        const std::vector<const void*>& input_values_soa,
        buf_index_t* output_buf_indices,
        bool* output_masks,
        int64_t count) {
    if (count == 0) return;

    const Key* keys = static_cast<const Key*>(input_keys);
    const int n_values = static_cast<int>(input_values_soa.size());

    // Stage the SoA value base pointers in device memory.
    const void** d_values_soa = static_cast<const void**>(
            MemoryManager::Malloc(n_values * sizeof(void*), this->device_));
    MemoryManager::MemcpyFromHost(d_values_soa, this->device_,
                                  input_values_soa.data(),
                                  n_values * sizeof(void*));

    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    int32_t* slot_state = slot_state_;
    buf_index_t* slot_buf_index = slot_buf_index_;
    const int64_t bucket_count = bucket_count_;
    Hash hash_fn;
    Eq eq_fn;

    queue_.parallel_for(count, [=](int64_t tid) {
             const Key key = keys[tid];
             output_buf_indices[tid] = 0;
             output_masks[tid] = false;

             const int64_t home =
                     static_cast<int64_t>(hash_fn(key) %
                                          static_cast<uint64_t>(bucket_count));

             while (true) {  // Restart on lost claim race.
                 int64_t first_deleted = -1;
                 bool finished = false;
                 bool restart = false;

                 for (int64_t probe = 0; probe < bucket_count; ++probe) {
                     const int64_t idx = (home + probe) % bucket_count;
                     sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>
                             st(slot_state[idx]);

                     int32_t s = st.load(sycl::memory_order::acquire);
                     // Wait out an in-progress writer so duplicates of the same
                     // key are detected rather than inserted twice.
                     while (s == kSYCLSlotLocked) {
                         s = st.load(sycl::memory_order::acquire);
                     }

                     if (s == kSYCLSlotOccupied) {
                         const buf_index_t bi = slot_buf_index[idx];
                         const Key* slot_key =
                                 static_cast<const Key*>(accessor.GetKeyPtr(bi));
                         if (eq_fn(*slot_key, key)) {
                             // Key already present: report existing slot.
                             output_buf_indices[tid] = bi;
                             output_masks[tid] = false;
                             finished = true;
                             break;
                         }
                         continue;
                     }

                     if (s == kSYCLSlotDeleted) {
                         if (first_deleted < 0) first_deleted = idx;
                         continue;
                     }

                     // s == kSYCLSlotEmpty: no existing key found. Claim the
                     // first tombstone seen, otherwise this empty slot.
                     const int64_t target =
                             (first_deleted >= 0) ? first_deleted : idx;
                     int32_t expected = (first_deleted >= 0) ? kSYCLSlotDeleted
                                                             : kSYCLSlotEmpty;
                     sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>
                             tst(slot_state[target]);
                     if (tst.compare_exchange_strong(
                                 expected, kSYCLSlotLocked,
                                 sycl::memory_order::acq_rel,
                                 sycl::memory_order::relaxed)) {
                         const buf_index_t bi = accessor.DeviceAllocate();
                         Key* slot_key =
                                 static_cast<Key*>(accessor.GetKeyPtr(bi));
                         *slot_key = key;

                         for (int j = 0; j < n_values; ++j) {
                             const int64_t dsize = accessor.value_dsizes_[j];
                             uint8_t* dst = static_cast<uint8_t*>(
                                     accessor.GetValuePtr(bi, j));
                             const uint8_t* src =
                                     static_cast<const uint8_t*>(
                                             d_values_soa[j]) +
                                     dsize * tid;
                             for (int64_t b = 0; b < dsize; ++b) {
                                 dst[b] = src[b];
                             }
                         }

                         slot_buf_index[target] = bi;
                         // Publish: release so readers that acquire see the
                         // key/value/buffer-index writes above.
                         tst.store(kSYCLSlotOccupied,
                                   sycl::memory_order::release);

                         output_buf_indices[tid] = bi;
                         output_masks[tid] = true;
                         finished = true;
                         break;
                     }
                     // Lost the claim race: restart the probe from home.
                     restart = true;
                     break;
                 }

                 if (finished || !restart) break;
             }
         }).wait_and_throw();

    MemoryManager::Free(d_values_soa, this->device_);
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Find(const void* input_keys,
                                          buf_index_t* output_buf_indices,
                                          bool* output_masks,
                                          int64_t count) {
    if (count == 0) return;

    const Key* keys = static_cast<const Key*>(input_keys);
    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    int32_t* slot_state = slot_state_;
    buf_index_t* slot_buf_index = slot_buf_index_;
    const int64_t bucket_count = bucket_count_;
    Hash hash_fn;
    Eq eq_fn;

    queue_.parallel_for(count, [=](int64_t tid) {
             const Key key = keys[tid];
             const int64_t home =
                     static_cast<int64_t>(hash_fn(key) %
                                          static_cast<uint64_t>(bucket_count));

             bool found = false;
             buf_index_t result = 0;
             for (int64_t probe = 0; probe < bucket_count; ++probe) {
                 const int64_t idx = (home + probe) % bucket_count;
                 const int32_t s = slot_state[idx];
                 if (s == kSYCLSlotEmpty) {
                     break;  // Probe chain ended; key not present.
                 }
                 if (s == kSYCLSlotOccupied) {
                     const buf_index_t bi = slot_buf_index[idx];
                     const Key* slot_key =
                             static_cast<const Key*>(accessor.GetKeyPtr(bi));
                     if (eq_fn(*slot_key, key)) {
                         found = true;
                         result = bi;
                         break;
                     }
                 }
                 // kSYCLSlotDeleted: continue probing.
             }
             output_masks[tid] = found;
             output_buf_indices[tid] = found ? result : 0;
         }).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Erase(const void* input_keys,
                                           bool* output_masks,
                                           int64_t count) {
    if (count == 0) return;

    const Key* keys = static_cast<const Key*>(input_keys);
    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    int32_t* slot_state = slot_state_;
    buf_index_t* slot_buf_index = slot_buf_index_;
    const int64_t bucket_count = bucket_count_;
    Hash hash_fn;
    Eq eq_fn;

    queue_.parallel_for(count, [=](int64_t tid) {
             const Key key = keys[tid];
             const int64_t home =
                     static_cast<int64_t>(hash_fn(key) %
                                          static_cast<uint64_t>(bucket_count));

             bool erased = false;
             for (int64_t probe = 0; probe < bucket_count; ++probe) {
                 const int64_t idx = (home + probe) % bucket_count;
                 sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                         st(slot_state[idx]);
                 int32_t s = st.load(sycl::memory_order::acquire);
                 if (s == kSYCLSlotEmpty) {
                     break;
                 }
                 if (s == kSYCLSlotOccupied) {
                     const buf_index_t bi = slot_buf_index[idx];
                     const Key* slot_key =
                             static_cast<const Key*>(accessor.GetKeyPtr(bi));
                     if (eq_fn(*slot_key, key)) {
                         int32_t expected = kSYCLSlotOccupied;
                         if (st.compare_exchange_strong(
                                     expected, kSYCLSlotDeleted,
                                     sycl::memory_order::acq_rel,
                                     sycl::memory_order::relaxed)) {
                             accessor.DeviceFree(bi);
                             erased = true;
                         }
                         break;  // Found the key on this probe chain.
                     }
                 }
                 // kSYCLSlotDeleted or different key: continue probing.
             }
             output_masks[tid] = erased;
         }).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::GetActiveIndices(
        buf_index_t* output_indices) {
    int32_t* slot_state = slot_state_;
    buf_index_t* slot_buf_index = slot_buf_index_;
    const int64_t bucket_count = bucket_count_;

    int* d_count =
            static_cast<int*>(MemoryManager::Malloc(sizeof(int), this->device_));
    queue_.memset(d_count, 0, sizeof(int)).wait_and_throw();

    queue_.parallel_for(bucket_count, [=](int64_t idx) {
             if (slot_state[idx] == kSYCLSlotOccupied) {
                 sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                         counter(*d_count);
                 const int pos = counter.fetch_add(1);
                 output_indices[pos] = slot_buf_index[idx];
             }
         }).wait_and_throw();

    int count = 0;
    MemoryManager::MemcpyToHost(&count, d_count, this->device_, sizeof(int));
    MemoryManager::Free(d_count, this->device_);
    return static_cast<int64_t>(count);
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Clear() {
    this->buffer_->ResetHeap();
    queue_.memset(slot_state_, 0, bucket_count_ * sizeof(int32_t))
            .wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;
    bucket_count_ = std::max<int64_t>(capacity * 2, 1);

    // Allocate the shared key/value buffer.
    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);
    buffer_accessor_.Setup(*this->buffer_);

    // Allocate and reset the open-addressing table (EMPTY == 0).
    slot_state_ = static_cast<int32_t*>(MemoryManager::Malloc(
            bucket_count_ * sizeof(int32_t), this->device_));
    slot_buf_index_ = static_cast<buf_index_t*>(MemoryManager::Malloc(
            bucket_count_ * sizeof(buf_index_t), this->device_));
    queue_.memset(slot_state_, 0, bucket_count_ * sizeof(int32_t))
            .wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Free() {
    buffer_accessor_.Shutdown(this->device_);
    if (slot_state_) {
        MemoryManager::Free(slot_state_, this->device_);
        slot_state_ = nullptr;
    }
    if (slot_buf_index_) {
        MemoryManager::Free(slot_buf_index_, this->device_);
        slot_buf_index_ = nullptr;
    }
}

}  // namespace core
}  // namespace open3d
