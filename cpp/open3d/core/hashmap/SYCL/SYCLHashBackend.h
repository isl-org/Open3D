// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

#include "open3d/core/BlockCopyDispatch.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/SYCL/SYCLHashBackendBufferAccessor.h"
#include "open3d/utility/Logging.h"

// SYCLHashBackend — device-side open-addressing hash table (linear probing).
//
// Design notes
//   • Slot transitions publish in a single CAS (Xe subgroup lanes run in
//     lockstep; a spinning/waiting lane can hang the device).
//   • Each slot is one packed uint64 (state, buf_index, fingerprint) — a
//     single load per probe.
//   • A device seq_cst fence separates writing a slot's key/value from the
//     CAS that publishes it, so probers never see stale or uninitialized
//     data.
//   • GetActiveIndices uses one work-group exclusive scan + one fetch_add
//     per group, not one atomic per occupied slot.
//   • Rehash triggers on non_empty_count (occupied + deleted), not live
//     size alone, so tombstones cannot silently fill the table.
//
// Algorithm
//   Table: USM array of packed slots (see PackSlot). Keys/values live in an
//   external HashBackendBuffer; each OCCUPIED slot stores a buf_index. Probe
//   index is (home + i) & (bucket_count - 1) with bucket_count a power of two.
//   HashMix (MurmurHash3 fmix64) is applied to FNV-1a before masking so local
//   keys do not cluster in low bits. A 28-bit fingerprint from the mixed hash
//   filters most key-buffer gathers on probe, using bits otherwise unused
//   between the packed state and buf_index fields at zero extra memory cost
//   (see PackSlot).
//   Insert (wait-free): linear probe; on EMPTY claim the first DELETED seen for
//   reuse. Allocate one buffer element, copy key/values (vectorized blocks),
//   device seq_cst fence, then CAS slot EMPTY/DELETED → OCCUPIED. Losers
//   restart from home (duplicate-key check, another empty slot). No per-slot
//   lock or spin. Buffer slots left allocated when a duplicate key wins the
//   race (same as CUDA one-slot-per-input). occupied_count / non_empty_count
//   updated per work-group reduction. Find / Erase: same probe; EMPTY ends the
//   chain. Erase CAS OCCUPIED → DELETED, frees buffer slot, decrements
//   occupied_count only. SYCLHashDeviceLookup: immutable table view for
//   kernels; plain loads, no atomics. Callers must not mutate the table while a
//   lookup view is in use.

namespace open3d {
namespace core {

// EMPTY must be 0 so memset-initialized slot_data_ is an empty table.
enum HashSlotState : uint32_t {
    kSlotEmpty = 0,
    kSlotOccupied = 1,
    kSlotDeleted = 2,
};

namespace {

// Insert/Find/Erase throughput improves monotonically with wg_size up to the
// device max (see benchmarks/core/HashMapSYCLTuning.cpp); clamped at
// construction time to the actual device max, see SYCLHashBackend's
// constructor.
constexpr int64_t kHashWgSize = 1024;

// bucket_count_ = NextPowerOfTwo(capacity * kHashBucketCountMultiplier),
// i.e. the inverse of the target max load factor.
constexpr int64_t kHashBucketCountMultiplier = 2;

// Packed slot: [63:36] fingerprint (28 bits), [35:32] state, [31:0] buf_index.
inline uint64_t PackSlot(uint32_t state, buf_index_t bi, uint32_t fingerprint) {
    return (static_cast<uint64_t>(fingerprint) << 36) |
           (static_cast<uint64_t>(state) << 32) | static_cast<uint64_t>(bi);
}

inline void UnpackSlot(uint64_t packed,
                       uint32_t& state,
                       buf_index_t& bi,
                       uint32_t& fingerprint) {
    bi = static_cast<buf_index_t>(packed & 0xffffffffULL);
    state = static_cast<uint32_t>((packed >> 32) & 0xfULL);
    fingerprint = static_cast<uint32_t>(packed >> 36);
}

// Round up to power of two for `(home + probe) & (bucket_count - 1)` indexing.
inline int64_t NextPowerOfTwo(int64_t n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

// fmix64 finalizer on FNV-1a output before bucket mask and fingerprint extract.
inline uint64_t HashMix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

}  // namespace

/// Read-only table view for SYCL kernels (see file header). Find() uses plain
/// loads; the table must stay immutable while the view is used.
template <typename Key, typename Hash, typename Eq>
struct SYCLHashDeviceLookup {
    uint64_t* slot_data = nullptr;
    int64_t bucket_count = 0;
    SYCLHashBackendBufferAccessor accessor;
    Hash hash_fn{};
    Eq eq_fn{};

    /// Returns buffer index, or static_cast<buf_index_t>(-1) if not found.
    buf_index_t Find(const Key& key) const {
        const int64_t mask = bucket_count - 1;
        const uint64_t hash = HashMix(hash_fn(key));
        const int64_t home = static_cast<int64_t>(hash & mask);
        const uint32_t my_fingerprint =
                static_cast<uint32_t>((hash >> 16) & 0xfffffffULL);

        for (int64_t probe = 0; probe < bucket_count; ++probe) {
            const int64_t idx = (home + probe) & mask;
            uint64_t packed = slot_data[idx];
            uint32_t s;
            buf_index_t bi;
            uint32_t fp;
            UnpackSlot(packed, s, bi, fp);
            if (s == kSlotEmpty) {
                break;
            }
            if (s == kSlotOccupied && fp == my_fingerprint) {
                const Key* slot_key =
                        static_cast<const Key*>(accessor.GetKeyPtr(bi));
                if (eq_fn(*slot_key, key)) {
                    return bi;
                }
            }
        }
        return static_cast<buf_index_t>(-1);
    }
};

/// SYCL DeviceHashBackend; algorithm is documented at the top of this file.
/// bucket_count_ = NextPowerOfTwo(capacity * kHashBucketCountMultiplier)
template <typename Key, typename Hash, typename Eq>
class SYCLHashBackend : public DeviceHashBackend {
public:
    SYCLHashBackend(int64_t init_capacity,
                    int64_t key_dsize,
                    const std::vector<int64_t>& value_dsizes,
                    const Device& device,
                    int64_t wg_size = kHashWgSize);
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
    /// Occupied + deleted slots (HashMap rehash guard; see file header).
    int64_t GetNonEmptyCount() const override;
    int64_t GetBucketCount() const override;
    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

    void Allocate(int64_t capacity) override;
    void Free() override;

    SYCLHashDeviceLookup<Key, Hash, Eq> GetDeviceLookup() const {
        SYCLHashDeviceLookup<Key, Hash, Eq> view;
        view.slot_data = slot_data_;
        view.bucket_count = bucket_count_;
        view.accessor = buffer_accessor_;
        return view;
    }

protected:
    SYCLHashBackendBufferAccessor buffer_accessor_;

    uint64_t* slot_data_ = nullptr;
    int* occupied_count_ = nullptr;
    int* non_empty_count_ = nullptr;
    int64_t bucket_count_ = 0;
    int64_t wg_size_ = kHashWgSize;

    sycl::queue queue_;
};

template <typename Key, typename Hash, typename Eq>
SYCLHashBackend<Key, Hash, Eq>::SYCLHashBackend(
        int64_t init_capacity,
        int64_t key_dsize,
        const std::vector<int64_t>& value_dsizes,
        const Device& device,
        int64_t wg_size)
    : DeviceHashBackend(init_capacity, key_dsize, value_dsizes, device),
      wg_size_(wg_size),
      queue_(sy::SYCLContext::GetInstance().GetDefaultQueue(device)) {
    const int64_t device_max_wg_size = static_cast<int64_t>(
            queue_.get_device()
                    .get_info<sycl::info::device::max_work_group_size>());
    wg_size_ = std::min(wg_size_, std::max<int64_t>(1, device_max_wg_size));
    Allocate(init_capacity);
}

template <typename Key, typename Hash, typename Eq>
SYCLHashBackend<Key, Hash, Eq>::~SYCLHashBackend() {
    Free();
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::Size() const {
    if (!occupied_count_) {
        return 0;
    }
    int count = 0;
    MemoryManager::MemcpyToHost(&count, occupied_count_, this->device_,
                                sizeof(int));
    return static_cast<int64_t>(count);
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::GetNonEmptyCount() const {
    if (!non_empty_count_) {
        return 0;
    }
    int count = 0;
    MemoryManager::MemcpyToHost(&count, non_empty_count_, this->device_,
                                sizeof(int));
    return static_cast<int64_t>(count);
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

    if (n_values > 16) {
        utility::LogError(
                "SYCL hashmap supports up to 16 value arrays, but got {}.",
                n_values);
    }

    // Copy host-side SoA pointers into a struct capturable by the SYCL kernel.
    struct ValuesSoA {
        const void* ptrs[16];
    } values_soa;
    for (int i = 0; i < n_values; ++i) {
        values_soa.ptrs[i] = input_values_soa[i];
    }

    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;
    int* occupied_count = occupied_count_;
    int* non_empty_count = non_empty_count_;
    constexpr int kMaxOuterIter = 1 << 20;  // Termination if table is full.
    Hash hash_fn;
    Eq eq_fn;

    const int64_t common_block_size = buffer_accessor_.common_block_size_;

    auto insert_kernel = [=](sycl::nd_item<1>
                                     item) [[intel::kernel_args_restrict]] {
        const int64_t tid = item.get_global_id(0);
        int my_new_occupied = 0;
        int my_new_nonempty = 0;

        if (tid < count) {
            sycl::atomic_fence(sycl::memory_order::seq_cst,
                               sycl::memory_scope::device);

            const Key key = keys[tid];
            output_buf_indices[tid] = 0;
            output_masks[tid] = false;

            const int64_t mask = bucket_count - 1;
            const uint64_t hash = HashMix(hash_fn(key));
            const int64_t home = static_cast<int64_t>(hash & mask);
            const uint32_t my_fingerprint =
                    static_cast<uint32_t>((hash >> 16) & 0xfffffffULL);

            bool finished = false;
            buf_index_t my_bi = SYCLHashBackendBufferAccessor::kInvalidBufIndex;
            bool key_published = false;
            int outer_iter = 0;
            while (!finished) {
                if (++outer_iter > kMaxOuterIter) {
                    break;
                }
                int64_t first_deleted = -1;
                bool restart = false;

                for (int64_t probe = 0; probe < bucket_count; ++probe) {
                    const int64_t idx = (home + probe) & mask;
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                            st(slot_data[idx]);

                    uint64_t packed = st.load(sycl::memory_order::acquire);
                    uint32_t s;
                    buf_index_t bi;
                    uint32_t fp;
                    UnpackSlot(packed, s, bi, fp);

                    if (s == kSlotOccupied) {
                        if (fp == my_fingerprint) {
                            sycl::atomic_fence(sycl::memory_order::seq_cst,
                                               sycl::memory_scope::device);
                            const Key* slot_key = static_cast<const Key*>(
                                    accessor.GetKeyPtr(bi));
                            if (eq_fn(*slot_key, key)) {
                                output_buf_indices[tid] = bi;
                                output_masks[tid] = false;
                                finished = true;
                                break;
                            }
                        }
                        continue;
                    }

                    if (s == kSlotDeleted) {
                        if (first_deleted < 0) first_deleted = idx;
                        continue;
                    }

                    if (!key_published) {
                        my_bi = accessor.DeviceAllocate();
                        if (my_bi ==
                            SYCLHashBackendBufferAccessor::kInvalidBufIndex) {
                            break;
                        }
                        Key* slot_key =
                                static_cast<Key*>(accessor.GetKeyPtr(my_bi));
                        *slot_key = key;

                        for (int j = 0; j < n_values; ++j) {
                            const int64_t blocks =
                                    accessor.value_blocks_per_element_[j];
                            DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(
                                    common_block_size, [&]() {
                                        using val_block_t = block_t;
                                        val_block_t* dst =
                                                reinterpret_cast<val_block_t*>(
                                                        accessor.GetValuePtr(
                                                                my_bi, j));
                                        const val_block_t* src =
                                                reinterpret_cast<
                                                        const val_block_t*>(
                                                        values_soa.ptrs[j]) +
                                                blocks * tid;
                                        for (int64_t b = 0; b < blocks; ++b) {
                                            dst[b] = src[b];
                                        }
                                    });
                        }

                        sycl::atomic_fence(sycl::memory_order::seq_cst,
                                           sycl::memory_scope::device);
                        key_published = true;
                    }

                    const int64_t target =
                            (first_deleted >= 0) ? first_deleted : idx;
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                            tst(slot_data[target]);
                    uint64_t expected_packed =
                            (first_deleted >= 0)
                                    ? tst.load(sycl::memory_order::acquire)
                                    : 0ULL;
                    const uint32_t prev_state = static_cast<uint32_t>(
                            (expected_packed >> 32) & 0xfULL);
                    if ((prev_state == kSlotEmpty ||
                         prev_state == kSlotDeleted) &&
                        tst.compare_exchange_strong(
                                expected_packed,
                                PackSlot(kSlotOccupied, my_bi, my_fingerprint),
                                sycl::memory_order::acq_rel,
                                sycl::memory_order::relaxed)) {
                        output_buf_indices[tid] = my_bi;
                        output_masks[tid] = true;
                        my_new_occupied = 1;
                        if (prev_state == kSlotEmpty) {
                            my_new_nonempty = 1;
                        }
                        finished = true;
                        break;
                    }

                    restart = true;
                    break;
                }

                if (finished || !restart) {
                    break;
                }
            }
        }  // tid < count

        const int wg_occupied = sycl::reduce_over_group(
                item.get_group(), my_new_occupied, sycl::plus<int>{});
        const int wg_nonempty = sycl::reduce_over_group(
                item.get_group(), my_new_nonempty, sycl::plus<int>{});
        if (item.get_local_id(0) == 0) {
            if (wg_occupied > 0) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                        oc(*occupied_count);
                oc.fetch_add(wg_occupied);
            }
            if (wg_nonempty > 0) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                        nec(*non_empty_count);
                nec.fetch_add(wg_nonempty);
            }
        }
    };

    const int64_t wg_size = wg_size_;
    const int64_t global_size = ((count + wg_size - 1) / wg_size) * wg_size;
    queue_.submit([&](sycl::handler& cgh) {
              cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                               insert_kernel);
          }).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Find(const void* input_keys,
                                          buf_index_t* output_buf_indices,
                                          bool* output_masks,
                                          int64_t count) {
    if (count == 0) return;

    const Key* keys = static_cast<const Key*>(input_keys);
    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;
    Hash hash_fn;
    Eq eq_fn;

    auto find_kernel =
            [=](sycl::nd_item<1> item) [[intel::kernel_args_restrict]] {
                const int64_t tid = item.get_global_id(0);
                if (tid >= count) return;
                const Key key = keys[tid];
                const int64_t mask = bucket_count - 1;
                const uint64_t hash = HashMix(hash_fn(key));
                const int64_t home = static_cast<int64_t>(hash & mask);
                const uint32_t my_fingerprint =
                        static_cast<uint32_t>((hash >> 16) & 0xfffffffULL);

                bool found = false;
                buf_index_t result = 0;
                for (int64_t probe = 0; probe < bucket_count; ++probe) {
                    const int64_t idx = (home + probe) & mask;
                    sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                            st(slot_data[idx]);
                    uint64_t packed = st.load(sycl::memory_order::acquire);
                    uint32_t s;
                    buf_index_t bi;
                    uint32_t fp;
                    UnpackSlot(packed, s, bi, fp);

                    if (s == kSlotEmpty) {
                        break;
                    }
                    if (s == kSlotOccupied && fp == my_fingerprint) {
                        const Key* slot_key =
                                static_cast<const Key*>(accessor.GetKeyPtr(bi));
                        if (eq_fn(*slot_key, key)) {
                            found = true;
                            result = bi;
                            break;
                        }
                    }
                }
                output_masks[tid] = found;
                output_buf_indices[tid] = found ? result : 0;
            };

    const int64_t wg_size = wg_size_;
    const int64_t global_size = ((count + wg_size - 1) / wg_size) * wg_size;
    queue_.submit([&](sycl::handler& cgh) {
              cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                               find_kernel);
          }).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Erase(const void* input_keys,
                                           bool* output_masks,
                                           int64_t count) {
    if (count == 0) return;

    const Key* keys = static_cast<const Key*>(input_keys);
    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;
    int* occupied_count = occupied_count_;
    Hash hash_fn;
    Eq eq_fn;

    auto erase_kernel = [=](sycl::nd_item<1>
                                    item) [[intel::kernel_args_restrict]] {
        const int64_t tid = item.get_global_id(0);
        if (tid >= count) return;
        const Key key = keys[tid];
        const int64_t mask = bucket_count - 1;
        const uint64_t hash = HashMix(hash_fn(key));
        const int64_t home = static_cast<int64_t>(hash & mask);
        const uint32_t my_fingerprint =
                static_cast<uint32_t>((hash >> 16) & 0xfffffffULL);

        bool erased = false;
        for (int64_t probe = 0; probe < bucket_count; ++probe) {
            const int64_t idx = (home + probe) & mask;
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                    st(slot_data[idx]);
            uint64_t packed = st.load(sycl::memory_order::acquire);
            uint32_t s;
            buf_index_t bi;
            uint32_t fp;
            UnpackSlot(packed, s, bi, fp);

            if (s == kSlotEmpty) {
                break;
            }
            if (s == kSlotOccupied && fp == my_fingerprint) {
                const Key* slot_key =
                        static_cast<const Key*>(accessor.GetKeyPtr(bi));
                if (eq_fn(*slot_key, key)) {
                    uint64_t expected_packed = packed;
                    uint64_t deleted_val = PackSlot(kSlotDeleted, bi, fp);
                    if (st.compare_exchange_strong(
                                expected_packed, deleted_val,
                                sycl::memory_order::acq_rel,
                                sycl::memory_order::relaxed)) {
                        accessor.DeviceFree(bi);
                        erased = true;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                                oc(*occupied_count);
                        oc.fetch_sub(1);
                    }
                    break;
                }
            }
        }
        output_masks[tid] = erased;
    };

    const int64_t wg_size = wg_size_;
    const int64_t global_size = ((count + wg_size - 1) / wg_size) * wg_size;
    queue_.submit([&](sycl::handler& cgh) {
              cgh.parallel_for(sycl::nd_range<1>(global_size, wg_size),
                               erase_kernel);
          }).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
int64_t SYCLHashBackend<Key, Hash, Eq>::GetActiveIndices(
        buf_index_t* output_indices) {
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;

    int* d_count = static_cast<int*>(
            MemoryManager::Malloc(sizeof(int), this->device_));
    queue_.memset(d_count, 0, sizeof(int)).wait_and_throw();

    const int64_t kWgSize = wg_size_;

    auto scan_kernel = [=](sycl::nd_item<1> item) {
        int64_t idx = item.get_global_id(0);
        auto group = item.get_group();

        bool is_occupied = false;
        buf_index_t bi = 0;
        if (idx < bucket_count) {
            const uint64_t packed = slot_data[idx];
            uint32_t s = static_cast<uint32_t>((packed >> 32) & 0xfULL);
            if (s == kSlotOccupied) {
                is_occupied = true;
                bi = static_cast<buf_index_t>(packed & 0xffffffffULL);
            }
        }

        int local_val = is_occupied ? 1 : 0;
        int local_offset = sycl::exclusive_scan_over_group(group, local_val,
                                                           sycl::plus<int>{});
        int group_total =
                sycl::reduce_over_group(group, local_val, sycl::plus<int>{});

        int group_start = 0;
        if (item.get_local_id(0) == 0 && group_total > 0) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                    counter(*d_count);
            group_start = counter.fetch_add(group_total);
        }
        group_start = sycl::group_broadcast(group, group_start, 0);

        if (is_occupied) {
            output_indices[group_start + local_offset] = bi;
        }
    };

    int64_t global_size = ((bucket_count + kWgSize - 1) / kWgSize) * kWgSize;
    queue_.submit([&](sycl::handler& cgh) {
              cgh.parallel_for(sycl::nd_range<1>(global_size, kWgSize),
                               scan_kernel);
          }).wait_and_throw();

    int count = 0;
    MemoryManager::MemcpyToHost(&count, d_count, this->device_, sizeof(int));
    MemoryManager::Free(d_count, this->device_);
    return static_cast<int64_t>(count);
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Clear() {
    this->buffer_->ResetHeap();
    queue_.memset(slot_data_, 0, bucket_count_ * sizeof(uint64_t))
            .wait_and_throw();
    if (occupied_count_) {
        queue_.memset(occupied_count_, 0, sizeof(int)).wait_and_throw();
    }
    if (non_empty_count_) {
        queue_.memset(non_empty_count_, 0, sizeof(int)).wait_and_throw();
    }
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Allocate(int64_t capacity) {
    this->capacity_ = capacity;
    bucket_count_ = NextPowerOfTwo(
            std::max<int64_t>(capacity * kHashBucketCountMultiplier, 1));

    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);
    buffer_accessor_.Setup(*this->buffer_);

    slot_data_ = static_cast<uint64_t*>(MemoryManager::Malloc(
            bucket_count_ * sizeof(uint64_t), this->device_));
    queue_.memset(slot_data_, 0, bucket_count_ * sizeof(uint64_t))
            .wait_and_throw();

    occupied_count_ = static_cast<int*>(
            MemoryManager::Malloc(sizeof(int), this->device_));
    queue_.memset(occupied_count_, 0, sizeof(int)).wait_and_throw();

    non_empty_count_ = static_cast<int*>(
            MemoryManager::Malloc(sizeof(int), this->device_));
    queue_.memset(non_empty_count_, 0, sizeof(int)).wait_and_throw();
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Free() {
    buffer_accessor_.Shutdown(this->device_);
    if (slot_data_) {
        MemoryManager::Free(slot_data_, this->device_);
        slot_data_ = nullptr;
    }
    if (occupied_count_) {
        MemoryManager::Free(occupied_count_, this->device_);
        occupied_count_ = nullptr;
    }
    if (non_empty_count_) {
        MemoryManager::Free(non_empty_count_, this->device_);
        non_empty_count_ = nullptr;
    }
}

}  // namespace core
}  // namespace open3d
