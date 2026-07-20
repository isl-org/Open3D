// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file SYCLHashBackend.h
/// \brief SYCL implementation of \ref DeviceHashBackend (open addressing, in-tree).
///
/// Open3D's tensor \ref HashMap and \ref HashSet share one backend API on CPU
/// (TBB), CUDA (stdgpu default, slab optional), and SYCL (this file). See also
/// `hashmap/SYCL_DESIGN.md` for a short overview; **this file header is the
/// maintainer reference.**
///
/// \section SYCLHashRole Role in the stack
///
/// - Keys are `MiniVec<int, dim>` with dim 1–6; keys and values live in
///   \ref HashBackendBuffer, not inside the probe table.
/// - Each occupied hash slot stores a **buf_index** into those buffers (unique-key
///   map, not a multimap).
/// - **Capacity growth** is owned by \ref HashMap::Reserve / \ref HashSet::Reserve
///   (export active entries → free backend → allocate → re-insert). This backend's
///   \ref SYCLHashBackend::Reserve() override is a no-op.
///
/// \section SYCLHashData Data structure
///
/// | Component | Description |
/// |-----------|-------------|
/// | Probing | Open addressing, **linear** probing, power-of-two bucket count |
/// | Index | `(home + i) & (bucket_count - 1)` |
/// | Slot (64-bit) | 28-bit **fingerprint** \| 4-bit **state** \| 32-bit **buf_index** (see PackSlot) |
/// | States | `EMPTY` (0), `OCCUPIED`, `DELETED` (tombstone); EMPTY=0 ⇒ zero USM is empty |
/// | HashMix | MurmurHash3 fmix64 on FNV-1a before mask; fingerprint skips most false key gathers |
///
/// \section SYCLHashConcurrency Concurrency and operations
///
/// **Insert (wait-free)**
/// - Linear probe; reuse first `DELETED` slot seen when claiming `EMPTY`.
/// - Allocate one buffer element, write key/values, **device seq_cst fence**, then a
///   **single CAS** `EMPTY`/`DELETED` → `OCCUPIED` (no lock spin — on Intel Xe,
///   a waiting subgroup lane can hang the device).
/// - Duplicate-key races: loser gets `masks=false` and **DeviceFree**'s its unused
///   buffer slot ⇒ returned **buf_indices are valid gather indices but not necessarily
///   dense in `[0, Size())`** (see \ref HashMap user docs).
///
/// **Find / Erase**
/// - Same probe; `EMPTY` ends the chain.
/// - Erase: CAS `OCCUPIED` → `DELETED`, free buffer slot; `occupied_count_` decremented.
/// - `non_empty_count_` tracks occupied + deleted (tombstone pressure).
///
/// **GetActiveIndices**
/// - Work-group exclusive scan + one global `fetch_add` per group (not one atomic per slot).
///
/// **SYCLHashDeviceLookup**
/// - Immutable snapshot for device kernels (`Find` uses plain loads, no atomics).
/// - Do **not** mutate the table while a lookup view is in use.
///
/// \section SYCLHashVsCuda Compared to CUDA backends
///
/// | Aspect | CUDA default (stdgpu) | SYCL (this file) |
/// |--------|----------------------|------------------|
/// | Dependency | Third-party stdgpu | None (in-tree) |
/// | Probing | Library-defined | Linear + stored fingerprint |
/// | In-kernel find | `map.find(key)` | `SYCLHashDeviceLookup::Find(key)` |
///
/// Callers that treat **buf_indices as dense row ids** (e.g. voxel aggregation) must
/// remap via unique insert slots (`masks==true`) or \ref HashMap::GetActiveIndices().
/// Uses that only need masks or active-index lists are unchanged.
///
/// \section SYCLHashFiles Related files
///
/// - `SYCL/SYCLHashBackendBufferAccessor.h` — USM key/value buffer access
/// - `SYCL/CreateSYCLHashBackend.cpp` — factory / dtype dispatch

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

namespace open3d {
namespace core {

/// Slot occupancy for packed hash table entries (see PackSlot layout in this
/// file's implementation). kSlotEmpty must be 0 (memset-initialized table).
enum HashSlotState : uint32_t {
    kSlotEmpty = 0,
    kSlotOccupied = 1,
    kSlotDeleted = 2,
};

namespace {

constexpr int64_t kHashWgSize = 1024;
/// The inverse of the target max load factor.
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

// MurmurHash3 fmix64 finalizer on FNV-1a output before bucket mask and
// fingerprint extract.
inline uint64_t HashMix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

}  // namespace

/// Read-only table view for device kernels (see file header).
template <typename Key, typename Hash, typename Eq>
struct SYCLHashDeviceLookup {
    uint64_t* slot_data = nullptr;           ///< USM packed slot array.
    int64_t bucket_count = 0;                ///< Power-of-two bucket count.
    SYCLHashBackendBufferAccessor accessor;  ///< Key/value buffer accessor.
    Hash hash_fn{};                          ///< Key hash functor.
    Eq eq_fn{};                              ///< Key equality functor.

    /// Linear-probe lookup; returns buffer index or -1 if not found.
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

/// \ref DeviceHashBackend for SYCL devices (algorithm in file header).
template <typename Key, typename Hash, typename Eq>
class SYCLHashBackend : public DeviceHashBackend {
public:
    SYCLHashBackend(int64_t init_capacity,
                    int64_t key_dsize,
                    const std::vector<int64_t>& value_dsizes,
                    const Device& device,
                    int64_t wg_size = kHashWgSize);
    ~SYCLHashBackend();

    /// No-op; use \ref HashMap::Reserve for capacity growth.
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
    /// Occupied + deleted slots (rehash guard; see file header).
    int64_t GetNonEmptyCount() const override;
    int64_t GetBucketCount() const override;
    std::vector<int64_t> BucketSizes() const override;
    float LoadFactor() const override;

    void Allocate(int64_t capacity) override;
    void Free() override;

    /// Snapshot for device kernels; table must not be mutated while in use.
    SYCLHashDeviceLookup<Key, Hash, Eq> GetDeviceLookup() const {
        SYCLHashDeviceLookup<Key, Hash, Eq> view;
        view.slot_data = slot_data_;
        view.bucket_count = bucket_count_;
        view.accessor = buffer_accessor_;
        return view;
    }

protected:
    SYCLHashBackendBufferAccessor buffer_accessor_;

    uint64_t* slot_data_ = nullptr;   ///< USM packed slots.
    int* occupied_count_ = nullptr;   ///< Device live entry count.
    int* non_empty_count_ = nullptr;  ///< Device occupied + tombstone count.
    int64_t bucket_count_ = 0;
    int64_t wg_size_ = kHashWgSize;  ///< SYCL work-group size for kernels.

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

    // Bulk-reserve `count` heap slots up front, one per thread, via plain
    // index arithmetic (no atomics) -- mirrors CUDA's SlabHashBackend::Insert
    // (see InsertKernelPass0). This is what makes the DeviceFree() call below
    // safe: since this kernel never calls DeviceAllocate() itself, its
    // DeviceFree() calls cannot race with a concurrent allocation of the same
    // slot. Interleaving DeviceAllocate() (lazily, inside the probe loop) and
    // DeviceFree() (on the leak path) within the same kernel invocation was
    // tried first, but heap_top_ fetch_add/fetch_sub pairs only order the
    // atomic counter itself, not the plain heap_[] array reads/writes tied to
    // it, so concurrent Allocate/Free calls could still observe/overwrite
    // each other's heap_[] slot before the intended visibility order was
    // established, corrupting the free list (observed as extra/duplicate
    // buf_indices ending up marked valid).
    const int prev_heap_top = this->buffer_->GetHeapTopIndex();
    {
        const int new_heap_top = prev_heap_top + static_cast<int>(count);
        queue_.memcpy(buffer_accessor_.heap_top_, &new_heap_top, sizeof(int))
                .wait();
    }

    SYCLHashBackendBufferAccessor accessor = buffer_accessor_;
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;
    const int64_t capacity = accessor.capacity_;
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

            // Slot for this thread was bulk-reserved on the host before
            // kernel launch (see prev_heap_top above): no atomic is needed
            // here, and no other thread can read/write this exact heap_[]
            // entry, since every tid maps to a disjoint reserved_index.
            const int64_t reserved_index =
                    static_cast<int64_t>(prev_heap_top) + tid;
            buf_index_t my_bi =
                    (reserved_index < capacity)
                            ? accessor.heap_[reserved_index]
                            : SYCLHashBackendBufferAccessor::kInvalidBufIndex;
            bool key_published = false;
            if (my_bi != SYCLHashBackendBufferAccessor::kInvalidBufIndex) {
                // Publish key/values immediately: whether this slot is
                // ultimately kept (CAS succeeds) or returned to the heap
                // (duplicate found / table full) is decided below, but the
                // write itself never races with anything since this slot is
                // exclusively owned by this thread until (at most) one
                // DeviceFree() call at the end.
                Key* slot_key = static_cast<Key*>(accessor.GetKeyPtr(my_bi));
                *slot_key = key;

                for (int j = 0; j < n_values; ++j) {
                    const int64_t blocks =
                            accessor.value_blocks_per_element_[j];
                    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(
                            common_block_size, [&]() {
                                using val_block_t = block_t;
                                val_block_t* dst =
                                        reinterpret_cast<val_block_t*>(
                                                accessor.GetValuePtr(my_bi, j));
                                const val_block_t* src =
                                        reinterpret_cast<const val_block_t*>(
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

            bool finished = false;
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
                        // Heap was exhausted before kernel launch (no slot
                        // reserved for this thread): nothing to insert.
                        break;
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

            // This thread's reserved slot was published (key/values written)
            // but never won the CAS that installs it into the table --
            // either because a concurrent insert of the same key won the
            // race first (duplicate found on a post-restart probe) or
            // kMaxOuterIter was exhausted. Return it to the heap so capacity
            // is not silently reduced. This DeviceFree() cannot race with a
            // concurrent DeviceAllocate(), since slots are only ever
            // allocated in bulk on the host before this kernel is launched.
            if (key_published && !output_masks[tid]) {
                accessor.DeviceFree(my_bi);
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
