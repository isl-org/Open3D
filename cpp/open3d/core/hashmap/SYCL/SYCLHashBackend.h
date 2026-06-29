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

#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLBlockCopyDispatch.h"
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
enum SYCLHashSlotState : uint32_t {
    kSYCLSlotEmpty = 0,
    kSYCLSlotLocked = 1,
    kSYCLSlotOccupied = 2,
    kSYCLSlotDeleted = 3,
};

namespace {

// ── Packed slot layout ───────────────────────────────────────────────────────
// Each table entry is a single uint64 combining three fields:
//   bits  0-31 : buf_index  — index into the external key/value buffer
//   bits 32-35 : state      — one of SYCLHashSlotState (4 values, 2 bits used)
//   bits 48-63 : fingerprint — 16-bit extract of the mixed hash
//
// Packing into one 64-bit word means each probe step issues a single atomic
// load instead of two separate array reads (slot_state[] + slot_buf_index[]).
// The fingerprint allows most key-mismatches to be rejected before
// dereferencing the external key buffer (an expensive random gather into USM
// device memory).
inline uint64_t PackSlot(uint32_t state, buf_index_t bi, uint16_t fingerprint) {
    return (static_cast<uint64_t>(fingerprint) << 48) |
           (static_cast<uint64_t>(state) << 32) | static_cast<uint64_t>(bi);
}

inline void UnpackSlot(uint64_t packed,
                       uint32_t& state,
                       buf_index_t& bi,
                       uint16_t& fingerprint) {
    bi = static_cast<buf_index_t>(packed & 0xffffffffULL);
    state = static_cast<uint32_t>((packed >> 32) & 0xfULL);
    fingerprint = static_cast<uint16_t>(packed >> 48);
}

// ── Power-of-two bucket count ────────────────────────────────────────────────
// Rounding bucket_count up to a power of two lets the slot index computation
// use a bitwise-AND mask  `(home + probe) & (bucket_count - 1)`  instead of a
// modulo `% bucket_count`. GPUs do not have a native 64-bit integer divide so
// the compiler emits a multi-instruction emulated sequence; masking costs a
// single ALU cycle. See also: HashMix, which compensates for the low-bit bias
// introduced by masking.
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

// ── Hash finalizer (fmix64 from MurmurHash3) ─────────────────────────────────
// MiniVecHash (in hashmap/Dispatch.h) uses FNV-1a, which has excellent overall
// distribution but weaker avalanche in its low-order bits relative to its
// high-order bits. When we keep only the low log2(bucket_count) bits via an
// AND mask, keys with spatially local patterns (e.g., consecutive 3D voxel
// coordinates) can produce clustering. HashMix applies the MurmurHash3 fmix64
// bijective finalizer — three XOR-shifts and two multiplies — to redistribute
// entropy uniformly across all 64 bits before masking. The same mixed hash is
// also used to derive the 16-bit fingerprint, so fingerprint bits are
// well-distributed independently of the low bits used for bucket placement.
inline uint64_t HashMix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

}  // namespace

/// Lightweight view captured by value into SYCL kernels for read-only hash
/// table lookup (e.g. VoxelBlockGrid raycasting). The underlying table is built
/// by SYCLHashBackend::Insert and must not be mutated while any kernel holds
/// this view. Because the table is immutable during the kernel, Find() uses
/// plain (non-atomic) loads for slightly lower overhead.
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
        const uint16_t my_fingerprint =
                static_cast<uint16_t>((hash >> 16) & 0xffffULL);

        for (int64_t probe = 0; probe < bucket_count; ++probe) {
            const int64_t idx = (home + probe) & mask;
            // Plain load: the table is immutable while this view is in use.
            uint64_t packed = slot_data[idx];
            uint32_t s;
            buf_index_t bi;
            uint16_t fp;
            UnpackSlot(packed, s, bi, fp);
            if (s == kSYCLSlotEmpty) {
                break;
            }
            // Fingerprint guard: skip the key-buffer dereference unless the
            // 16-bit hash extract matches. False-positive rate ~1/65536.
            if (s == kSYCLSlotOccupied && fp == my_fingerprint) {
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

/// A native SYCL device hash backend implementing DeviceHashBackend.
///
/// Design: lock-free open-addressing table with linear probing and tombstone
/// deletion. Key design choices:
///
/// Packed 64-bit slot
///   Each table slot is one uint64 (bits 0-31: buf_index, 32-35: state,
///   48-63: fingerprint). This halves probe memory traffic vs. two separate
///   arrays (slot_state + slot_buf_index) and embeds a 16-bit fingerprint that
///   rejects most key-mismatches before the expensive external key-buffer
///   dereference.
///
/// Power-of-two bucket count + fmix64 finalizer
///   bucket_count = NextPowerOfTwo(capacity * 2), so the table is at most ~50%
///   full. Power-of-two sizing replaces the emulated 64-bit modulo in the probe
///   index computation with a single bitwise-AND mask. The fmix64 finalizer
///   (HashMix) is applied to FNV-1a's output before masking to ensure uniform
///   low-bit distribution for spatially local keys.
///
/// Non-empty slot counter for tombstone-safe rehash triggering
///   `non_empty_count_` counts occupied + deleted slots. HashMap::Insert uses
///   this (via GetNonEmptyCount) to trigger a rebuild if non-empty slots would
///   fill the table, preventing the complete elimination of EMPTY sentinels
///   that would otherwise cause silent insert failure under heavy delete-insert
///   churn.
///
/// Concurrency: slot ownership is claimed with a single compare-exchange on the
///   packed slot word (state EMPTY/DELETED -> LOCKED). Key, value, and
///   buf_index are written before the slot is published (LOCKED -> OCCUPIED)
///   with an explicit device-scoped release fence followed by a relaxed store.
///   The fence is necessary because on Intel GPU non-atomic stores write to
///   per-EU L1 cache (not coherent across EUs); only a device-scoped atomic
///   fence flushes them to the shared L2. A concurrent prober on a different
///   EU loads the slot with acquire semantics: after the fence+store it will
///   see the fully-written key and value. A probe that observes LOCKED does NOT
///   spin (spinning within the same SIMD subgroup would deadlock if the lock
///   holder is in the same subgroup). Instead it sets restart=true and
///   re-probes from home: the lock holder publishes in a few instructions, so
///   the next pass finds OCCUPIED and detects the duplicate correctly.
///
/// Value copying: performed in aligned blocks of common_block_size_ bytes
///   (via DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL and sycl::vec<>) to let the
///   compiler emit wide vector loads/stores instead of byte loops.
///
/// GetActiveIndices: uses a workgroup-local exclusive scan and one atomic
///   fetch_add per group rather than a global atomic per occupied slot,
///   reducing contention on the output counter for large tables.
template <typename Key, typename Hash, typename Eq>
class SYCLHashBackend : public DeviceHashBackend {
public:
    SYCLHashBackend(int64_t init_capacity,
                    int64_t key_dsize,
                    const std::vector<int64_t>& value_dsizes,
                    const Device& device);
    ~SYCLHashBackend();

    // Reserve is a no-op at the backend level; growth and rehash are driven by
    // HashMap::Reserve, which calls Free() + Allocate(new_capacity) +
    // re-insert.
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
    /// Returns the number of non-empty slots (occupied + deleted tombstones).
    /// Used by HashMap::Insert to trigger a rehash before tombstones saturate
    /// the table and eliminate all EMPTY sentinel slots.
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

    // Flat open-addressing table in USM device memory.
    // Each entry is a packed uint64 (see slot layout in anonymous namespace).
    uint64_t* slot_data_ = nullptr;  /* [bucket_count_] */
    int* occupied_count_ = nullptr;  /* device: live OCCUPIED slots */
    int* non_empty_count_ = nullptr; /* device: occupied + deleted slots */
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

    // Copy host-side SoA pointers into a small plain struct that can be
    // captured by value into the SYCL kernel.
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
    // Hard limit on outer-loop restarts: prevents a future regression from
    // causing a GPU watchdog timeout. Legitimate inserts never exceed a
    // handful of restarts; 512 is a conservative upper bound.
    constexpr int kMaxOuterIter = 512;
    const int64_t capacity = this->capacity_;
    Hash hash_fn;
    Eq eq_fn;

    // Captured by value into the kernel so the value-copy DISPATCH inside the
    // kernel body can use it as a uniform (same for all work-items).
    const int64_t common_block_size = buffer_accessor_.common_block_size_;
    queue_.parallel_for(count, [=](int64_t tid) {
                    // Invalidate this EU's L1 data cache before probing.
                    // Intel Xe GPU L1 caches persist across kernel
                    // dispatches and across USM allocations. Without this
                    // fence, stale L1 entries from a previous kernel (e.g.
                    // after HashMap::Clear() resets heap_top to 0, causing
                    // buf_index reuse with new key values) can cause a
                    // prober to read the wrong key, miss the existing entry,
                    // and insert a duplicate.
                    //
                    // seq_cst generates `fence.ugm.global.all` on Intel Xe
                    // (flush+invalidate L1). A release fence only generates
                    // `fence.ugm.global.none` and is insufficient.
                    //
                    // Cost: once per work-item at kernel start. No dirty L1
                    // lines exist at this point, so only the invalidation
                    // actually runs — negligible overhead.
                    sycl::atomic_fence(sycl::memory_order::seq_cst,
                                       sycl::memory_scope::device);

                    const Key key = keys[tid];
                    output_buf_indices[tid] = 0;
                    output_masks[tid] = false;

                    const int64_t mask = bucket_count - 1;
                    const uint64_t hash = HashMix(hash_fn(key));
                    const int64_t home = static_cast<int64_t>(hash & mask);
                    const uint16_t my_fingerprint =
                            static_cast<uint16_t>((hash >> 16) & 0xffffULL);

                    // Outer loop restarts when a claim race is lost or a
                    // LOCKED slot is encountered. The hard limit prevents a
                    // future regression from causing a GPU watchdog timeout.
                    int outer_iter = 0;
                    while (outer_iter++ < kMaxOuterIter) {
                        int64_t first_deleted = -1;
                        bool finished = false;
                        bool restart = false;

                        for (int64_t probe = 0; probe < bucket_count; ++probe) {
                            const int64_t idx = (home + probe) & mask;
                            sycl::atomic_ref<uint64_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device>
                                    st(slot_data[idx]);

                            uint64_t packed =
                                    st.load(sycl::memory_order::acquire);
                            uint32_t s;
                            buf_index_t bi;
                            uint16_t fp;
                            UnpackSlot(packed, s, bi, fp);

                            // A concurrent writer holds this slot in the
                            // LOCKED state. Spinning here is unsafe: if
                            // this thread and the lock holder share the
                            // same SIMD subgroup, their lanes execute in
                            // lockstep — the spinning lane would prevent
                            // the lock-holding lane from completing its
                            // write and releasing the lock (subgroup-level
                            // deadlock → GPU hang → UR_RESULT_ERROR_DEVICE_LOST).
                            //
                            // Instead restart the probe from home. The
                            // lock holder publishes in a few GPU
                            // instructions, so the next pass will see
                            // OCCUPIED and find the key as a duplicate.
                            if (s == kSYCLSlotLocked) {
                                restart = true;
                                break;
                            }

                            if (s == kSYCLSlotOccupied) {
                                // Fingerprint guard: only dereference the
                                // external key buffer when the 16-bit hash
                                // extract matches (false-positive ~1/65536).
                                if (fp == my_fingerprint) {
                                    // Invalidate this EU's L1 before reading
                                    // the key. On Intel Xe GPU an acquire
                                    // load of the slot does NOT invalidate
                                    // other L1 lines. If this thread has a
                                    // stale L1 entry for GetKeyPtr(bi) (e.g.
                                    // from a previous round after Clear()),
                                    // the non-atomic key read below would
                                    // return the old cached value, causing a
                                    // false-miss and a duplicate insertion.
                                    // seq_cst generates fence.ugm.global.all
                                    // which flushes+invalidates the EU's L1,
                                    // so the key load fetches fresh data from
                                    // device-coherent L2.
                                    sycl::atomic_fence(
                                            sycl::memory_order::seq_cst,
                                            sycl::memory_scope::device);
                                    const Key* slot_key =
                                            static_cast<const Key*>(
                                                    accessor.GetKeyPtr(bi));
                                    if (eq_fn(*slot_key, key)) {
                                        // Key already present: report
                                        // existing slot, no insertion.
                                        output_buf_indices[tid] = bi;
                                        output_masks[tid] = false;
                                        finished = true;
                                        break;
                                    }
                                }
                                continue;
                            }

                            if (s == kSYCLSlotDeleted) {
                                // Remember the first tombstone for
                                // Robin-Hood-style reuse.
                                if (first_deleted < 0) first_deleted = idx;
                                continue;
                            }

                            // s == kSYCLSlotEmpty: key is not present.
                            // Claim the first tombstone seen (if any) to
                            // reuse it and shorten future probe chains;
                            // otherwise claim this empty slot.
                            const int64_t target =
                                    (first_deleted >= 0) ? first_deleted : idx;
                            uint64_t desired_packed =
                                    PackSlot(kSYCLSlotLocked, 0, 0);

                            sycl::atomic_ref<uint64_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device>
                                    tst(slot_data[target]);
                            // Re-read the target slot atomically. For a
                            // tombstone (first_deleted >= 0) its contents
                            // may have changed since we first probed it;
                            // a plain (non-atomic) read would be a data
                            // race. For an empty slot (first_deleted < 0)
                            // the current value is guaranteed to be 0
                            // (kSYCLSlotEmpty == 0) — skip the load.
                            uint64_t expected_packed =
                                    (first_deleted >= 0)
                                            ? tst.load(
                                                      sycl::memory_order::
                                                              acquire)
                                            : 0ULL;
                            if (tst.compare_exchange_strong(
                                        expected_packed, desired_packed,
                                        sycl::memory_order::acq_rel,
                                        sycl::memory_order::relaxed)) {
                                const buf_index_t new_bi =
                                        accessor.DeviceAllocate();

                                // Heap bounds guard: if the heap is
                                // somehow exhausted, release the lock
                                // (restore EMPTY) and exit cleanly rather
                                // than writing past the end of the buffer.
                                if (new_bi >=
                                    static_cast<buf_index_t>(capacity)) {
                                    tst.store(0ULL,
                                              sycl::memory_order::release);
                                    finished = true;
                                    break;
                                }

                                Key* slot_key = static_cast<Key*>(
                                        accessor.GetKeyPtr(new_bi));
                                *slot_key = key;

                                // Copy values in aligned blocks.
                                // common_block_size is uniform (same for all
                                // work-items), so the if/else inside DISPATCH
                                // is a uniform branch: only one path runs per
                                // kernel invocation with no lane divergence.
                                // reinterpret_cast is required here; the
                                // static_cast from void* to sycl::vec* is
                                // undefined behaviour.
                                for (int j = 0; j < n_values; ++j) {
                                    const int64_t blocks =
                                            accessor.value_blocks_per_element_
                                                    [j];
                                    DISPATCH_DIVISOR_SIZE_TO_BLOCK_T_SYCL(
                                            common_block_size, [&]() {
                                                using val_block_t = block_t;
                                                val_block_t* dst =
                                                        reinterpret_cast<
                                                                val_block_t*>(
                                                                accessor
                                                                        .GetValuePtr(
                                                                                new_bi,
                                                                                j));
                                                const val_block_t* src =
                                                        reinterpret_cast<
                                                                const val_block_t*>(
                                                                values_soa
                                                                        .ptrs[j]) +
                                                        blocks * tid;
                                                for (int64_t b = 0; b < blocks;
                                                     ++b) {
                                                    dst[b] = src[b];
                                                }
                                            });
                                }

                                // Flush key/value writes from per-EU L1
                                // write-back cache to device-coherent L2
                                // before publishing the slot as OCCUPIED.
                                //
                                // Background: on Intel Xe GPU, non-atomic
                                // stores (key/value above) land in the EU's
                                // private write-back L1 cache, not in the
                                // shared L2. The subsequent release store
                                // (tst.store below) only flushes the
                                // write-combining buffer (WCB) — it does NOT
                                // write back dirty L1 lines to L2. A probing
                                // thread on a different EU therefore reads
                                // stale zeros for the key via L2, causing a
                                // false "not-found" and duplicate insertion.
                                //
                                // sycl::atomic_fence(seq_cst, device)
                                // generates the IGC instruction
                                // `fence.ugm.global.all`, which writes back
                                // AND invalidates the entire EU's L1, forcing
                                // key/value data into L2 before the slot
                                // is published. seq_cst is required; a
                                // release fence only generates
                                // `fence.ugm.global.none` (WCB-flush only).
                                sycl::atomic_fence(
                                        sycl::memory_order::seq_cst,
                                        sycl::memory_scope::device);

                                // Publish: probers that load this slot with
                                // acquire will now see correct key/value data.
                                uint64_t final_packed =
                                        PackSlot(kSYCLSlotOccupied, new_bi,
                                                 my_fingerprint);
                                tst.store(final_packed,
                                          sycl::memory_order::release);

                                output_buf_indices[tid] = new_bi;
                                output_masks[tid] = true;

                                sycl::atomic_ref<int,
                                                 sycl::memory_order::relaxed,
                                                 sycl::memory_scope::device>
                                        oc(*occupied_count);
                                oc.fetch_add(1);

                                // non_empty_count only increments for a
                                // new EMPTY → OCCUPIED transition; tombstone
                                // → OCCUPIED reuse keeps the existing count.
                                const uint32_t prev_state =
                                        static_cast<uint32_t>(
                                                (expected_packed >> 32) &
                                                0xfULL);
                                if (prev_state == kSYCLSlotEmpty) {
                                    sycl::atomic_ref<
                                            int, sycl::memory_order::relaxed,
                                            sycl::memory_scope::device>
                                            nec(*non_empty_count);
                                    nec.fetch_add(1);
                                }

                                finished = true;
                                break;
                            }
                            // Lost the claim race: restart the probe from
                            // home to re-check for a duplicate.
                            restart = true;
                            break;
                        }

                        if (finished || !restart) break;
                    }
                });
    queue_.wait_and_throw();
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

    queue_.parallel_for(count, [=](int64_t tid) {
              const Key key = keys[tid];
              const int64_t mask = bucket_count - 1;
              const uint64_t hash = HashMix(hash_fn(key));
              const int64_t home = static_cast<int64_t>(hash & mask);
              const uint16_t my_fingerprint =
                      static_cast<uint16_t>((hash >> 16) & 0xffffULL);

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
                  uint16_t fp;
                  UnpackSlot(packed, s, bi, fp);

                  if (s == kSYCLSlotEmpty) {
                      break;  // Probe chain ended; key is not in the table.
                  }
                  // kSYCLSlotLocked: slot is being written by a concurrent
                  // Insert. Continue probing — the key hasn't been published
                  // yet, so it is not visible to this Find. Spinning would
                  // risk a subgroup deadlock (see Insert's LOCKED comment).
                  if (s == kSYCLSlotOccupied && fp == my_fingerprint) {
                      const Key* slot_key =
                              static_cast<const Key*>(accessor.GetKeyPtr(bi));
                      if (eq_fn(*slot_key, key)) {
                          found = true;
                          result = bi;
                          break;
                      }
                  }
                  // kSYCLSlotDeleted or fingerprint mismatch: continue probing.
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
    uint64_t* slot_data = slot_data_;
    const int64_t bucket_count = bucket_count_;
    int* occupied_count = occupied_count_;
    Hash hash_fn;
    Eq eq_fn;

    queue_.parallel_for(count, [=](int64_t tid) {
              const Key key = keys[tid];
              const int64_t mask = bucket_count - 1;
              const uint64_t hash = HashMix(hash_fn(key));
              const int64_t home = static_cast<int64_t>(hash & mask);
              const uint16_t my_fingerprint =
                      static_cast<uint16_t>((hash >> 16) & 0xffffULL);

              bool erased = false;
              for (int64_t probe = 0; probe < bucket_count; ++probe) {
                  const int64_t idx = (home + probe) & mask;
                  sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                          st(slot_data[idx]);
                  uint64_t packed = st.load(sycl::memory_order::acquire);
                  uint32_t s;
                  buf_index_t bi;
                  uint16_t fp;
                  UnpackSlot(packed, s, bi, fp);

                  if (s == kSYCLSlotEmpty) {
                      break;  // Probe chain ended; key is not in the table.
                  }
                  // kSYCLSlotLocked: continue probing (same reasoning as Find).
                  if (s == kSYCLSlotOccupied && fp == my_fingerprint) {
                      const Key* slot_key =
                              static_cast<const Key*>(accessor.GetKeyPtr(bi));
                      if (eq_fn(*slot_key, key)) {
                          // Transition OCCUPIED -> DELETED atomically. Keep the
                          // buf_index and fingerprint in the slot so that any
                          // concurrent probe on the same chain can still read a
                          // consistent (now-stale) slot without ABA confusion.
                          uint64_t expected_packed = packed;
                          uint64_t deleted_val =
                                  PackSlot(kSYCLSlotDeleted, bi, fp);
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
                              // non_empty_count is not decremented: tombstones
                              // remain "non-empty" for capacity-guard purposes.
                          }
                          break;  // Found the key; done regardless of CAS
                                  // result.
                      }
                  }
                  // kSYCLSlotDeleted or fingerprint mismatch: continue probing.
              }
              output_masks[tid] = erased;
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

    // Use workgroup-local exclusive scan to aggregate occupied buffer indices:
    // each work-group atomically claims a contiguous output range with a single
    // fetch_add on d_count, then scatters its occupied entries. This scales
    // much better than one global atomic per occupied slot (the original
    // design) for large tables where many work-groups find occupied entries
    // concurrently.
    constexpr int64_t kWgSize = 256;
    int64_t global_size = ((bucket_count + kWgSize - 1) / kWgSize) * kWgSize;

    queue_.submit([&](sycl::handler& cgh) {
              cgh.parallel_for(
                      sycl::nd_range<1>(global_size, kWgSize),
                      [=](sycl::nd_item<1> item) {
                          int64_t idx = item.get_global_id(0);
                          auto group = item.get_group();

                          bool is_occupied = false;
                          buf_index_t bi = 0;
                          if (idx < bucket_count) {
                              sycl::atomic_ref<uint64_t,
                                               sycl::memory_order::relaxed,
                                               sycl::memory_scope::device>
                                      st(slot_data[idx]);
                              uint64_t packed =
                                      st.load(sycl::memory_order::acquire);
                              uint32_t s = static_cast<uint32_t>(
                                      (packed >> 32) & 0xfULL);
                              if (s == kSYCLSlotOccupied) {
                                  is_occupied = true;
                                  bi = static_cast<buf_index_t>(packed &
                                                                0xffffffffULL);
                              }
                          }

                          // Intra-group prefix sum: each thread knows its local
                          // output offset within the group's contiguous block.
                          int local_val = is_occupied ? 1 : 0;
                          int local_offset = sycl::exclusive_scan_over_group(
                                  group, local_val, sycl::plus<int>{});
                          int group_total = sycl::reduce_over_group(
                                  group, local_val, sycl::plus<int>{});

                          // Thread 0 atomically reserves the group's output
                          // range and broadcasts the start offset.
                          int group_start = 0;
                          if (item.get_local_id(0) == 0 && group_total > 0) {
                              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                               sycl::memory_scope::device>
                                      counter(*d_count);
                              group_start = counter.fetch_add(group_total);
                          }
                          group_start =
                                  sycl::group_broadcast(group, group_start, 0);

                          if (is_occupied) {
                              output_indices[group_start + local_offset] = bi;
                          }
                      });
          }).wait_and_throw();

    int count = 0;
    MemoryManager::MemcpyToHost(&count, d_count, this->device_, sizeof(int));
    MemoryManager::Free(d_count, this->device_);
    return static_cast<int64_t>(count);
}

template <typename Key, typename Hash, typename Eq>
void SYCLHashBackend<Key, Hash, Eq>::Clear() {
    this->buffer_->ResetHeap();
    // slot_data is zeroed entirely: all slots revert to EMPTY (== 0).
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
    // Round up to the next power of two for cheap masking in probe loops.
    bucket_count_ = NextPowerOfTwo(std::max<int64_t>(capacity * 2, 1));

    // Allocate the shared key/value buffer.
    this->buffer_ = std::make_shared<HashBackendBuffer>(
            this->capacity_, this->key_dsize_, this->value_dsizes_,
            this->device_);
    buffer_accessor_.Setup(*this->buffer_);

    // Allocate and zero-initialize the slot array. EMPTY == 0, so memset to
    // zero correctly initialises all slots as empty.
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
