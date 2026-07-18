// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Hyperparameter sweep for the SYCL hash backend (see
// cpp/open3d/core/hashmap/SYCL/SYCLHashBackend.h): kernel work-group size
// (wg_size, in Insert/Find/Erase GetActiveIndices).
//
// bucket_count_multiplier is fixed at 2 (target load factor 0.5) in
// SYCLHashBackend to keep slot_data_ memory bounded.

#ifdef BUILD_SYCL_MODULE

#include <benchmark/benchmark.h>

#include <numeric>
#include <random>
#include <vector>

#include "open3d/core/Device.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/hashmap/HashBackendBuffer.h"
#include "open3d/core/hashmap/SYCL/SYCLHashBackend.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace benchmarks {
namespace core_hashmap {

using KeyT = utility::MiniVec<int, 1>;
using HashT = utility::MiniVecHash<int, 1>;
using EqT = utility::MiniVecEq<int, 1>;
using BackendT = core::SYCLHashBackend<KeyT, HashT, EqT>;

enum class HashOp { kInsert, kFind, kErase };

namespace {

// Host-side key/value generation, mirrors HashData in
// benchmarks/core/HashMap.cpp: duplicate_factor controls how many distinct
// "slots" the capacity-many keys collide into (duplicate_factor=1 -> all
// unique keys, the harder case for probe-chain length).
struct KeyValueData {
    KeyValueData(int64_t count, int64_t slots) {
        keys.resize(count);
        vals.resize(count);
        std::vector<int64_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine(0));
        for (int64_t i = 0; i < count; ++i) {
            int64_t v = indices[i] % slots;
            keys[i] = KeyT(static_cast<int>(v * kKeyFactor));
            vals[i] = static_cast<int>(v);
        }
    }
    static constexpr int kKeyFactor = 101;
    std::vector<KeyT> keys;
    std::vector<int> vals;
};

template <typename T>
T* ToDevice(const std::vector<T>& host, const core::Device& device) {
    T* dptr = static_cast<T*>(
            core::MemoryManager::Malloc(host.size() * sizeof(T), device));
    core::MemoryManager::MemcpyFromHost(dptr, device, host.data(),
                                        host.size() * sizeof(T));
    return dptr;
}

// Args (via ->ArgsProduct, read with state.range(i)):
//   0: wg_size
// capacity/duplicate_factor/op are fixed per BENCHMARK_CAPTURE call
// (captured, not swept).
void HashMapSYCLTune(benchmark::State& state,
                     int64_t capacity,
                     int64_t duplicate_factor,
                     HashOp op) {
    const int64_t wg_size = state.range(0);

    const core::Device device("SYCL:0");
    const int64_t slots = std::max<int64_t>(1, capacity / duplicate_factor);
    KeyValueData data(capacity, slots);

    KeyT* d_keys = ToDevice(data.keys, device);
    int* d_vals = ToDevice(data.vals, device);
    core::buf_index_t* d_buf_indices =
            static_cast<core::buf_index_t*>(core::MemoryManager::Malloc(
                    capacity * sizeof(core::buf_index_t), device));
    bool* d_masks = static_cast<bool*>(
            core::MemoryManager::Malloc(capacity * sizeof(bool), device));

    auto make_backend = [&]() {
        return std::make_unique<BackendT>(capacity, sizeof(KeyT),
                                          std::vector<int64_t>{sizeof(int)},
                                          device, wg_size);
    };

    auto run_insert = [&](BackendT& backend) {
        backend.Insert(d_keys, {d_vals}, d_buf_indices, d_masks, capacity);
    };

    try {
        // Warm-up; surfaces invalid wg_size combinations (e.g. device launch
        // failures) early.
        auto warmup = make_backend();
        run_insert(*warmup);
        if (op == HashOp::kFind) {
            warmup->Find(d_keys, d_buf_indices, d_masks, capacity);
        } else if (op == HashOp::kErase) {
            warmup->Erase(d_keys, d_masks, capacity);
            run_insert(*warmup);  // Restore state for the erase timing loop.
        }
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
        core::MemoryManager::Free(d_keys, device);
        core::MemoryManager::Free(d_vals, device);
        core::MemoryManager::Free(d_buf_indices, device);
        core::MemoryManager::Free(d_masks, device);
        return;
    }

    if (op == HashOp::kInsert) {
        for (auto _ : state) {
            state.PauseTiming();
            auto backend = make_backend();
            state.ResumeTiming();
            run_insert(*backend);
        }
    } else if (op == HashOp::kFind) {
        auto backend = make_backend();
        run_insert(*backend);
        for (auto _ : state) {
            backend->Find(d_keys, d_buf_indices, d_masks, capacity);
        }
    } else {
        for (auto _ : state) {
            state.PauseTiming();
            auto backend = make_backend();
            run_insert(*backend);
            state.ResumeTiming();
            backend->Erase(d_keys, d_masks, capacity);
        }
    }

    core::MemoryManager::Free(d_keys, device);
    core::MemoryManager::Free(d_vals, device);
    core::MemoryManager::Free(d_buf_indices, device);
    core::MemoryManager::Free(d_masks, device);
}

void HashMapSYCLTuneInsert(benchmark::State& state,
                           int64_t capacity,
                           int64_t duplicate_factor) {
    HashMapSYCLTune(state, capacity, duplicate_factor, HashOp::kInsert);
}

void HashMapSYCLTuneFind(benchmark::State& state,
                         int64_t capacity,
                         int64_t duplicate_factor) {
    HashMapSYCLTune(state, capacity, duplicate_factor, HashOp::kFind);
}

void HashMapSYCLTuneErase(benchmark::State& state,
                          int64_t capacity,
                          int64_t duplicate_factor) {
    HashMapSYCLTune(state, capacity, duplicate_factor, HashOp::kErase);
}

// 1M-entry hashmap, mostly-unique keys (duplicate_factor=1), sweeping
// wg_size for each of Insert/Find/Erase.
BENCHMARK_CAPTURE(HashMapSYCLTuneInsert, 1M_dup1, 1000000, 1)
        ->ArgsProduct({{32, 64, 128, 256, 512, 1024}})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(HashMapSYCLTuneFind, 1M_dup1, 1000000, 1)
        ->ArgsProduct({{32, 64, 128, 256, 512, 1024}})
        ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(HashMapSYCLTuneErase, 1M_dup1, 1000000, 1)
        ->ArgsProduct({{32, 64, 128, 256, 512, 1024}})
        ->Unit(benchmark::kMillisecond);

}  // namespace

}  // namespace core_hashmap
}  // namespace benchmarks
}  // namespace open3d

#endif  // BUILD_SYCL_MODULE
