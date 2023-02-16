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

#include "open3d/core/hashmap/HashMap.h"

#include <benchmark/benchmark.h>

#include <numeric>
#include <random>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"

namespace open3d {
namespace core {

template <typename K, typename V>
class HashData {
public:
    HashData(int count, int slots) {
        keys_.resize(count);
        vals_.resize(count);

        std::vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(),
                     std::default_random_engine(0));

        // Ensure enough duplicates for harder tests
        for (int i = 0; i < count; ++i) {
            int v = indices[i] % slots;
            keys_[i] = K(v * k_factor_);
            vals_[i] = V(v);
        }
    }

public:
    const int k_factor_ = 101;
    std::vector<K> keys_;
    std::vector<V> vals_;
};

void HashInsertInt(benchmark::State& state,
                   int capacity,
                   int duplicate_factor,
                   const Device& device,
                   const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<int, int> data(capacity, slots);

    Tensor keys(data.keys_, {capacity}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {1}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {1}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Insert(keys, values, buf_indices, masks);

        cuda::Synchronize(device);
        state.PauseTiming();

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }
        state.ResumeTiming();
    }
}

void HashEraseInt(benchmark::State& state,
                  int capacity,
                  int duplicate_factor,
                  const Device& device,
                  const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<int, int> data(capacity, slots);

    Tensor keys(data.keys_, {capacity}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {1}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {1}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Erase(keys, masks);

        cuda::Synchronize(device);
        state.PauseTiming();

        int64_t s = hashmap.Size();
        if (s != 0) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.", 0,
                    s);
        }
        state.ResumeTiming();
    }
}

void HashFindInt(benchmark::State& state,
                 int capacity,
                 int duplicate_factor,
                 const Device& device,
                 const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<int, int> data(capacity, slots);

    Tensor keys(data.keys_, {capacity}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap(capacity, core::Int32, {1}, core::Int32, {1}, device,
                    backend);
    Tensor buf_indices, masks;
    // Insert as warp-up
    hashmap.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        hashmap.Find(keys, buf_indices, masks);
        cuda::Synchronize(device);
    }
}

void HashClearInt(benchmark::State& state,
                  int capacity,
                  int duplicate_factor,
                  const Device& device,
                  const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<int, int> data(capacity, slots);

    Tensor keys(data.keys_, {capacity}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {1}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {1}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        hashmap.Insert(keys, values, buf_indices, masks);

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Clear();

        cuda::Synchronize(device);
        state.PauseTiming();

        s = hashmap.Size();
        if (s != 0) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.", 0,
                    s);
        }
        state.ResumeTiming();
    }
}

void HashReserveInt(benchmark::State& state,
                    int capacity,
                    int duplicate_factor,
                    const Device& device,
                    const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<int, int> data(capacity, slots);

    Tensor keys(data.keys_, {capacity}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {1}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {1}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        hashmap.Insert(keys, values, buf_indices, masks);

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Reserve(2 * capacity);

        cuda::Synchronize(device);
        state.PauseTiming();

        s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }
        state.ResumeTiming();
    }
}

class Int3 {
public:
    Int3() : x_(0), y_(0), z_(0){};
    Int3(int k) : x_(k), y_(k * 2), z_(k * 4){};
    bool operator==(const Int3& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }
    int x_;
    int y_;
    int z_;
};

void HashInsertInt3(benchmark::State& state,
                    int capacity,
                    int duplicate_factor,
                    const Device& device,
                    const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<Int3, int> data(capacity, slots);

    std::vector<int> keys_Int3;
    keys_Int3.assign(reinterpret_cast<int*>(data.keys_.data()),
                     reinterpret_cast<int*>(data.keys_.data()) + 3 * capacity);
    Tensor keys(keys_Int3, {capacity, 3}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {3}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {3}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Insert(keys, values, buf_indices, masks);

        cuda::Synchronize(device);
        state.PauseTiming();

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }
        state.ResumeTiming();
    }
}

void HashEraseInt3(benchmark::State& state,
                   int capacity,
                   int duplicate_factor,
                   const Device& device,
                   const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<Int3, int> data(capacity, slots);

    std::vector<int> keys_Int3;
    keys_Int3.assign(reinterpret_cast<int*>(data.keys_.data()),
                     reinterpret_cast<int*>(data.keys_.data()) + 3 * capacity);
    Tensor keys(keys_Int3, {capacity, 3}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {3}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {3}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;
        hashmap.Insert(keys, values, buf_indices, masks);

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Erase(keys, masks);

        cuda::Synchronize(device);
        state.PauseTiming();

        int64_t s = hashmap.Size();
        if (s != 0) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.", 0,
                    s);
        }
        state.ResumeTiming();
    }
}

void HashFindInt3(benchmark::State& state,
                  int capacity,
                  int duplicate_factor,
                  const Device& device,
                  const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<Int3, int> data(capacity, slots);

    std::vector<int> keys_Int3;
    keys_Int3.assign(reinterpret_cast<int*>(data.keys_.data()),
                     reinterpret_cast<int*>(data.keys_.data()) + 3 * capacity);
    Tensor keys(keys_Int3, {capacity, 3}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap(capacity, core::Int32, {3}, core::Int32, {1}, device,
                    backend);
    Tensor buf_indices, masks;
    hashmap.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        hashmap.Find(keys, buf_indices, masks);
        cuda::Synchronize(device);
    }
}

void HashClearInt3(benchmark::State& state,
                   int capacity,
                   int duplicate_factor,
                   const Device& device,
                   const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<Int3, int> data(capacity, slots);

    std::vector<int> keys_Int3;
    keys_Int3.assign(reinterpret_cast<int*>(data.keys_.data()),
                     reinterpret_cast<int*>(data.keys_.data()) + 3 * capacity);
    Tensor keys(keys_Int3, {capacity, 3}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {3}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {3}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        hashmap.Insert(keys, values, buf_indices, masks);

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Clear();

        state.PauseTiming();
        cuda::Synchronize(device);

        s = hashmap.Size();
        if (s != 0) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.", 0,
                    s);
        }
        state.ResumeTiming();
    }
}

void HashReserveInt3(benchmark::State& state,
                     int capacity,
                     int duplicate_factor,
                     const Device& device,
                     const HashBackendType& backend) {
    int slots = std::max(1, capacity / duplicate_factor);
    HashData<Int3, int> data(capacity, slots);

    std::vector<int> keys_Int3;
    keys_Int3.assign(reinterpret_cast<int*>(data.keys_.data()),
                     reinterpret_cast<int*>(data.keys_.data()) + 3 * capacity);
    Tensor keys(keys_Int3, {capacity, 3}, core::Int32, device);
    Tensor values(data.vals_, {capacity}, core::Int32, device);

    HashMap hashmap_warmup(capacity, core::Int32, {3}, core::Int32, {1}, device,
                           backend);
    Tensor buf_indices, masks;
    hashmap_warmup.Insert(keys, values, buf_indices, masks);

    for (auto _ : state) {
        state.PauseTiming();
        HashMap hashmap(capacity, core::Int32, {3}, core::Int32, {1}, device,
                        backend);
        Tensor buf_indices, masks;

        hashmap.Insert(keys, values, buf_indices, masks);

        int64_t s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }

        cuda::Synchronize(device);
        state.ResumeTiming();

        hashmap.Reserve(2 * capacity);

        cuda::Synchronize(device);
        state.PauseTiming();

        s = hashmap.Size();
        if (s != slots) {
            utility::LogError(
                    "Error returning hashmap size, expected {}, but got {}.",
                    slots, s);
        }
        state.ResumeTiming();
    }
}

// Note: to enable large scale insertion (> 1M entries), change
// default_max_load_factor() in stdgpu from 1.0 to 1.2~1.4.
#define ENUM_BM_CAPACITY(FN, FACTOR, DEVICE, BACKEND)                          \
    BENCHMARK_CAPTURE(FN, BACKEND##_100_##FACTOR, 100, FACTOR, DEVICE,         \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_1000_##FACTOR, 1000, FACTOR, DEVICE,       \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_10000_##FACTOR, 10000, FACTOR, DEVICE,     \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_100000_##FACTOR, 100000, FACTOR, DEVICE,   \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMillisecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_1000000_##FACTOR, 1000000, FACTOR, DEVICE, \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMillisecond);

#define ENUM_BM_FACTOR(FN, DEVICE, BACKEND)   \
    ENUM_BM_CAPACITY(FN, 1, DEVICE, BACKEND)  \
    ENUM_BM_CAPACITY(FN, 2, DEVICE, BACKEND)  \
    ENUM_BM_CAPACITY(FN, 4, DEVICE, BACKEND)  \
    ENUM_BM_CAPACITY(FN, 8, DEVICE, BACKEND)  \
    ENUM_BM_CAPACITY(FN, 16, DEVICE, BACKEND) \
    ENUM_BM_CAPACITY(FN, 32, DEVICE, BACKEND)

#ifdef BUILD_CUDA_MODULE
#define ENUM_BM_BACKEND(FN)                                     \
    ENUM_BM_FACTOR(FN, Device("CPU:0"), HashBackendType::TBB)   \
    ENUM_BM_FACTOR(FN, Device("CUDA:0"), HashBackendType::Slab) \
    ENUM_BM_FACTOR(FN, Device("CUDA:0"), HashBackendType::StdGPU)
#else
#define ENUM_BM_BACKEND(FN) \
    ENUM_BM_FACTOR(FN, Device("CPU:0"), HashBackendType::TBB)
#endif

ENUM_BM_BACKEND(HashInsertInt)
ENUM_BM_BACKEND(HashInsertInt3)
ENUM_BM_BACKEND(HashEraseInt)
ENUM_BM_BACKEND(HashEraseInt3)
ENUM_BM_BACKEND(HashFindInt)
ENUM_BM_BACKEND(HashFindInt3)
ENUM_BM_BACKEND(HashClearInt)
ENUM_BM_BACKEND(HashClearInt3)
ENUM_BM_BACKEND(HashReserveInt)
ENUM_BM_BACKEND(HashReserveInt3)

}  // namespace core
}  // namespace open3d
