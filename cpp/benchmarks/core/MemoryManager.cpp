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

#include "open3d/core/MemoryManager.h"

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

enum class MemoryManagerBackend { Direct, Cached };

std::shared_ptr<MemoryManagerDevice> MakeMemoryManager(
        const Device& device, const MemoryManagerBackend& backend) {
    std::shared_ptr<MemoryManagerDevice> device_mm;
    switch (device.GetType()) {
        case Device::DeviceType::CPU:
            device_mm = std::make_shared<MemoryManagerCPU>();
            break;
#ifdef BUILD_CUDA_MODULE
        case Device::DeviceType::CUDA:
            device_mm = std::make_shared<MemoryManagerCUDA>();
            break;
#endif
        default:
            utility::LogError("Unimplemented device {}.", device.ToString());
            break;
    }

    switch (backend) {
        case MemoryManagerBackend::Direct:
            return device_mm;
        case MemoryManagerBackend::Cached:
            return std::make_shared<MemoryManagerCached>(device_mm);
        default:
            utility::LogError("Unimplemented backend.");
            break;
    }
}

void Malloc(benchmark::State& state,
            int size,
            const Device& device,
            const MemoryManagerBackend& backend) {
    MemoryManagerCached::ReleaseCache(device);

    auto device_mm = MakeMemoryManager(device, backend);

    // Warmup.
    {
        void* ptr = device_mm->Malloc(size, device);
        device_mm->Free(ptr, device);
        cuda::Synchronize(device);
    }

    for (auto _ : state) {
        void* ptr = device_mm->Malloc(size, device);
        cuda::Synchronize(device);

        state.PauseTiming();
        device_mm->Free(ptr, device);
        cuda::Synchronize(device);
        state.ResumeTiming();
    }

    MemoryManagerCached::ReleaseCache(device);
}

void Free(benchmark::State& state,
          int size,
          const Device& device,
          const MemoryManagerBackend& backend) {
    MemoryManagerCached::ReleaseCache(device);

    auto device_mm = MakeMemoryManager(device, backend);

    // Warmup.
    {
        void* ptr = device_mm->Malloc(size, device);
        device_mm->Free(ptr, device);
        cuda::Synchronize(device);
    }

    for (auto _ : state) {
        state.PauseTiming();
        void* ptr = device_mm->Malloc(size, device);
        cuda::Synchronize(device);
        state.ResumeTiming();

        device_mm->Free(ptr, device);
        cuda::Synchronize(device);
    }

    MemoryManagerCached::ReleaseCache(device);
}

#define ENUM_BM_SIZE(FN, DEVICE, DEVICE_NAME, BACKEND)                         \
    BENCHMARK_CAPTURE(FN, BACKEND##_100_##DEVICE_NAME, 100, DEVICE, BACKEND)   \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_1000_##DEVICE_NAME, 1000, DEVICE, BACKEND) \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_10000_##DEVICE_NAME, 10000, DEVICE,        \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_100000_##DEVICE_NAME, 100000, DEVICE,      \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_1000000_##DEVICE_NAME, 1000000, DEVICE,    \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_10000000_##DEVICE_NAME, 10000000, DEVICE,  \
                      BACKEND)                                                 \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_100000000_##DEVICE_NAME, 100000000,        \
                      DEVICE, BACKEND)                                         \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, BACKEND##_1000000000_##DEVICE_NAME, 1000000000,      \
                      DEVICE, BACKEND)                                         \
            ->Unit(benchmark::kMicrosecond);

#ifdef BUILD_CUDA_MODULE
#define ENUM_BM_BACKEND(FN)                                                \
    ENUM_BM_SIZE(FN, Device("CPU:0"), CPU, MemoryManagerBackend::Direct)   \
    ENUM_BM_SIZE(FN, Device("CPU:0"), CPU, MemoryManagerBackend::Cached)   \
    ENUM_BM_SIZE(FN, Device("CUDA:0"), CUDA, MemoryManagerBackend::Direct) \
    ENUM_BM_SIZE(FN, Device("CUDA:0"), CUDA, MemoryManagerBackend::Cached)
#else
#define ENUM_BM_BACKEND(FN)                                              \
    ENUM_BM_SIZE(FN, Device("CPU:0"), CPU, MemoryManagerBackend::Direct) \
    ENUM_BM_SIZE(FN, Device("CPU:0"), CPU, MemoryManagerBackend::Cached)
#endif

ENUM_BM_BACKEND(Malloc)
ENUM_BM_BACKEND(Free)

}  // namespace core
}  // namespace open3d
