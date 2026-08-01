// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"

#if defined(SYCL_LANGUAGE_VERSION)
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SYCLUtils.h"
#endif

namespace open3d {
namespace core {

void Reduction(benchmark::State& state, const Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    Tensor src(shape, core::Int64, device);
    Tensor warm_up = src.Sum({1});
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = src.Sum({1});
#ifdef BUILD_CUDA_MODULE
        if (device.IsCUDA()) {
            cuda::Synchronize(device);
        }
#endif
#if defined(SYCL_LANGUAGE_VERSION)
        if (device.IsSYCL()) {
            core::sy::SYCLContext::GetInstance().GetDefaultQueue(device).wait();
        }
#endif
    }
}

BENCHMARK_CAPTURE(Reduction, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Reduction, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

#if defined(SYCL_LANGUAGE_VERSION)
BENCHMARK_CAPTURE(Reduction, SYCL, Device("SYCL:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace open3d
