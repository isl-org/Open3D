// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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
        cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(Reduction, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Reduction, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace open3d
