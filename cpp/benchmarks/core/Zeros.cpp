// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <benchmark/benchmark.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

void Zeros(benchmark::State& state, const Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};

    Tensor warm_up = Tensor::Zeros(shape, core::Float32, device);
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = Tensor::Zeros(shape, core::Float32, device);
        cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(Zeros, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Zeros, CUDA, Device("CUDA:0"))->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace open3d
