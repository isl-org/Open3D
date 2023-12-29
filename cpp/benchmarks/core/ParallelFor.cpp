// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/ParallelFor.h"

#include <benchmark/benchmark.h>

#include <numeric>
#include <vector>

#ifdef BUILD_ISPC_MODULE
#include "ParallelFor_ispc.h"
#endif

namespace open3d {
namespace core {

void ParallelForScalar(benchmark::State& state, int size) {
    std::vector<float> input(size);
    std::vector<float> output(size);
    std::iota(input.begin(), input.end(), 0.0f);

    // Warmup.
    {
        core::ParallelFor(core::Device("CPU:0"), size, [&](int64_t idx) {
            float x = input[idx];
            float x2 = x * x;
            output[idx] = x2;
        });
    }

    for (auto _ : state) {
        core::ParallelFor(core::Device("CPU:0"), size, [&](int64_t idx) {
            float x = input[idx];
            float x2 = x * x;
            output[idx] = x2;
        });
    }
}

void ParallelForVectorized(benchmark::State& state, int size) {
    std::vector<float> input(size);
    std::vector<float> output(size);
    std::iota(input.begin(), input.end(), 0.0f);

    // Warmup.
    {
        core::ParallelFor(
                core::Device("CPU:0"), size,
                [&](int64_t idx) {
                    float x = input[idx];
                    float x2 = x * x;
                    output[idx] = x2;
                },
                OPEN3D_VECTORIZED(SquareKernel, input.data(), output.data()));
    }

    for (auto _ : state) {
        core::ParallelFor(
                core::Device("CPU:0"), size,
                [&](int64_t idx) {
                    float x = input[idx];
                    float x2 = x * x;
                    output[idx] = x2;
                },
                OPEN3D_VECTORIZED(SquareKernel, input.data(), output.data()));
    }
}

#define ENUM_BM_SIZE(FN)                                                       \
    BENCHMARK_CAPTURE(FN, CPU##100, 100)->Unit(benchmark::kMicrosecond);       \
    BENCHMARK_CAPTURE(FN, CPU##1000, 1000)->Unit(benchmark::kMicrosecond);     \
    BENCHMARK_CAPTURE(FN, CPU##10000, 10000)->Unit(benchmark::kMicrosecond);   \
    BENCHMARK_CAPTURE(FN, CPU##100000, 100000)->Unit(benchmark::kMicrosecond); \
    BENCHMARK_CAPTURE(FN, CPU##1000000, 1000000)                               \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, CPU##10000000, 10000000)                             \
            ->Unit(benchmark::kMicrosecond);                                   \
    BENCHMARK_CAPTURE(FN, CPU##100000000, 100000000)                           \
            ->Unit(benchmark::kMicrosecond);

ENUM_BM_SIZE(ParallelForScalar)
ENUM_BM_SIZE(ParallelForVectorized)

}  // namespace core
}  // namespace open3d
