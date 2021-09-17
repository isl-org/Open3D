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

#include "open3d/core/ParallelFor.h"

#include <benchmark/benchmark.h>

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
