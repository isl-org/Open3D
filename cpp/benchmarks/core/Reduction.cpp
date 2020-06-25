// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"

#include <benchmark/benchmark.h>

namespace open3d {
namespace core {

static void ReductionCPU(benchmark::State& state) {
    Device device("CPU:0");
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    Tensor src(shape, Dtype::Int64, device);
    Tensor warm_up = src.Sum({1});
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = src.Sum({1});
    }
}

// Fixture does play very well with static initialization in Open3D. Use the
// simple BENCHMARK here.
// https://github.com/google/benchmark/issues/498
BENCHMARK(ReductionCPU)->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE

static void ReductionCUDA(benchmark::State& state) {
    Device device("CUDA:0");
    int64_t large_dim = (1ULL << 27) + 10;
    SizeVector shape{2, large_dim};
    Tensor src(shape, Dtype::Int64, device);
    Tensor warm_up = src.Sum({1});
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = src.Sum({1});
    }
}

BENCHMARK(ReductionCUDA)->Unit(benchmark::kMillisecond);

#endif

}  // namespace core
}  // namespace open3d
