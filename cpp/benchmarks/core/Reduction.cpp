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

#include <benchmark/benchmark.h>

#include "benchmarks/Benchmarks.h"
#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Kernel.h"

namespace open3d {
namespace benchmarks {

void Reduction(benchmark::State& state, const core::Device& device) {
    int64_t large_dim = (1ULL << 27) + 10;
    core::SizeVector shape{2, large_dim};
    core::Tensor src(shape, core::Int64, device);
    core::Tensor warm_up = src.Sum({1});
    (void)warm_up;
    for (auto _ : state) {
        core::Tensor dst = src.Sum({1});
        core::cuda::Synchronize(device);
    }
}

BENCHMARK_CAPTURE(Reduction, CPU, core::Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Reduction, CUDA, core::Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace benchmarks
}  // namespace open3d
