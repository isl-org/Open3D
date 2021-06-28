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

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

void Inverse3x3(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}}, device);

    Tensor warm_up = A.Inverse();
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = A.Inverse();
    }
}

void Inverse4x4(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3, 5}, {0, 1, 2, 4}, {5, 6, 0, 7}, {2, 4, 6, 8}}, device);

    Tensor warm_up = A.Inverse();
    (void)warm_up;
    for (auto _ : state) {
        Tensor dst = A.Inverse();
    }
}

void Det3x3(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}}, device);

    double det = A.Det();
    (void)det;
    for (auto _ : state) {
        det = A.Det();
    }
}

void Det4x4(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3, 5}, {0, 1, 2, 4}, {5, 6, 0, 7}, {2, 4, 6, 8}}, device);

    double det = A.Det();
    (void)det;
    for (auto _ : state) {
        det = A.Det();
    }
}

void SVD3x3(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}}, device);

    auto warm_up = A.SVD();
    (void)warm_up;
    for (auto _ : state) {
        warm_up = A.SVD();
    }
}

void SVD4x4(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3, 5}, {0, 1, 2, 4}, {5, 6, 0, 7}, {2, 4, 6, 8}}, device);

    auto warm_up = A.SVD();
    (void)warm_up;
    for (auto _ : state) {
        warm_up = A.SVD();
    }
}

void Solve3x3(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3}, {0, 1, 4}, {5, 6, 0}}, device);

    core::Tensor b = core::Tensor::Init<float>({5, 6, 2}, device);

    auto warm_up = A.Solve(b);
    (void)warm_up;
    for (auto _ : state) {
        warm_up = A.Solve(b);
    }
}

void Solve4x4(benchmark::State& state, const Device& device) {
    core::Tensor A = core::Tensor::Init<float>(
            {{1, 2, 3, 5}, {0, 1, 2, 4}, {5, 6, 0, 7}, {2, 4, 6, 8}}, device);
    core::Tensor b = core::Tensor::Init<float>({5, 6, 2, 8}, device);

    auto warm_up = A.Solve(b);
    (void)warm_up;
    for (auto _ : state) {
        warm_up = A.Solve(b);
    }
}

BENCHMARK_CAPTURE(Inverse3x3, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Inverse3x3, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(Inverse4x4, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Inverse4x4, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(Det3x3, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Det3x3, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(Det4x4, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Det4x4, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(SVD3x3, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(SVD3x3, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(SVD4x4, CPU, Device("CPU:0"))->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(SVD4x4, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(Solve3x3, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Solve3x3, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_CAPTURE(Solve4x4, CPU, Device("CPU:0"))
        ->Unit(benchmark::kMillisecond);

#ifdef BUILD_CUDA_MODULE
BENCHMARK_CAPTURE(Solve4x4, CUDA, Device("CUDA:0"))
        ->Unit(benchmark::kMillisecond);
#endif

}  // namespace core
}  // namespace open3d
