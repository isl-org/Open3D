// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/ParallelFor.h"

#include "open3d/core/Tensor.h"
#if defined(SYCL_LANGUAGE_VERSION)
#include "open3d/core/SYCLContext.h"
#include "open3d/core/SYCLUtils.h"
#endif

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

#if defined(SYCL_LANGUAGE_VERSION)
void ParallelForSYCL(benchmark::State& state, int size) {
    core::Device device("SYCL:0");
    if (!core::sy::IsDeviceAvailable(device)) {
        state.SkipWithError("SYCL device not available");
        return;
    }
    core::Tensor input = core::Tensor::Ones({size}, core::Float32, device);
    core::Tensor output = core::Tensor::Zeros({size}, core::Float32, device);
    float* input_ptr = input.GetDataPtr<float>();
    float* output_ptr = output.GetDataPtr<float>();

    // Warmup.
    {
        core::ParallelFor(device, size, [=](int64_t idx) {
            float x = input_ptr[idx];
            float x2 = x * x;
            output_ptr[idx] = x2;
        });
        core::sy::SYCLContext::GetInstance().GetDefaultQueue(device).wait();
    }

    for (auto _ : state) {
        core::ParallelFor(device, size, [=](int64_t idx) {
            float x = input_ptr[idx];
            float x2 = x * x;
            output_ptr[idx] = x2;
        });
        core::sy::SYCLContext::GetInstance().GetDefaultQueue(device).wait();
    }
}

#define ENUM_BM_SIZE_SYCL(FN)                                                 \
    BENCHMARK_CAPTURE(FN, SYCL##100, 100)->Unit(benchmark::kMicrosecond);     \
    BENCHMARK_CAPTURE(FN, SYCL##1000, 1000)->Unit(benchmark::kMicrosecond);   \
    BENCHMARK_CAPTURE(FN, SYCL##10000, 10000)->Unit(benchmark::kMicrosecond); \
    BENCHMARK_CAPTURE(FN, SYCL##100000, 100000)                               \
            ->Unit(benchmark::kMicrosecond);                                  \
    BENCHMARK_CAPTURE(FN, SYCL##1000000, 1000000)                             \
            ->Unit(benchmark::kMicrosecond);                                  \
    BENCHMARK_CAPTURE(FN, SYCL##10000000, 10000000)                           \
            ->Unit(benchmark::kMicrosecond);

ENUM_BM_SIZE_SYCL(ParallelForSYCL)
#endif

}  // namespace core
}  // namespace open3d
