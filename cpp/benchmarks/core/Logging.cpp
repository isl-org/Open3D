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

#include "open3d/utility/Logging.h"

#include <benchmark/benchmark.h>

namespace open3d {
namespace core {

void BenchmarkLogDebugEmpty(benchmark::State& state,
                            const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogDebug("");
    for (auto _ : state) {
        utility::LogDebug("");
    }
}

void BenchmarkLogDebugSmallString(benchmark::State& state,
                                  const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogDebug("Random Debug Info.");
    for (auto _ : state) {
        utility::LogDebug("Random Debug Info.");
    }
}

void BenchmarkLogDebugLongString(benchmark::State& state,
                                 const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogDebug(
            "Random Debug Info. This is a long string, to check the "
            "performance of LogDebug wrt to size of input string.");
    for (auto _ : state) {
        utility::LogDebug(
                "Random Debug Info. This is a long string, to check the "
                "performance of LogDebug wrt to size of input string.");
    }
}

void BenchmarkLogDebugWithLongStringAsInput(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    std::string test =
            "Random Debug Info. This is a long string, to check the "
            "performance of LogDebug wrt to size of input string.";
    utility::LogDebug("{}", test);
    for (auto _ : state) {
        utility::LogDebug("{}", test);
    }
}

void BenchmarkLogDebugWithDoubleAsInput(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    double x = 5.0;
    utility::LogDebug("Double value as input: {}", x);
    for (auto _ : state) {
        utility::LogDebug("Double value as input: {}", x);
    }
}

void BenchmarkLogDebugWithDoubleAsInputFmt(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    double x = 5.0;
    utility::LogDebug("Double value as input: {:f}", x);
    for (auto _ : state) {
        utility::LogDebug("Double value as input: {:f}", x);
    }
}

void BenchmarkLogWarningEmpty(benchmark::State& state,
                              const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogWarning("");
    for (auto _ : state) {
        utility::LogWarning("");
    }
}

void BenchmarkLogWarningSmallString(benchmark::State& state,
                                    const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogWarning("Random Debug Info.");
    for (auto _ : state) {
        utility::LogWarning("Random Debug Info.");
    }
}

void BenchmarkLogWarningLongString(benchmark::State& state,
                                   const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    utility::LogWarning(
            "Random Debug Info. This is a long string, to check the "
            "performance of LogDebug wrt to size of input string.");
    for (auto _ : state) {
        utility::LogWarning(
                "Random Debug Info. This is a long string, to check the "
                "performance of LogDebug wrt to size of input string.");
    }
}

void BenchmarkLogWarningWithLongStringAsInput(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    std::string test =
            "Random Debug Info. This is a long string, to check the "
            "performance of LogDebug wrt to size of input string.";
    utility::LogWarning("{}", test);
    for (auto _ : state) {
        utility::LogDebug("{}", test);
    }
}

void BenchmarkLogWarningWithDoubleAsInput(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    double x = 5.0;
    utility::LogWarning("Double value as input: {}", x);
    for (auto _ : state) {
        utility::LogWarning("Double value as input: {}", x);
    }
}

void BenchmarkLogWarningWithDoubleAsInputFmt(
        benchmark::State& state, const utility::VerbosityLevel verbosity) {
    utility::SetVerbosityLevel(verbosity);
    double x = 5.0;
    utility::LogWarning("Double value as input: {:f}", x);
    for (auto _ : state) {
        utility::LogWarning("Double value as input: {:f}", x);
    }
}

BENCHMARK_CAPTURE(BenchmarkLogDebugEmpty, Empty, utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugSmallString,
                  SmallString,
                  utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugLongString,
                  LongString,
                  utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithLongStringAsInput,
                  LongStringAsInput,
                  utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithDoubleAsInput,
                  Double,
                  utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithDoubleAsInputFmt,
                  DoubleFmt,
                  utility::VerbosityLevel::Info)
        ->Unit(benchmark::kNanosecond);

BENCHMARK_CAPTURE(BenchmarkLogDebugEmpty,
                  (W)Empty,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugSmallString,
                  (W)SmallString,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugLongString,
                  (W)LongString,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithLongStringAsInput,
                  (W)LongStringAsInput,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithDoubleAsInput,
                  (W)Double,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogDebugWithDoubleAsInputFmt,
                  DoubleFmt,
                  utility::VerbosityLevel::Warning)
        ->Unit(benchmark::kNanosecond);

BENCHMARK_CAPTURE(BenchmarkLogWarningEmpty,
                  Empty,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogWarningSmallString,
                  SmallString,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogWarningLongString,
                  LongString,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogWarningWithLongStringAsInput,
                  LongStringAsInput,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogWarningWithDoubleAsInput,
                  Double,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);
BENCHMARK_CAPTURE(BenchmarkLogWarningWithDoubleAsInputFmt,
                  DoubleFmt,
                  utility::VerbosityLevel::Error)
        ->Unit(benchmark::kNanosecond);

}  // namespace core
}  // namespace open3d
