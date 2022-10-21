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

#include <type_traits>

#include "benchmarks/benchmark_utilities/Rand.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

enum class BinaryOpCode {
    Add,
    Sub,
    Mul,
    Div,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Neq,
};

static std::function<Tensor(const Tensor&, const Tensor&)> MakeOperation(
        BinaryOpCode op) {
    switch (op) {
        case BinaryOpCode::Add:
            return std::plus<Tensor>();

        case BinaryOpCode::Sub:
            return std::minus<Tensor>();

        case BinaryOpCode::Mul:
            return std::multiplies<Tensor>();

        case BinaryOpCode::Div:
            return std::divides<Tensor>();

        case BinaryOpCode::LogicalAnd:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs && rhs;
            };

        case BinaryOpCode::LogicalOr:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs || rhs;
            };

        case BinaryOpCode::LogicalXor:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs.LogicalXor(rhs);
            };

        case BinaryOpCode::Gt:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs > rhs;
            };

        case BinaryOpCode::Ge:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs >= rhs;
            };

        case BinaryOpCode::Lt:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs < rhs;
            };

        case BinaryOpCode::Le:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs <= rhs;
            };

        case BinaryOpCode::Eq:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs == rhs;
            };

        case BinaryOpCode::Neq:
            return [](const Tensor& lhs, const Tensor& rhs) -> Tensor {
                return lhs != rhs;
            };

        default:
            utility::LogError("Unknown operation {}", op);
    }
}

void BinaryEW(benchmark::State& state,
              int size,
              BinaryOpCode op_code,
              const Dtype& dtype,
              const Device& device) {
    Tensor lhs = benchmarks::Rand({1, size}, 1, {1, 127}, dtype, device);
    Tensor rhs = benchmarks::Rand({1, size}, 2, {1, 127}, dtype, device);
    auto op = MakeOperation(op_code);

    Tensor result = op(lhs, rhs);
    benchmark::DoNotOptimize(result);

    for (auto _ : state) {
        Tensor result = op(lhs, rhs);
        benchmark::DoNotOptimize(result);

        cuda::Synchronize(device);
    }
}

#define ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, DTYPE)                   \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100, 100,       \
                      BinaryOpCode::OP, DTYPE, DEVICE)                     \
            ->Unit(benchmark::kMillisecond);                               \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100000, 100000, \
                      BinaryOpCode::OP, DTYPE, DEVICE)                     \
            ->Unit(benchmark::kMillisecond);                               \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100000000,      \
                      100000000, BinaryOpCode::OP, DTYPE, DEVICE)          \
            ->Unit(benchmark::kMillisecond);

#define ENUM_BM_DTYPE(FN, OP, DEVICE, DEVICE_NAME)     \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int8)    \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt8)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int16)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt16)  \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int32)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt32)  \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int64)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt64)  \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float32) \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float64)

#define ENUM_BM_DTYPE_WITH_BOOL(FN, OP, DEVICE, DEVICE_NAME) \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Bool)          \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int8)          \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt8)         \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int16)         \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt16)        \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int32)         \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt32)        \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int64)         \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt64)        \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float32)       \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float64)

#define ENUM_BM_DTYPE_SYCL(FN, OP, DEVICE, DEVICE_NAME) \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int8)     \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt8)    \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int16)    \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt16)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int32)    \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt32)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int64)    \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt64)   \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float32)

#define ENUM_BM_DTYPE_SYCL_WITH_BOOL(FN, OP, DEVICE, DEVICE_NAME) \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Bool)               \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int8)               \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt8)              \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int16)              \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt16)             \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int32)              \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt32)             \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Int64)              \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, UInt64)             \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float32)

ENUM_BM_DTYPE(BinaryEW, Add, Device("CPU:0"), CPU)
ENUM_BM_DTYPE(BinaryEW, Sub, Device("CPU:0"), CPU)
ENUM_BM_DTYPE(BinaryEW, Mul, Device("CPU:0"), CPU)
ENUM_BM_DTYPE(BinaryEW, Div, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalAnd, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalOr, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalXor, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Gt, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Ge, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Lt, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Le, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Eq, Device("CPU:0"), CPU)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Neq, Device("CPU:0"), CPU)

#ifdef BUILD_CUDA_MODULE
ENUM_BM_DTYPE(BinaryEW, Add, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE(BinaryEW, Sub, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE(BinaryEW, Mul, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE(BinaryEW, Div, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalAnd, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalOr, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, LogicalXor, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Gt, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Ge, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Lt, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Le, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Eq, Device("CUDA:0"), CUDA)
ENUM_BM_DTYPE_WITH_BOOL(BinaryEW, Neq, Device("CUDA:0"), CUDA)
#endif

#ifdef BUILD_SYCL_MODULE
ENUM_BM_DTYPE_SYCL(BinaryEW, Add, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL(BinaryEW, Sub, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL(BinaryEW, Mul, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL(BinaryEW, Div, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, LogicalAnd, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, LogicalOr, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, LogicalXor, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Gt, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Ge, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Lt, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Le, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Eq, Device("SYCL:0"), SYCL)
ENUM_BM_DTYPE_SYCL_WITH_BOOL(BinaryEW, Neq, Device("SYCL:0"), SYCL)
#endif

BENCHMARK_MAIN();

}  // namespace core
}  // namespace open3d
