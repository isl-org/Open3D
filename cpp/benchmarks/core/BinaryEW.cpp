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

//#ifdef BUILD_CUDA_MODULE
//#define ENUM_BM_TENSOR(FN, OP)
//    ENUM_BM_DTYPE(FN, OP, Device("CPU:0"), CPU)
//    ENUM_BM_DTYPE(FN, OP, Device("CUDA:0"), CUDA)
//#else
#define ENUM_BM_TENSOR(FN, OP) ENUM_BM_DTYPE(FN, OP, Device("CPU:0"), CPU)
//#endif

//#ifdef BUILD_CUDA_MODULE
//#define ENUM_BM_TENSOR_WTIH_BOOL(FN, OP)
//    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CPU:0"), CPU)
//    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CUDA:0"), CUDA)
//#else
#define ENUM_BM_TENSOR_WTIH_BOOL(FN, OP) \
    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CPU:0"), CPU)
//#endif

ENUM_BM_TENSOR(BinaryEW, Add)
ENUM_BM_TENSOR(BinaryEW, Sub)
ENUM_BM_TENSOR(BinaryEW, Mul)
ENUM_BM_TENSOR(BinaryEW, Div)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, LogicalAnd)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, LogicalOr)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, LogicalXor)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Gt)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Ge)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Lt)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Le)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Eq)
ENUM_BM_TENSOR_WTIH_BOOL(BinaryEW, Neq)

}  // namespace core
}  // namespace open3d
