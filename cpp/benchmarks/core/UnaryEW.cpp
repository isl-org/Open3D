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

#include <random>
#include <type_traits>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

enum class UnaryOpCode {
    Sqrt,
    Sin,
    Cos,
    Neg,
    Exp,
    Abs,
    IsNan,
    IsInf,
    IsFinite,
    Floor,
    Ceil,
    Round,
    Trunc,
    LogicalNot,
};

std::function<Tensor(const Tensor&)> MakeOperation(UnaryOpCode op) {
    switch (op) {
        case UnaryOpCode::Sqrt:
            return [](const Tensor& arg) -> Tensor { return arg.Sqrt(); };

        case UnaryOpCode::Sin:
            return [](const Tensor& arg) -> Tensor { return arg.Sin(); };

        case UnaryOpCode::Cos:
            return [](const Tensor& arg) -> Tensor { return arg.Cos(); };

        case UnaryOpCode::Neg:
            return [](const Tensor& arg) -> Tensor { return arg.Neg(); };

        case UnaryOpCode::Exp:
            return [](const Tensor& arg) -> Tensor { return arg.Exp(); };

        case UnaryOpCode::Abs:
            return [](const Tensor& arg) -> Tensor { return arg.Abs(); };

        case UnaryOpCode::IsNan:
            return [](const Tensor& arg) -> Tensor { return arg.IsNan(); };

        case UnaryOpCode::IsInf:
            return [](const Tensor& arg) -> Tensor { return arg.IsInf(); };

        case UnaryOpCode::IsFinite:
            return [](const Tensor& arg) -> Tensor { return arg.IsFinite(); };

        case UnaryOpCode::Floor:
            return [](const Tensor& arg) -> Tensor { return arg.Floor(); };

        case UnaryOpCode::Ceil:
            return [](const Tensor& arg) -> Tensor { return arg.Ceil(); };

        case UnaryOpCode::Round:
            return [](const Tensor& arg) -> Tensor { return arg.Round(); };

        case UnaryOpCode::Trunc:
            return [](const Tensor& arg) -> Tensor { return arg.Trunc(); };

        case UnaryOpCode::LogicalNot:
            return [](const Tensor& arg) -> Tensor { return arg.LogicalNot(); };

        default:
            utility::LogError("Unknown operation {}", op);
    }
}

static Tensor RandomUniform(const SizeVector& shape,
                            size_t seed,
                            const std::pair<Scalar, Scalar>& range,
                            Dtype dtype,
                            const Device& device = Device("CPU:0")) {
    // Initialize on CPU, then copy to device
    Tensor random = Tensor::Empty(shape, dtype, Device("CPU:0"));

    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
        scalar_t low = range.first.To<scalar_t>();
        scalar_t high = range.second.To<scalar_t>();

        TensorIterator random_it(random);
        using uniform_distribution = std::conditional_t<
                std::is_same<scalar_t, bool>::value,
                std::uniform_int_distribution<uint16_t>,
                std::conditional_t<
                        std::is_same<scalar_t, uint8_t>::value,
                        std::uniform_int_distribution<uint16_t>,
                        std::conditional_t<
                                std::is_same<scalar_t, int8_t>::value,
                                std::uniform_int_distribution<int16_t>,
                                std::conditional_t<
                                        std::is_integral<scalar_t>::value,
                                        std::uniform_int_distribution<scalar_t>,
                                        std::conditional_t<
                                                std::is_floating_point<
                                                        scalar_t>::value,
                                                std::uniform_real_distribution<
                                                        scalar_t>,
                                                void>>>>>;

        int num_threads = utility::EstimateMaxThreads();
        std::vector<std::default_random_engine> rng;
        for (int64_t i = 0; i < num_threads; ++i) {
            rng.emplace_back(seed + i);
        }
        uniform_distribution dist(low, high);

        ParallelFor(Device("CPU:0"), num_threads, [&](int64_t i) {
            int64_t start = random.NumElements() * i / num_threads;
            int64_t end = std::min<int64_t>(
                    random.NumElements() * (i + 1) / num_threads,
                    random.NumElements());
            for (int64_t idx = start; idx < end; ++idx) {
                *static_cast<scalar_t*>(random_it.GetPtr(idx)) = dist(rng[i]);
            }
        });
    });

    return random.To(device);
}

void UnaryEW(benchmark::State& state,
             int size,
             UnaryOpCode op_code,
             const Dtype& dtype,
             const Device& device) {
    Tensor arg = RandomUniform({1, size}, 1, {1, 127}, dtype, device);
    auto op = MakeOperation(op_code);

    Tensor result = op(arg);
    benchmark::DoNotOptimize(result);

    for (auto _ : state) {
        Tensor result = op(arg);
        benchmark::DoNotOptimize(result);

        cuda::Synchronize(device);
    }
}

#define ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, DTYPE)                   \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100, 100,       \
                      UnaryOpCode::OP, DTYPE, DEVICE)                      \
            ->Unit(benchmark::kMillisecond);                               \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100000, 100000, \
                      UnaryOpCode::OP, DTYPE, DEVICE)                      \
            ->Unit(benchmark::kMillisecond);                               \
    BENCHMARK_CAPTURE(FN, OP##__##DEVICE_NAME##_##DTYPE##__100000000,      \
                      100000000, UnaryOpCode::OP, DTYPE, DEVICE)           \
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

#define ENUM_BM_DTYPE_FLOAT(FN, OP, DEVICE, DEVICE_NAME) \
    ENUM_BM_SIZE(FN, OP, DEVICE, DEVICE_NAME, Float32)   \
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
//#define ENUM_BM_TENSOR_FLOAT(FN, OP)
//    ENUM_BM_DTYPE_FLOAT(FN, OP, Device("CPU:0"), CPU)
//    ENUM_BM_DTYPE_FLOAT(FN, OP, Device("CUDA:0"), CUDA)
//#else
#define ENUM_BM_TENSOR_FLOAT(FN, OP) \
    ENUM_BM_DTYPE_FLOAT(FN, OP, Device("CPU:0"), CPU)
//#endif

//#ifdef BUILD_CUDA_MODULE
//#define ENUM_BM_TENSOR_WTIH_BOOL(FN, OP)
//    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CPU:0"), CPU)
//    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CUDA:0"), CUDA)
//#else
#define ENUM_BM_TENSOR_WTIH_BOOL(FN, OP) \
    ENUM_BM_DTYPE_WITH_BOOL(FN, OP, Device("CPU:0"), CPU)
//#endif

ENUM_BM_TENSOR_FLOAT(UnaryEW, Sqrt)
ENUM_BM_TENSOR_FLOAT(UnaryEW, Sin)
ENUM_BM_TENSOR_FLOAT(UnaryEW, Cos)
ENUM_BM_TENSOR_FLOAT(UnaryEW, Exp)
ENUM_BM_TENSOR_FLOAT(UnaryEW, IsNan)
ENUM_BM_TENSOR_FLOAT(UnaryEW, IsInf)
ENUM_BM_TENSOR_FLOAT(UnaryEW, IsFinite)
ENUM_BM_TENSOR(UnaryEW, Abs)
ENUM_BM_TENSOR(UnaryEW, Neg)
ENUM_BM_TENSOR(UnaryEW, Floor)
ENUM_BM_TENSOR(UnaryEW, Ceil)
ENUM_BM_TENSOR(UnaryEW, Round)
ENUM_BM_TENSOR(UnaryEW, Trunc)
ENUM_BM_TENSOR_WTIH_BOOL(UnaryEW, LogicalNot)

}  // namespace core
}  // namespace open3d
