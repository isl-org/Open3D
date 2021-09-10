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

#pragma once

#include <cstdint>
#include <type_traits>

#include "open3d/core/Device.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Overload.h"
#include "open3d/utility/Parallel.h"
#include "open3d/utility/Preprocessor.h"

namespace open3d {
namespace core {

/// Run a vectorized function in parallel on CPU or CUDA.
///
/// \param device The device for the parallel for loop to run on.
/// \param n The number of workloads.
/// \param vec_func The vectorized function to be executed in parallel. The
/// function should be provided using the OPEN3D_VECTORIZED_LAMBDA macro
/// and expects all additional variables to work on, e.g.,
/// `OPEN3D_VECTORIZED_LAMBDA(MyLambdaKernel, some_used_variable)`.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernel to be used on both CPU and CUDA, capture the variables by value.
template <typename vec_func_t, typename func_t>
void ParallelForVectorized(const Device& device,
                           int64_t n,
                           const vec_func_t& vec_func,
                           const func_t& func) {
#ifdef BUILD_ISPC_MODULE

#ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
#else
    int num_threads = utility::EstimateMaxThreads();
    ParallelForCPU_(device, num_threads, [&](int64_t i) {
        int64_t start = n * i / num_threads;
        int64_t end = std::min<int64_t>(n * (i + 1) / num_threads, n);
        vec_func(start, end);
    });
#endif

#else

    ParallelFor(device, n, func);

#endif
}

#ifdef BUILD_ISPC_MODULE

// Internal helper macro.
#define OPEN3D_CALL_ISPC_KERNEL_(LambdaKernel, start, end, ...) \
    using namespace ispc;                                       \
    LambdaKernel(start, end, __VA_ARGS__);

#else

// Internal helper macro.
#define OPEN3D_CALL_ISPC_KERNEL_(LambdaKernel, start, end, ...)          \
    utility::LogError(                                                   \
            "ISPC module disabled. Unable to call vectorized kernel {}", \
            OPEN3D_STRINGIFY(LambdaKernel));

#endif

/// Internal helper macro.
#define OPEN3D_OVERLOADED_LAMBDA_(T, LambdaKernel, ...)                       \
    [&](T, int64_t start, int64_t end) {                                      \
        OPEN3D_CALL_ISPC_KERNEL_(                                             \
                OPEN3D_CONCAT(LambdaKernel, OPEN3D_CONCAT(_, T)), start, end, \
                __VA_ARGS__);                                                 \
    }

/// OPEN3D_VECTORIZED_LAMBDA(LambdaKernel, ...)
///
/// Defines a lambda function to call the provided kernel.
///
/// Use the OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA macro to define the
/// kernel in the ISPC source file.
#define OPEN3D_VECTORIZED_LAMBDA(LambdaKernel, ...)                      \
    [&](int64_t start, int64_t end) {                                    \
        OPEN3D_CALL_ISPC_KERNEL_(LambdaKernel, start, end, __VA_ARGS__); \
    }

/// OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(T, LambdaKernel, ...)
///
/// Defines a lambda function to call the provided template-like kernel.
/// Supported types:
/// - bool
/// - unsigned + signed {8,16,32,64} bit integers,
/// - float, double
///
/// Use the OPEN3D_EXPORT_TEMPLATE_VECTORIZED_LAMBDA macro to define the
/// kernel in the ISPC source file.
#define OPEN3D_TEMPLATE_VECTORIZED_LAMBDA(T, LambdaKernel, ...)                \
    [&](int64_t start, int64_t end) {                                          \
        static_assert(std::is_arithmetic<T>::value,                            \
                      "Data type is not an arithmetic type");                  \
        utility::Overload(                                                     \
                OPEN3D_OVERLOADED_LAMBDA_(bool, LambdaKernel, __VA_ARGS__),    \
                OPEN3D_OVERLOADED_LAMBDA_(uint8_t, LambdaKernel, __VA_ARGS__), \
                OPEN3D_OVERLOADED_LAMBDA_(int8_t, LambdaKernel, __VA_ARGS__),  \
                OPEN3D_OVERLOADED_LAMBDA_(uint16_t, LambdaKernel,              \
                                          __VA_ARGS__),                        \
                OPEN3D_OVERLOADED_LAMBDA_(int16_t, LambdaKernel, __VA_ARGS__), \
                OPEN3D_OVERLOADED_LAMBDA_(uint32_t, LambdaKernel,              \
                                          __VA_ARGS__),                        \
                OPEN3D_OVERLOADED_LAMBDA_(int32_t, LambdaKernel, __VA_ARGS__), \
                OPEN3D_OVERLOADED_LAMBDA_(uint64_t, LambdaKernel,              \
                                          __VA_ARGS__),                        \
                OPEN3D_OVERLOADED_LAMBDA_(int64_t, LambdaKernel, __VA_ARGS__), \
                OPEN3D_OVERLOADED_LAMBDA_(float, LambdaKernel, __VA_ARGS__),   \
                OPEN3D_OVERLOADED_LAMBDA_(double, LambdaKernel, __VA_ARGS__),  \
                [&](auto&& generic, int64_t start, int64_t end) {              \
                    utility::LogError(                                         \
                            "Unsupported data type {} for calling "            \
                            "vectorized kernel {}",                            \
                            typeid(generic).name(),                            \
                            OPEN3D_STRINGIFY(LambdaKernel));                   \
                })(T{}, start, end);                                           \
    }

}  // namespace core
}  // namespace open3d
