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

#include <cuda.h>
#include <cuda_runtime.h>

#include "open3d/core/CUDAUtils.h"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

namespace open3d {
namespace core {
namespace kernel {
namespace cuda_launcher {

static constexpr int64_t default_block_size = 128;
static constexpr int64_t default_thread_size = 4;

/// Calls f(n) with the "grid-stride loops" pattern.
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; i++) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

/// Run a function in parallel with CUDA.
///
/// This is typically used together with cpu_launcher::ParallelFor() to share
/// the same code between CPU and CUDA. For example:
///
/// ```cpp
/// #if defined(__CUDACC__)
///     namespace launcher = core::kernel::cuda_launcher;
/// #else
///     namespace launcher = core::kernel::cpu_launcher;
/// #endif
///
/// launcher::ParallelFor(num_workloads, [=] OPEN3D_DEVICE(int64_t idx) {
///     process_workload(idx);
/// });
/// ```
///
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the kernel
/// to be used on both CPU and CUDA, capture the variables by value.
template <typename func_t>
void ParallelFor(int64_t n, const func_t& func) {
    if (n == 0) {
        return;
    }
    int64_t items_per_block = default_block_size * default_thread_size;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    ElementWiseKernel<default_block_size, default_thread_size>
            <<<grid_size, default_block_size, 0, core::cuda::GetStream()>>>(
                    n, func);
    OPEN3D_GET_LAST_CUDA_ERROR("ParallelFor failed.");
}

}  // namespace cuda_launcher
}  // namespace kernel
}  // namespace core
}  // namespace open3d
