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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"

// CUDA kernel launcher's goal is to separate scheduling (looping through each
// valid element) and computation (operations performed on each element).
//
// The kernel launch mechanism is inspired by PyTorch's launch Loops.cuh.
// See: https://tinyurl.com/y4lak257

static constexpr int64_t threads_per_block = 128;
static constexpr int64_t items_per_thread = 4;

namespace open3d {
namespace kernel {

// Applies f for each element
// Works for unary / binary elementwise operations
template <int64_t threads_per_block, int64_t items_per_thread, typename func_t>
__global__ void ElementWiseKernel(int64_t num_elems, func_t f) {
    int64_t items_per_block = threads_per_block * items_per_thread;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < items_per_thread; i++) {
        if (idx < num_elems) {
            f(idx);
            idx += threads_per_block;
        }
    }
}

class CUDALauncher {
public:
    template <typename func_t>
    static void LaunchUnaryEWKernel(const Indexer& indexer,
                                    func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t num_elems = indexer.NumWorkloads();
        int64_t items_per_block = threads_per_block * items_per_thread;
        int64_t grid_size = (num_elems + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<threads_per_block, items_per_thread>
                <<<grid_size, threads_per_block, 0>>>(num_elems, f);
    }

    template <typename func_t>
    static void LaunchBinaryEWKernel(const Indexer& indexer,
                                     func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t num_elems = indexer.NumWorkloads();
        int64_t items_per_block = threads_per_block * items_per_thread;
        int64_t grid_size = (num_elems + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetInputPtr(1, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<threads_per_block, items_per_thread>
                <<<grid_size, threads_per_block, 0>>>(num_elems, f);
    }

    template <typename func_t>
    static void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                            func_t element_kernel) {
        OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(func_t);

        int64_t num_elems = indexer.NumWorkloads();
        int64_t items_per_block = threads_per_block * items_per_thread;
        int64_t grid_size = (num_elems + items_per_block - 1) / items_per_block;

        auto f = [=] OPEN3D_HOST_DEVICE(int64_t workload_idx) {
            element_kernel(indexer.GetInputPtr(workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        };

        ElementWiseKernel<threads_per_block, items_per_thread>
                <<<grid_size, threads_per_block, 0>>>(num_elems, f);
    }
};

}  // namespace kernel
}  // namespace open3d
