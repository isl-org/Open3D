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

#include <vector>

#include "open3d/core/AdvancedIndexing.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/ParallelUtil.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {
namespace cpu_launcher {

/// Run a function in parallel on CPU.
///
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// This is typically used together with cuda_launcher::LaunchParallel() to
/// share the same code between CPU and CUDA. For example:
///
/// ```cpp
/// #if defined(__CUDACC__)
///     namespace launcher = core::kernel::cuda_launcher;
/// #else
///     namespace launcher = core::kernel::cpu_launcher;
/// #endif
///
/// launcher::LaunchParallel(num_workloads, [=] OPEN3D_DEVICE(int64_t i) {
///     process_workload(i);
/// });
/// ```
template <typename func_t>
void LaunchParallel(int64_t n, const func_t& func) {
#pragma omp parallel for schedule(static) if (GetMaxThreads() != 1 && \
                                              !InParallel())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}

/// Run a function in parallel on CPU when the number of workloads is larger
/// than a threshold.
///
/// \param n The number of workloads.
/// \param min_parallel_size If \p n <= \p min_parallel_size, the jobs will
/// be executed in serial.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
template <typename func_t>
void LaunchParallel(int64_t n, int64_t min_parallel_size, const func_t& func) {
#pragma omp parallel for schedule(static) if (n > min_parallel_size && \
                                              GetMaxThreads() != 1 &&  \
                                              !InParallel())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}

/// Fills tensor[:][i] with func(i).
///
/// \param indexer The input tensor and output tensor to the indexer are the
/// same (as a hack), since the tensor are filled in-place.
/// \param func A function that takes pointer location and
/// workload index i, computes the value to fill, and fills the value at the
/// pointer location.
template <typename func_t>
void LaunchIndexFillKernel(const Indexer& indexer, const func_t& func) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        func(indexer.GetInputPtr(0, i), i);
    }
}

template <typename func_t>
void LaunchUnaryEWKernel(const Indexer& indexer, const func_t& func) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        func(indexer.GetInputPtr(0, i), indexer.GetOutputPtr(i));
    }
}

template <typename func_t>
void LaunchBinaryEWKernel(const Indexer& indexer, const func_t& func) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        func(indexer.GetInputPtr(0, i), indexer.GetInputPtr(1, i),
             indexer.GetOutputPtr(i));
    }
}

template <typename func_t>
void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                 const func_t& func) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        func(indexer.GetInputPtr(i), indexer.GetOutputPtr(i));
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionKernelSerial(const Indexer& indexer, const func_t& func) {
    for (int64_t i = 0; i < indexer.NumWorkloads(); ++i) {
        func(indexer.GetInputPtr(0, i), indexer.GetOutputPtr(i));
    }
}

/// Create num_threads workers to compute partial reductions and then reduce
/// to the final results. This only applies to reduction op with one output.
template <typename scalar_t, typename func_t>
void LaunchReductionKernelTwoPass(const Indexer& indexer,
                                  const func_t& func,
                                  scalar_t identity) {
    if (indexer.NumOutputElements() > 1) {
        utility::LogError(
                "Internal error: two-pass reduction only works for "
                "single-output reduction ops.");
    }
    int64_t num_workloads = indexer.NumWorkloads();
    int64_t num_threads = GetMaxThreads();
    int64_t workload_per_thread =
            (num_workloads + num_threads - 1) / num_threads;
    std::vector<scalar_t> thread_results(num_threads, identity);

#pragma omp parallel for schedule(static)
    for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        int64_t start = thread_idx * workload_per_thread;
        int64_t end = std::min(start + workload_per_thread, num_workloads);
        for (int64_t i = start; i < end; ++i) {
            func(indexer.GetInputPtr(0, i), &thread_results[thread_idx]);
        }
    }
    void* output_ptr = indexer.GetOutputPtr(0);
    for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        func(&thread_results[thread_idx], output_ptr);
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionParallelDim(const Indexer& indexer, const func_t& func) {
    // Prefers outer dimension >= num_threads.
    const int64_t* indexer_shape = indexer.GetMasterShape();
    const int64_t num_dims = indexer.NumDims();
    int64_t num_threads = GetMaxThreads();

    // Init best_dim as the outer-most non-reduction dim.
    int64_t best_dim = num_dims - 1;
    while (best_dim >= 0 && indexer.IsReductionDim(best_dim)) {
        best_dim--;
    }
    for (int64_t dim = best_dim; dim >= 0 && !indexer.IsReductionDim(dim);
         --dim) {
        if (indexer_shape[dim] >= num_threads) {
            best_dim = dim;
            break;
        } else if (indexer_shape[dim] > indexer_shape[best_dim]) {
            best_dim = dim;
        }
    }
    if (best_dim == -1) {
        utility::LogError(
                "Internal error: all dims are reduction dims, use "
                "LaunchReductionKernelTwoPass instead.");
    }

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < indexer_shape[best_dim]; ++i) {
        Indexer sub_indexer(indexer);
        sub_indexer.ShrinkDim(best_dim, i, 1);
        LaunchReductionKernelSerial<scalar_t>(sub_indexer, func);
    }
}

}  // namespace cpu_launcher
}  // namespace kernel
}  // namespace core
}  // namespace open3d
