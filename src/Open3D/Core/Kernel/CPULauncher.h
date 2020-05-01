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

#include <cassert>
#include <vector>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/ParallelUtil.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

class CPULauncher {
public:
    template <typename func_t>
    static void LaunchUnaryEWKernel(const Indexer& indexer,
                                    func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        }
    }

    template <typename func_t>
    static void LaunchBinaryEWKernel(const Indexer& indexer,
                                     func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetInputPtr(1, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        }
    }

    template <typename func_t>
    static void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                            func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            element_kernel(indexer.GetInputPtr(workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        }
    }

    template <typename scalar_t, typename func_t>
    static void LaunchReductionKernelSerial(const Indexer& indexer,
                                            func_t element_kernel) {
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            element_kernel(indexer.GetInputPtr(0, workload_idx),
                           indexer.GetOutputPtr(workload_idx));
        }
    }

    /// Create num_threads workers to compute partial reductions and then reduce
    /// to the final results. This only applies to reduction op with one output.
    template <typename scalar_t, typename func_t>
    static void LaunchReductionKernelTwoPass(const Indexer& indexer,
                                             func_t element_kernel,
                                             scalar_t identity) {
        if (indexer.NumOutputElements() > 1) {
            utility::LogError(
                    "Internal error: two-pass reduction only works for "
                    "single-output reduction ops.");
        }
        int64_t num_workloads = indexer.NumWorkloads();
        int64_t num_threads = parallel_util::GetMaxThreads();
        int64_t workload_per_thread =
                (num_workloads + num_threads - 1) / num_threads;
        std::vector<scalar_t> thread_results(num_threads, identity);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            int64_t start = thread_idx * workload_per_thread;
            int64_t end = std::min(start + workload_per_thread, num_workloads);
            for (int64_t workload_idx = start; workload_idx < end;
                 ++workload_idx) {
                element_kernel(indexer.GetInputPtr(0, workload_idx),
                               &thread_results[thread_idx]);
            }
        }
        void* output_ptr = indexer.GetOutputPtr(0);
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            element_kernel(&thread_results[thread_idx], output_ptr);
        }
    }

    template <typename scalar_t, typename func_t>
    static void LaunchReductionParallelDim(const Indexer& indexer,
                                           func_t element_kernel) {
        // Prefers outer dimension >= num_threads.
        const int64_t* indexer_shape = indexer.GetMasterShape();
        const int64_t num_dims = indexer.NumDims();
        int64_t num_threads = parallel_util::GetMaxThreads();

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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < indexer_shape[best_dim]; ++i) {
            Indexer sub_indexer(indexer);
            sub_indexer.ShrinkDim(best_dim, i, 1);
            LaunchReductionKernelSerial<scalar_t>(sub_indexer, element_kernel);
        }
    }
};

}  // namespace kernel
}  // namespace open3d
