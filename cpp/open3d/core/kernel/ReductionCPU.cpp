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

#include <limits>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Reduction.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename scalar_t>
static inline scalar_t CPUSumReductionKernel(scalar_t a, scalar_t b) {
    return a + b;
}

template <typename scalar_t>
static inline scalar_t CPUProdReductionKernel(scalar_t a, scalar_t b) {
    return a * b;
}

template <typename scalar_t>
static inline scalar_t CPUMinReductionKernel(scalar_t a, scalar_t b) {
    return std::min(a, b);
}

template <typename scalar_t>
static inline scalar_t CPUMaxReductionKernel(scalar_t a, scalar_t b) {
    return std::max(a, b);
}

static inline uint8_t CPUAllReductionKernel(uint8_t a, uint8_t b) {
    return a && b;
}

static inline uint8_t CPUAnyReductionKernel(uint8_t a, uint8_t b) {
    return a || b;
}

template <typename scalar_t>
static inline std::pair<int64_t, scalar_t> CPUArgMinReductionKernel(
        int64_t a_idx, scalar_t a, int64_t b_idx, scalar_t b) {
    if (a < b) {
        return {a_idx, a};
    } else {
        return {b_idx, b};
    }
}

template <typename scalar_t>
static inline std::pair<int64_t, scalar_t> CPUArgMaxReductionKernel(
        int64_t a_idx, scalar_t a, int64_t b_idx, scalar_t b) {
    if (a > b) {
        return {a_idx, a};
    } else {
        return {b_idx, b};
    }
}

class CPUReductionEngine {
public:
    CPUReductionEngine(const CPUReductionEngine&) = delete;
    CPUReductionEngine& operator=(const CPUReductionEngine&) = delete;
    CPUReductionEngine(const Indexer& indexer) : indexer_(indexer) {}

    template <typename func_t, typename scalar_t>
    void Run(const func_t& reduce_func, scalar_t identity) {
        // See: PyTorch's TensorIterator::parallel_reduce for the reference
        // design of reduction strategy.
        if (utility::EstimateMaxThreads() == 1 || utility::InParallel()) {
            LaunchReductionKernelSerial<scalar_t>(indexer_, reduce_func);
        } else if (indexer_.NumOutputElements() <= 1) {
            LaunchReductionKernelTwoPass<scalar_t>(indexer_, reduce_func,
                                                   identity);
        } else {
            LaunchReductionParallelDim<scalar_t>(indexer_, reduce_func);
        }
    }

private:
    template <typename scalar_t, typename func_t>
    static void LaunchReductionKernelSerial(const Indexer& indexer,
                                            func_t element_kernel) {
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            scalar_t* src = reinterpret_cast<scalar_t*>(
                    indexer.GetInputPtr(0, workload_idx));
            scalar_t* dst = reinterpret_cast<scalar_t*>(
                    indexer.GetOutputPtr(workload_idx));
            *dst = element_kernel(*src, *dst);
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
        int64_t num_threads = utility::EstimateMaxThreads();
        int64_t workload_per_thread =
                (num_workloads + num_threads - 1) / num_threads;
        std::vector<scalar_t> thread_results(num_threads, identity);

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            int64_t start = thread_idx * workload_per_thread;
            int64_t end = std::min(start + workload_per_thread, num_workloads);
            for (int64_t workload_idx = start; workload_idx < end;
                 ++workload_idx) {
                scalar_t* src = reinterpret_cast<scalar_t*>(
                        indexer.GetInputPtr(0, workload_idx));
                thread_results[thread_idx] =
                        element_kernel(*src, thread_results[thread_idx]);
            }
        }
        scalar_t* dst = reinterpret_cast<scalar_t*>(indexer.GetOutputPtr(0));
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            *dst = element_kernel(thread_results[thread_idx], *dst);
        }
    }

    template <typename scalar_t, typename func_t>
    static void LaunchReductionParallelDim(const Indexer& indexer,
                                           func_t element_kernel) {
        // Prefers outer dimension >= num_threads.
        const int64_t* indexer_shape = indexer.GetMasterShape();
        const int64_t num_dims = indexer.NumDims();
        int64_t num_threads = utility::EstimateMaxThreads();

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

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int64_t i = 0; i < indexer_shape[best_dim]; ++i) {
            Indexer sub_indexer(indexer);
            sub_indexer.ShrinkDim(best_dim, i, 1);
            LaunchReductionKernelSerial<scalar_t>(sub_indexer, element_kernel);
        }
    }

private:
    Indexer indexer_;
};

class CPUArgReductionEngine {
public:
    CPUArgReductionEngine(const CPUArgReductionEngine&) = delete;
    CPUArgReductionEngine& operator=(const CPUArgReductionEngine&) = delete;
    CPUArgReductionEngine(const Indexer& indexer) : indexer_(indexer) {}

    template <typename func_t, typename scalar_t>
    void Run(const func_t& reduce_func, scalar_t identity) {
        // Arg-reduction needs to iterate each output element separately in
        // sub-iterations. Each output elemnent corresponds to multiple input
        // elements. We need to keep track of the indices within each
        // sub-iteration.
        int64_t num_output_elements = indexer_.NumOutputElements();

#pragma omp parallel for schedule(static) \
        num_threads(utility::EstimateMaxThreads())
        for (int64_t output_idx = 0; output_idx < num_output_elements;
             output_idx++) {
            // sub_indexer.NumWorkloads() == ipo.
            // sub_indexer's workload_idx is indexer_'s ipo_idx.
            Indexer sub_indexer = indexer_.GetPerOutputIndexer(output_idx);
            scalar_t dst_val = identity;
            for (int64_t workload_idx = 0;
                 workload_idx < sub_indexer.NumWorkloads(); workload_idx++) {
                int64_t src_idx = workload_idx;
                scalar_t* src_val = reinterpret_cast<scalar_t*>(
                        sub_indexer.GetInputPtr(0, workload_idx));
                int64_t* dst_idx = reinterpret_cast<int64_t*>(
                        sub_indexer.GetOutputPtr(0, workload_idx));
                std::tie(*dst_idx, dst_val) =
                        reduce_func(src_idx, *src_val, *dst_idx, dst_val);
            }
        }
    }

private:
    Indexer indexer_;
};

void ReductionCPU(const Tensor& src,
                  Tensor& dst,
                  const SizeVector& dims,
                  bool keepdim,
                  ReductionOpCode op_code) {
    if (s_regular_reduce_ops.find(op_code) != s_regular_reduce_ops.end()) {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        CPUReductionEngine re(indexer);
        DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
            scalar_t identity;
            switch (op_code) {
                case ReductionOpCode::Sum:
                    identity = 0;
                    dst.Fill(identity);
                    re.Run(CPUSumReductionKernel<scalar_t>, identity);
                    break;
                case ReductionOpCode::Prod:
                    identity = 1;
                    dst.Fill(identity);
                    re.Run(CPUProdReductionKernel<scalar_t>, identity);
                    break;
                case ReductionOpCode::Min:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Min.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::max();
                        dst.Fill(identity);
                        re.Run(CPUMinReductionKernel<scalar_t>, identity);
                    }
                    break;
                case ReductionOpCode::Max:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support Max.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::lowest();
                        dst.Fill(identity);
                        re.Run(CPUMaxReductionKernel<scalar_t>, identity);
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_arg_reduce_ops.find(op_code) != s_arg_reduce_ops.end()) {
        if (dst.GetDtype() != core::Int64) {
            utility::LogError("Arg-reduction must have int64 output dtype.");
        }
        // Accumulation buffer to store temporary min/max values.
        Tensor dst_acc(dst.GetShape(), src.GetDtype(), src.GetDevice());

        Indexer indexer({src}, {dst, dst_acc}, DtypePolicy::INPUT_SAME, dims);
        CPUArgReductionEngine re(indexer);
        DISPATCH_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
            scalar_t identity;
            switch (op_code) {
                case ReductionOpCode::ArgMin:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support ArgMin.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::max();
                        dst_acc.Fill(identity);
                        re.Run(CPUArgMinReductionKernel<scalar_t>, identity);
                    }
                    break;
                case ReductionOpCode::ArgMax:
                    if (indexer.NumWorkloads() == 0) {
                        utility::LogError(
                                "Zero-size Tensor does not support ArgMax.");
                    } else {
                        identity = std::numeric_limits<scalar_t>::lowest();
                        dst_acc.Fill(identity);
                        re.Run(CPUArgMaxReductionKernel<scalar_t>, identity);
                    }
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_boolean_reduce_ops.find(op_code) !=
               s_boolean_reduce_ops.end()) {
        if (src.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean input tensor.");
        }
        if (dst.GetDtype() != core::Bool) {
            utility::LogError(
                    "Boolean reduction only supports boolean output tensor.");
        }
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        CPUReductionEngine re(indexer);
        switch (op_code) {
            case ReductionOpCode::All:
                // Identity == true. 0-sized tensor, returns true.
                dst.Fill(true);
                re.Run(CPUAllReductionKernel, static_cast<uint8_t>(true));
                break;
            case ReductionOpCode::Any:
                // Identity == false. 0-sized tensor, returns false.
                dst.Fill(false);
                re.Run(CPUAnyReductionKernel, static_cast<uint8_t>(false));
                break;
            default:
                utility::LogError("Unsupported op code.");
                break;
        }
    } else {
        utility::LogError("Unsupported op code.");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
