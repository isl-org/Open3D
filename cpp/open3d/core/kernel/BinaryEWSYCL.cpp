
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

#include <type_traits>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/BinaryEW.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename src_t, typename dst_t, typename BinaryEWOp>
static void LaunchBinaryEWKernel(sy::queue& queue, const Indexer& indexer) {
    const int64_t num_workloads = indexer.NumWorkloads();
    queue.submit([&](sy::handler& h) {
             h.parallel_for(num_workloads, [indexer](int64_t i) {
                 *indexer.GetOutputPtr<dst_t>(i) =
                         BinaryEWOp()(*indexer.GetInputPtr<src_t>(0, i),
                                      *indexer.GetInputPtr<src_t>(1, i));
             });
         }).wait();
}

template <typename T>
struct MaxFunc {
    T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <typename T>
struct MinFunc {
    T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

void BinaryEWSYCL(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  const BinaryEWOpCode& op_code) {
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    sy::queue& queue = sycl::GetDefaultQueue(lhs.GetDevice());

    if (IsBinaryEWBoolean(op_code)) {
        Indexer indexer;
        if (dst_dtype == src_dtype) {
            // Inplace boolean op's output type is the same as the input.
            // e.g. np.logical_and(a, b, out=a), where a, b are floats.
            indexer = Indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        } else if (dst_dtype == core::Bool) {
            // dst_dtype == core::Bool: lhs, rhs, dst are all boolean. This
            // is the most common case.
            indexer = Indexer({lhs, rhs}, dst,
                              DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
        const int64_t num_workloads = indexer.NumWorkloads();

        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
            using src_t = scalar_t;
            DISPATCH_BOOL_OR_TYPE(dst_dtype, src_t, [&]() {
                using dst_t = scalar_t;
                if (op_code == BinaryEWOpCode::LogicalAnd) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::logical_and<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::LogicalOr) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::logical_or<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::LogicalXor) {
                    // For bool values, != is the same as for xor.
                    // For non-bool values, we still use != to compute xor.
                    LaunchBinaryEWKernel<src_t, dst_t,
                                         std::not_equal_to<src_t>>(queue,
                                                                   indexer);
                } else if (op_code == BinaryEWOpCode::Gt) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::greater<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::Lt) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::less<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::Ge) {
                    LaunchBinaryEWKernel<src_t, dst_t,
                                         std::greater_equal<src_t>>(queue,
                                                                    indexer);
                } else if (op_code == BinaryEWOpCode::Le) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::less_equal<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::Eq) {
                    LaunchBinaryEWKernel<src_t, dst_t, std::equal_to<src_t>>(
                            queue, indexer);
                } else if (op_code == BinaryEWOpCode::Ne) {
                    LaunchBinaryEWKernel<src_t, dst_t,
                                         std::not_equal_to<src_t>>(queue,
                                                                   indexer);
                } else {
                    utility::LogError("Unsupported BinaryEWOpCode {}.",
                                      op_code);
                }
            });
        });
    } else if (op_code == BinaryEWOpCode::Maximum ||
               op_code == BinaryEWOpCode::Minimum) {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        const int64_t num_workloads = indexer.NumWorkloads();
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL_SYCL(src_dtype, [&]() {
            if (op_code == BinaryEWOpCode::Maximum) {
                LaunchBinaryEWKernel<scalar_t, scalar_t, MaxFunc<scalar_t>>(
                        queue, indexer);
            } else if (op_code == BinaryEWOpCode::Minimum) {
                LaunchBinaryEWKernel<scalar_t, scalar_t, MinFunc<scalar_t>>(
                        queue, indexer);
            } else {
                utility::LogError("Unsupported BinaryEWOpCode {}.", op_code);
            }
        });
    } else {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        const int64_t num_workloads = indexer.NumWorkloads();
        DISPATCH_DTYPE_TO_TEMPLATE_SYCL(src_dtype, [&]() {
            if (op_code == BinaryEWOpCode::Add) {
                LaunchBinaryEWKernel<scalar_t, scalar_t, std::plus<scalar_t>>(
                        queue, indexer);
            } else if (op_code == BinaryEWOpCode::Sub) {
                LaunchBinaryEWKernel<scalar_t, scalar_t, std::minus<scalar_t>>(
                        queue, indexer);
            } else if (op_code == BinaryEWOpCode::Mul) {
                LaunchBinaryEWKernel<scalar_t, scalar_t,
                                     std::multiplies<scalar_t>>(queue, indexer);
            } else if (op_code == BinaryEWOpCode::Div) {
                LaunchBinaryEWKernel<scalar_t, scalar_t,
                                     std::divides<scalar_t>>(queue, indexer);
            } else {
                utility::LogError("Unsupported BinaryEWOpCode {}.", op_code);
            }
        });
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
