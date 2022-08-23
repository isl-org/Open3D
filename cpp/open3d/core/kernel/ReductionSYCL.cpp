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
#include "open3d/core/SYCLUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/Reduction.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

namespace open3d {
namespace core {
namespace kernel {

void ReductionSYCL(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code) {
    sy::queue& queue = sycl::GetDefaultQueue(src.GetDevice());

    if (s_regular_reduce_ops.count(op_code)) {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        const int64_t num_workloads = indexer.NumWorkloads();
        DISPATCH_DTYPE_TO_TEMPLATE_SYCL(src.GetDtype(), [&]() {
            scalar_t identity;
            switch (op_code) {
                case ReductionOpCode::Sum:
                    identity = 0;
                    dst.Fill(identity);
                    queue.submit([&](sy::handler& h) {
                             h.parallel_for(1, [indexer,
                                                num_workloads](int64_t i) {
                                 for (int64_t workload_idx = 0;
                                      workload_idx < num_workloads;
                                      workload_idx++) {
                                     scalar_t* src =
                                             reinterpret_cast<scalar_t*>(
                                                     indexer.GetInputPtr(
                                                             0, workload_idx));
                                     scalar_t* dst =
                                             reinterpret_cast<scalar_t*>(
                                                     indexer.GetOutputPtr(
                                                             workload_idx));
                                     *dst = (*src) + (*dst);
                                 }
                             });
                         }).wait();
                    break;
                default:
                    utility::LogError("Unsupported op code.");
                    break;
            }
        });
    } else if (s_arg_reduce_ops.count(op_code)) {
        utility::LogError("To be implemented.");

    } else if (s_boolean_reduce_ops.count(op_code)) {
        utility::LogError("To be implemented.");
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
