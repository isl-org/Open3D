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

template <typename scalar_t>
OPEN3D_SYCL_EXTERNAL scalar_t CPUSumReductionKernel(scalar_t a, scalar_t b) {
    return a + b;
}

class SYCLReductionEngine {
public:
    SYCLReductionEngine(const SYCLReductionEngine&) = delete;
    SYCLReductionEngine& operator=(const SYCLReductionEngine&) = delete;
    SYCLReductionEngine(const Indexer& indexer) : indexer_(indexer) {}

    template <typename scalar_t>
    void Run(const ReductionOpCode& op_code,
             scalar_t identity,
             const Device& device) {
        sy::queue& queue = sycl::GetDefaultQueue(device);
        const int64_t num_workloads = indexer_.NumWorkloads();

        queue.submit([&](sy::handler& h) {
                 auto reducer =
                         sy::reduction(indexer_.GetOutputPtr<scalar_t>(0),
                                       identity, sy::plus<scalar_t>());
                 h.parallel_for(
                         sy::range<1>(num_workloads), reducer,
                         [num_workloads, this](sy::id<1> i, auto& reducer_arg) {
                             scalar_t* src =
                                     indexer_.GetInputPtr<scalar_t>(-0, i);
                             reducer_arg += *src;
                         });
             }).wait();
    }

private:
    Indexer indexer_;
};

void ReductionSYCL(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code) {
    if (s_regular_reduce_ops.count(op_code)) {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME, dims);
        SYCLReductionEngine re(indexer);

        DISPATCH_DTYPE_TO_TEMPLATE_SYCL(src.GetDtype(), [&]() {
            re.Run<scalar_t>(op_code, 0, src.GetDevice());
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
