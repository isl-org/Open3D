// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

void IndexAddSYCL_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst) {
    // index: [N,], src: [N, D], dst: [M, D]
    // In Indexer, output shape defines the actual primary strides.
    // However, in IndexAdd_, input dominates the iterations.
    // So put dst (output) at indexer's input, and src (input) at output.
    Indexer indexer({dst}, src, DtypePolicy::NONE);

    // Index is simply a 1D contiguous tensor, with a different stride
    // behavior to src. So use raw pointer for simplicity.
    auto index_ptr = index.GetDataPtr<int64_t>();

    int64_t broadcasting_elems = 1;
    for (int64_t d = 1; d < src.NumDims(); ++d) {
        broadcasting_elems *= src.GetShape(d);
    }
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(src.GetDevice());

    // TODO: Replace with SYCL reduction API
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        queue.parallel_for(index.GetLength(), [=](int64_t workload_idx) {
                 int64_t reduction_idx = workload_idx / broadcasting_elems;
                 int64_t broadcasting_idx = workload_idx % broadcasting_elems;

                 const int64_t idx = index_ptr[reduction_idx];
                 int64_t dst_idx = idx * broadcasting_elems + broadcasting_idx;

                 // Note input and output is switched here to adapt to the
                 // indexer
                 scalar_t* src_ptr = indexer.GetOutputPtr<scalar_t>(0, idx);
                 scalar_t* dst_ptr = indexer.GetInputPtr<scalar_t>(0, dst_idx);
                 sycl::atomic_ref<scalar_t, sycl::memory_order::acq_rel,
                                  sycl::memory_scope::device>(*dst_ptr) +=
                         *src_ptr;
             }).wait_and_throw();
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
