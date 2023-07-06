// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Dispatch.h"
#include "open3d/core/Indexer.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

template <typename func_t>
void LaunchIndexReductionKernel(int64_t dim,
                                const Device& device,
                                const Tensor& index,
                                const Tensor& src,
                                Tensor& dst,
                                const func_t& element_kernel) {
    // index: [N,], src: [N, D], dst: [M, D]
    // In Indexer, output shape defines the actual master strides.
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
    auto element_func = [=](int64_t workload_idx) {
        int reduction_idx = workload_idx / broadcasting_elems;
        int broadcasting_idx = workload_idx % broadcasting_elems;

        const int64_t idx = index_ptr[reduction_idx];
        int64_t dst_idx = idx * broadcasting_elems + broadcasting_idx;

        void* src_ptr = indexer.GetOutputPtr(0, workload_idx);
        void* dst_ptr = indexer.GetInputPtr(0, dst_idx);
        // Note input and output is switched here to adapt to the indexer
        element_kernel(src_ptr, dst_ptr);
    };

    // TODO: check in detail
    // No OpenMP could be faster, otherwise there would be thousands of atomics.
    for (int64_t d = 0; d < indexer.NumWorkloads(); ++d) {
        element_func(d);
    }
}

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CPUSumKernel(const void* src, void* dst) {
    scalar_t* dst_s_ptr = static_cast<scalar_t*>(dst);
    const scalar_t* src_s_ptr = static_cast<const scalar_t*>(src);
    *dst_s_ptr += *src_s_ptr;
}

void IndexAddCPU_(int64_t dim,
                  const Tensor& index,
                  const Tensor& src,
                  Tensor& dst) {
    DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(src.GetDtype(), [&]() {
        LaunchIndexReductionKernel(dim, src.GetDevice(), index, src, dst,
                                   [](const void* src, void* dst) {
                                       CPUSumKernel<scalar_t>(src, dst);
                                   });
    });
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
