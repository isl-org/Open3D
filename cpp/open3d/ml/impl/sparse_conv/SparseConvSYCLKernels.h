// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL port of SparseConvCUDAKernels.{h,cu}. Declares the im2col-style
// "FillColumn" kernels shared by the SparseConv forward/backprop/transpose
// ops. See SparseConvCUDAKernels.h for the detailed per-parameter docs; the
// semantics here are identical, only the device queue type changed
// (cudaStream_t -> sycl::queue&).
#pragma once

#include <sycl/sycl.hpp>
#include <vector>

namespace open3d {
namespace ml {
namespace impl {

/// SYCL port of FillColumn: copies/gathers neighbor features (optionally
/// normalized) into the im2col-style patch matrix used as the GEMM's B
/// operand. One work-group per output point; work-items in the group stride
/// over input channels (and, for the normalizer sum, over neighbors).
/// Depends on \p deps and returns the completion event of its last kernel
/// (the zero-init fill and the fill-column kernel are chained internally);
/// callers thread this event into e.g. the following GEMM's dependencies
/// instead of blocking-waiting on it.
template <class TReal, class TIndex, class TKernelIndex>
sycl::event FillColumnSYCL(sycl::queue& queue,
                           TReal* columns,
                           int in_channels,
                           TIndex begin_idx,
                           TIndex end_idx,
                           TIndex num_out,
                           TIndex num_inp,
                           const TReal* const inp_features,
                           const TReal* const inp_importance,
                           size_t neighbors_index_size,
                           const TIndex* const neighbors_index,
                           const TKernelIndex* const neighbors_kernel_index,
                           const TReal* const neighbors_importance,
                           const int64_t* const neighbors_row_splits,
                           const int num_kernel_elements,
                           bool normalize,
                           const std::vector<sycl::event>& deps = {});

/// SYCL port of FillColumnTranspose: same as FillColumnSYCL but accumulates
/// (+=) into the patch matrix and normalizes by the *input* point's neighbor
/// count/importance sum instead of the output point's, matching the
/// transpose-convolution's scatter-style column construction. Depends on
/// \p deps and returns the completion event of its last kernel.
template <class TReal, class TIndex, class TKernelIndex>
sycl::event FillColumnTransposeSYCL(
        sycl::queue& queue,
        TReal* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        TIndex num_inp,
        const TReal* const inp_features,
        const TReal* const inp_neighbors_importance_sum,
        const int64_t* const inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* const neighbors_index,
        const TKernelIndex* const neighbors_kernel_index,
        const TReal* const neighbors_importance,
        const int64_t* const neighbors_row_splits,
        const int num_kernel_elements,
        bool normalize,
        const std::vector<sycl::event>& deps = {});

}  // namespace impl
}  // namespace ml
}  // namespace open3d
