// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of ReduceSubarraysSum — ports ReduceSubarraysSum.cuh.
// One work-group per sub-array: work-items grid-stride over
// values[row_splits[i]..row_splits[i+1]) accumulating a partial sum, then
// sycl::reduce_over_group combines them (same pattern as FillColumnSYCL's
// normalizer in impl/sparse_conv/SparseConvSYCLKernels.cpp). Test tolerance
// for floating-point dtypes (rtol=1e-5) permits the reassociated summation
// order; integer dtypes remain exact.

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace impl {

namespace {
// Work-group size for the per-sub-array reduction: 256, a sensible default
// for launches that have no hardware-specific tuning of their own (unlike
// e.g. the conv FillColumn kernels' warp-per-point=32, which deliberately
// mirrors the CUDA design).
constexpr size_t kReduceSubarraysSumWGSize = 256;
}  // namespace

/// Each work-group i sums values[row_splits[i]..row_splits[i+1]) into
/// out_sums[i].
template <class T>
void ReduceSubarraysSumSYCL(sycl::queue& queue,
                            const T* const values,
                            const size_t values_size,
                            const int64_t* const row_splits,
                            const size_t num_arrays,
                            T* out_sums) {
    if (num_arrays == 0) return;

    const size_t wg = kReduceSubarraysSumWGSize;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_arrays * wg),
                                  sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) {
                    const size_t i = item.get_group(0);
                    const size_t lid = item.get_local_id(0);
                    const size_t begin_idx = static_cast<size_t>(row_splits[i]);
                    const size_t end_idx =
                            static_cast<size_t>(row_splits[i + 1]);

                    T local_sum = T(0);
                    for (size_t j = begin_idx + lid; j < end_idx; j += wg) {
                        local_sum += values[j];
                    }
                    T sum = sycl::reduce_over_group(item.get_group(), local_sum,
                                                    sycl::plus<T>());
                    if (lid == 0) {
                        out_sums[i] = sum;
                    }
                });
    });
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
