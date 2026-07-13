// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of ReduceSubarraysSum — ports ReduceSubarraysSum.cuh.
// One work-item per sub-array: serial sum over
// values[row_splits[i]..row_splits[i+1]].

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace impl {

/// Each work-item i sums values[row_splits[i]..row_splits[i+1]) into
/// out_sums[i].
template <class T>
void ReduceSubarraysSumSYCL(sycl::queue& queue,
                            const T* const values,
                            const size_t values_size,
                            const int64_t* const row_splits,
                            const size_t num_arrays,
                            T* out_sums) {
    if (num_arrays == 0) return;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_arrays), [=](sycl::item<1> item) {
            const size_t i = item.get_id(0);
            const size_t begin_idx = static_cast<size_t>(row_splits[i]);
            const size_t end_idx = static_cast<size_t>(row_splits[i + 1]);

            T sum = T(0);
            for (size_t j = begin_idx; j < end_idx; ++j) {
                sum += values[j];
            }
            out_sums[i] = sum;
        });
    });
    queue.wait_and_throw();
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
