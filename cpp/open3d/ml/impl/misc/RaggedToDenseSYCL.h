// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of RaggedToDense — ports RaggedToDense.cuh to DPC++.
// One work-item per row: copy values into the dense output and pad with
// default_value when the row is shorter than out_col_size.

#pragma once

#include <sycl/sycl.hpp>

namespace open3d {
namespace ml {
namespace impl {

/// SYCL kernel body for RaggedToDense.  Each work-item handles one row of the
/// ragged tensor.
template <class T>
void RaggedToDenseSYCL(sycl::queue& queue,
                       const T* const values,
                       const int64_t* const row_splits,
                       const size_t row_splits_size,
                       const size_t out_col_size,
                       const T* const default_value,
                       const size_t default_value_size,
                       T* out_values) {
    if (row_splits_size <= 1) return;

    const size_t num_rows = row_splits_size - 1;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_rows), [=](sycl::item<1> item) {
            const size_t i = item.get_id(0);

            const int64_t start = row_splits[i];
            const int64_t end_raw = row_splits[i + 1];
            const int64_t end = sycl::min(
                    static_cast<int64_t>(out_col_size) + start, end_raw);

            // Copy valid values
            T* out_ptr = out_values + i * out_col_size * default_value_size;
            for (int64_t inp_idx =
                         start * static_cast<int64_t>(default_value_size);
                 inp_idx < end * static_cast<int64_t>(default_value_size);
                 ++inp_idx, ++out_ptr) {
                *out_ptr = values[inp_idx];
            }

            // Pad remaining columns with default_value
            out_ptr = out_values + i * out_col_size * default_value_size +
                      (end - start) * static_cast<int64_t>(default_value_size);
            for (int64_t j = end - start;
                 j < static_cast<int64_t>(out_col_size);
                 ++j, out_ptr += default_value_size) {
                for (size_t k = 0; k < default_value_size; ++k) {
                    out_ptr[k] = default_value[k];
                }
            }
        });
    });
    queue.wait_and_throw();
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
