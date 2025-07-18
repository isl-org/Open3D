// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/parallel_for.h>

namespace open3d {
namespace ml {
namespace impl {

/// Creates a dense tensor from a ragged tensor.
///
/// Example where each value has size 2:
///  values = [[0,0],[1,1],[2,2],[3,3],[4,4]]
///  row_splits = [0,2,5]
///  out_col_size=3
///  default_value=[-1,-1]
///  default_value_size = 2
///
///  will return
///
///  out_values = [[[0,0],[1,1],[-1,-1]], [[2,2],[3,3],[4,4]]]
///
///
/// \param values    Linear memory with all values.
///
/// \param row_splits    Defines the start and end of each entry in the ragged
///        tensor. This is an exclusive prefix sum with 0 as the first element
///        and the length of all values as the last element.
///
/// \param row_splits_size    The length of the row_splits vector.
///
/// \param out_col_size    The output column size. This is the second dim of
///        the dense output tensor.
///
/// \param default_value    The default value to use if there are not enough
///        values for filling the row.
///
/// \param default_value_size    The size of the default value.
///
/// \param out_values    This is the output array. The size of the array must
///        be [row_splits_size-1, out_col_size, default_value_size].
///
template <class T>
void RaggedToDenseCPU(const T* const values,
                      const int64_t* const row_splits,
                      const size_t row_splits_size,
                      const size_t out_col_size,
                      const T* const default_value,
                      const size_t default_value_size,
                      T* out_values) {
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, row_splits_size - 1),
            [&](const tbb::blocked_range<size_t>& r) {
                for (int64_t i = r.begin(); i != r.end(); ++i) {
                    const int64_t start = row_splits[i];
                    const int64_t end = std::min(int64_t(out_col_size) + start,
                                                 row_splits[i + 1]);

                    T* out_ptr =
                            out_values + i * out_col_size * default_value_size;

                    std::copy(values + start * default_value_size,
                              values + end * default_value_size, out_ptr);

                    // fill remaining columns with the default value
                    out_ptr = out_ptr + (end - start) * default_value_size;
                    for (int64_t j = end - start; j < out_col_size;
                         ++j, out_ptr += default_value_size) {
                        std::copy(default_value,
                                  default_value + default_value_size, out_ptr);
                    }
                }
            });
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
