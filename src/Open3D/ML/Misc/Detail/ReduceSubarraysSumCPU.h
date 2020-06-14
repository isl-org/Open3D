// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#pragma once

#include "tbb/parallel_for.h"

namespace open3d {
namespace ml {
namespace detail {

/// Reduces subarrays in linear memory with the sum operation.
/// The sum for empty subarrays is 0.
///
/// \param values          The linear array with all values
/// \param values_size     Number of elements of \p values
/// \param prefix_sum      The exclusive prefix sum of the number of elements
///                        for each array
/// \param prefix_sum_size The number of subarrays
/// \param out_sums        The preallocated output array with size
///                        \p prefix_sum_size
template <class T>
void ReduceSubarraysSumCPU(const T* const values,
                           const size_t values_size,
                           const int64_t* const prefix_sum,
                           const size_t prefix_sum_size,
                           T* out_sums) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, prefix_sum_size),
                      [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              size_t begin_idx = prefix_sum[i];
                              // use values_size as end_idx for the last
                              // subarray
                              size_t end_idx = (i + 1 < prefix_sum_size
                                                        ? prefix_sum[i + 1]
                                                        : values_size);

                              T sum = T(0);
                              for (size_t j = begin_idx; j < end_idx; ++j) {
                                  sum += values[j];
                              }
                              out_sums[i] = sum;
                          }
                      });
}

}  // namespace detail
}  // namespace ml
}  // namespace open3d
