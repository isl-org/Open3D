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

#include <tbb/parallel_for.h>

namespace open3d {
namespace ml {
namespace impl {

/// Reduces subarrays in linear memory with the sum operation.
/// The sum for empty subarrays is 0.
///
/// \param values          The linear array with all values
/// \param values_size     Number of elements of \p values
/// \param row_splits      Defines the start and end of each subarray. This is
///                        an exclusive prefix sum with 0 as the first element
///                        and the length of \p values as last element.
///                        The size is \p num_arrays + 1
/// \param num_arrays      The number of subarrays
/// \param out_sums        The preallocated output array with size
///                        \p num_arrays
template <class T>
void ReduceSubarraysSumCPU(const T* const values,
                           const size_t values_size,
                           const int64_t* const row_splits,
                           const size_t num_arrays,
                           T* out_sums) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_arrays),
                      [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              size_t begin_idx = row_splits[i];
                              size_t end_idx = row_splits[i + 1];

                              T sum = T(0);
                              for (size_t j = begin_idx; j < end_idx; ++j) {
                                  sum += values[j];
                              }
                              out_sums[i] = sum;
                          }
                      });
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
