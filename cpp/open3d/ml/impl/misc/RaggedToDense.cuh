// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <cub/cub.cuh>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {

/// Kernel for RaggedToDenseCUDA
template <class T>
__global__ void RaggedToDenseCUDAKernel(
        const T* const __restrict__ values,
        const int64_t* const __restrict__ row_splits,
        const size_t row_splits_size,
        const size_t out_col_size,
        const T* const __restrict__ default_value,
        const size_t default_value_size,
        T* __restrict__ out_values) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= row_splits_size) return;

    const int64_t start = row_splits[i];
    const int64_t end = min(int64_t(out_col_size) + start, row_splits[i + 1]);

    T* out_ptr = out_values + i * out_col_size * default_value_size;

    for (int64_t inp_idx = start * default_value_size;
         inp_idx < end * default_value_size; ++inp_idx, ++out_ptr) {
        *out_ptr = values[inp_idx];
    }

    // fill remaining columns with the default value
    out_ptr = out_values + i * out_col_size * default_value_size;
    out_ptr = out_ptr + (end - start) * default_value_size;
    for (int64_t j = end - start; j < out_col_size;
         ++j, out_ptr += default_value_size) {
        for (int64_t k = 0; k < default_value_size; ++k) {
            out_ptr[k] = default_value[k];
        }
    }
}
}  // namespace

/// Creates a dense tensor from a ragged tensor.
/// All pointer arguments point to device memory unless stated otherwise.
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
void RaggedToDenseCUDA(const cudaStream_t& stream,
                       const T* const values,
                       const int64_t* const row_splits,
                       const size_t row_splits_size,
                       const size_t out_col_size,
                       const T* const default_value,
                       const size_t default_value_size,
                       T* out_values) {
    using namespace open3d::utility;
    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(DivUp(row_splits_size - 1, block.x));

    if (grid.x) {
        RaggedToDenseCUDAKernel<T><<<grid, block, 0, stream>>>(
                values, row_splits, row_splits_size, out_col_size,
                default_value, default_value_size, out_values);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
