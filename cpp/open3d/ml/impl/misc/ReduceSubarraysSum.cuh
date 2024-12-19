// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/utility/Helper.h"

using namespace open3d::utility;

namespace open3d {
namespace ml {
namespace impl {

/// Kernel for ReduceSubarraysSumCUDA
template <class T>
__global__ void ReduceSubarraysSumCUDAKernel(
        const T* const __restrict__ values,
        const size_t values_size,
        const int64_t* const __restrict__ row_splits,
        const size_t num_arrays,
        T* __restrict__ out_sums) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_arrays) return;

    size_t begin_idx = row_splits[i];
    size_t end_idx = row_splits[i + 1];

    T sum = T(0);

    for (size_t j = begin_idx; j < end_idx; ++j) {
        sum += values[j];
    }
    out_sums[i] = sum;
}

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
void ReduceSubarraysSumCUDA(const cudaStream_t& stream,
                            const T* const values,
                            const size_t values_size,
                            const int64_t* const row_splits,
                            const size_t num_arrays,
                            T* out_sums) {
    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(DivUp(num_arrays, block.x));

    if (grid.x) {
        ReduceSubarraysSumCUDAKernel<T><<<grid, block, 0, stream>>>(
                values, values_size, row_splits, num_arrays, out_sums);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
