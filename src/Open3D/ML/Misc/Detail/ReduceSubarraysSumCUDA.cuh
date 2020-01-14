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
#include "Open3D/Utility/Helper.h"

using namespace open3d::utility;

namespace open3d {
namespace ml {
namespace detail {

template <class T>
__global__ void ReduceSubarraysSumKernel(
        T* __restrict__ out_sums,
        const T* const __restrict__ values,
        size_t values_size,
        const int64_t* const __restrict__ prefix_sum,
        size_t prefix_sum_size) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= prefix_sum_size) return;

    size_t begin_idx = prefix_sum[i];
    size_t end_idx =
            (i + 1 < prefix_sum_size ? prefix_sum[i + 1] : values_size);

    T sum = T(0);

    for (size_t j = begin_idx; j < end_idx; ++j) {
        sum += values[j];
    }
    out_sums[i] = sum;
}

template <class T>
void ReduceSubarraysSumCUDA(const cudaStream_t& stream,
                            const T* const values,
                            const size_t values_size,
                            const int64_t* const prefix_sum,
                            const size_t prefix_sum_size,
                            T* out_sums) {
    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(DivUp(prefix_sum_size, block.x));

    if (grid.x) {
        ReduceSubarraysSumKernel<T><<<grid, block, 0, stream>>>(
                out_sums, values, values_size, prefix_sum, prefix_sum_size);
    }
}

}  // namespace detail
}  // namespace ml
}  // namespace open3d
