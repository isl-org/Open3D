// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

// Private header. Do not include in Open3d.h.

#pragma once

#include <cmath>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

template <typename scalar_t>
__device__ inline void WarpReduceSum(volatile scalar_t* local_sum,
                                     const int tid) {
    local_sum[tid] += local_sum[tid + 32];
    local_sum[tid] += local_sum[tid + 16];
    local_sum[tid] += local_sum[tid + 8];
    local_sum[tid] += local_sum[tid + 4];
    local_sum[tid] += local_sum[tid + 2];
    local_sum[tid] += local_sum[tid + 1];
}

template <typename scalar_t, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile scalar_t* local_sum) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum[tid] += local_sum[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum[tid] += local_sum[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum[tid] += local_sum[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        WarpReduceSum<scalar_t>(local_sum, tid);
    }
}

template <typename scalar_t, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile scalar_t* local_sum0,
                                      volatile scalar_t* local_sum1) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<scalar_t>(local_sum0, tid);
        WarpReduceSum<scalar_t>(local_sum1, tid);
    }
}

template <typename scalar_t, size_t BLOCK_SIZE>
__device__ inline void BlockReduceSum(const int tid,
                                      volatile scalar_t* local_sum0,
                                      volatile scalar_t* local_sum1,
                                      volatile scalar_t* local_sum2) {
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            local_sum0[tid] += local_sum0[tid + 256];
            local_sum1[tid] += local_sum1[tid + 256];
            local_sum2[tid] += local_sum2[tid + 256];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();
    }

    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceSum<scalar_t>(local_sum0, tid);
        WarpReduceSum<scalar_t>(local_sum1, tid);
        WarpReduceSum<scalar_t>(local_sum2, tid);
    }
}

template <typename scalar_t, size_t BLOCK_SIZE>
__device__ inline void ReduceSum6x6LinearSystem(const int tid,
                                                bool valid,
                                                const scalar_t* reduction,
                                                volatile scalar_t* local_sum0,
                                                volatile scalar_t* local_sum1,
                                                volatile scalar_t* local_sum2,
                                                scalar_t* global_sum) {
    // Sum reduction: JtJ(21) and Jtr(6)
    for (size_t i = 0; i < 27; i += 3) {
        local_sum0[tid] = valid ? reduction[i + 0] : 0;
        local_sum1[tid] = valid ? reduction[i + 1] : 0;
        local_sum2[tid] = valid ? reduction[i + 2] : 0;
        __syncthreads();

        BlockReduceSum<scalar_t, BLOCK_SIZE>(tid, local_sum0, local_sum1,
                                             local_sum2);

        if (tid == 0) {
            atomicAdd(&global_sum[i + 0], local_sum0[0]);
            atomicAdd(&global_sum[i + 1], local_sum1[0]);
            atomicAdd(&global_sum[i + 2], local_sum2[0]);
        }
        __syncthreads();
    }

    // Sum reduction: residual(1) and inlier(1)
    {
        local_sum0[tid] = valid ? reduction[27] : 0;
        local_sum1[tid] = valid ? reduction[28] : 0;
        __syncthreads();

        BlockReduceSum<scalar_t, BLOCK_SIZE>(tid, local_sum0, local_sum1);
        if (tid == 0) {
            atomicAdd(&global_sum[27], local_sum0[0]);
            atomicAdd(&global_sum[28], local_sum1[0]);
        }
        __syncthreads();
    }
}

template <typename scalar_t, size_t BLOCK_SIZE>
__device__ inline void ReduceSum6x6InformationJacobian(
        const int tid,
        bool valid,
        const scalar_t* reduction,
        volatile scalar_t* local_sum0,
        volatile scalar_t* local_sum1,
        volatile scalar_t* local_sum2,
        scalar_t* global_sum) {
    // Sum reduction: JtJ(21)
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = valid ? reduction[i + 0] : 0;
        local_sum1[tid] = valid ? reduction[i + 1] : 0;
        local_sum2[tid] = valid ? reduction[i + 2] : 0;
        __syncthreads();

        BlockReduceSum<scalar_t, BLOCK_SIZE>(tid, local_sum0, local_sum1,
                                             local_sum2);

        if (tid == 0) {
            atomicAdd(&global_sum[i + 0], local_sum0[0]);
            atomicAdd(&global_sum[i + 1], local_sum1[0]);
            atomicAdd(&global_sum[i + 2], local_sum2[0]);
        }
        __syncthreads();
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
