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
// MIT License
//
// Copyright (c) Facebook, Inc. and its affiliates.
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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ----------------------------------------------------------------------------
// original path: faiss/faiss/gpu/utils/Reductions.cuh
// ----------------------------------------------------------------------------

#pragma once

#include <cuda.h>

#include "open3d/core/nns/kernel/BlockMerge.cuh"
#include "open3d/core/nns/kernel/PtxUtils.cuh"
#include "open3d/core/nns/kernel/ReductionOps.cuh"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {

template <typename T, typename Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAll(T val, Op op) {
#pragma unroll
    for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
        val = op(val, shfl_xor(val, mask));
    }

    return val;
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
    return warpReduceAll<T, Sum<T>, ReduceWidth>(val, Sum<T>());
}

/// Performs a block-wide reduction
template <typename T, typename Op, bool BroadcastAll, bool KillWARDependency>
__device__ inline T blockReduceAll(T val, Op op, T* smem) {
    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;

    val = warpReduceAll<T, Op>(val, op);
    if (laneId == 0) {
        smem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = laneId < divUp(blockDim.x, kWarpSize) ? smem[laneId]
                                                    : op.identity();
        val = warpReduceAll<T, Op>(val, op);

        if (BroadcastAll) {
            __threadfence_block();

            if (laneId == 0) {
                smem[0] = val;
            }
        }
    }

    if (BroadcastAll) {
        __syncthreads();
        val = smem[0];
    }

    if (KillWARDependency) {
        __syncthreads();
    }

    return val;
}

/// Performs a block-wide reduction of multiple values simultaneously
template <int Num,
          typename T,
          typename Op,
          bool BroadcastAll,
          bool KillWARDependency>
__device__ inline void blockReduceAll(T val[Num], Op op, T* smem) {
    int laneId = getLaneId();
    int warpId = threadIdx.x / kWarpSize;

#pragma unroll
    for (int i = 0; i < Num; ++i) {
        val[i] = warpReduceAll<T, Op>(val[i], op);
    }

    if (laneId == 0) {
#pragma unroll
        for (int i = 0; i < Num; ++i) {
            smem[warpId * Num + i] = val[i];
        }
    }

    __syncthreads();

    if (warpId == 0) {
#pragma unroll
        for (int i = 0; i < Num; ++i) {
            val[i] = laneId < divUp(blockDim.x, kWarpSize)
                             ? smem[laneId * Num + i]
                             : op.identity();
            val[i] = warpReduceAll<T, Op>(val[i], op);
        }

        if (BroadcastAll) {
            __threadfence_block();

            if (laneId == 0) {
#pragma unroll
                for (int i = 0; i < Num; ++i) {
                    smem[i] = val[i];
                }
            }
        }
    }

    if (BroadcastAll) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < Num; ++i) {
            val[i] = smem[i];
        }
    }

    if (KillWARDependency) {
        __syncthreads();
    }
}

/// Sums a register value across the entire block
template <typename T, bool BroadcastAll, bool KillWARDependency>
__device__ inline T blockReduceAllSum(T val, T* smem) {
    return blockReduceAll<T, Sum<T>, BroadcastAll, KillWARDependency>(
            val, Sum<T>(), smem);
}

template <int Num, typename T, bool BroadcastAll, bool KillWARDependency>
__device__ inline void blockReduceAllSum(T vals[Num], T* smem) {
    return blockReduceAll<Num, T, Sum<T>, BroadcastAll, KillWARDependency>(
            vals, Sum<T>(), smem);
}

}  // namespace core
}  // namespace open3d
