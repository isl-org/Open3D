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
// original path: faiss/faiss/gpu/impl/L2Select.cu
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/kernel/Limits.cuh"
#include "open3d/core/nns/kernel/Pair.cuh"
#include "open3d/core/nns/kernel/Reduction.cuh"
#include "open3d/core/nns/kernel/ReductionOps.cuh"
#include "open3d/core/nns/kernel/Select.cuh"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {
namespace nns {

// L2 + select kernel for k == 1, implements re-use of ||c||^2
template <typename T, int kRowsPerBlock, int kBlockSize>
__global__ void l2SelectMin1(T* productDistances,
                             T* centroidDistances,
                             T* outDistances,
                             int32_t* outIndices,
                             int num_points,
                             int dim) {
    // Each block handles kRowsPerBlock rows of the distances (results)
    Pair<T, int> threadMin[kRowsPerBlock];
    __shared__ Pair<T, int> blockMin[kRowsPerBlock * (kBlockSize / kWarpSize)];

    T distance[kRowsPerBlock];

#pragma unroll
    for (int i = 0; i < kRowsPerBlock; ++i) {
        threadMin[i].k = Limits<T>::getMax();
        threadMin[i].v = -1;
    }

    // blockIdx.x: which chunk of rows we are responsible for updating
    int rowStart = blockIdx.x * kRowsPerBlock;

    // FIXME: if we have exact multiples, don't need this
    bool endRow = (blockIdx.x == gridDim.x - 1);

    if (endRow) {
        if (num_points % kRowsPerBlock == 0) {
            endRow = false;
        }
    }

    if (endRow) {
        for (int row = rowStart; row < num_points; ++row) {
            for (int col = threadIdx.x; col < dim; col += blockDim.x) {
                distance[0] = centroidDistances[col] +
                              productDistances[row + dim + col];

                if (distance[0] < threadMin[0].k) {
                    threadMin[0].k = distance[0];
                    threadMin[0].v = col;
                }
            }

            // Reduce within the block
            threadMin[0] = blockReduceAll<Pair<T, int>, Min<Pair<T, int>>,
                                          false, false>(
                    threadMin[0], Min<Pair<T, int>>(), blockMin);

            if (threadIdx.x == 0) {
                outDistances[row + 0] = threadMin[0].k;
                outIndices[row + 0] = threadMin[0].v;
            }

            // so we can use the shared memory again
            __syncthreads();

            threadMin[0].k = Limits<T>::getMax();
            threadMin[0].v = -1;
        }
    } else {
        for (int col = threadIdx.x; col < dim; col += blockDim.x) {
            T centroidDistance = centroidDistances[col];

#pragma unroll
            for (int row = 0; row < kRowsPerBlock; ++row) {
                distance[row] = productDistances[(rowStart + row) * dim + col];
            }

#pragma unroll
            for (int row = 0; row < kRowsPerBlock; ++row) {
                distance[row] = distance[row] + centroidDistance;
            }

#pragma unroll
            for (int row = 0; row < kRowsPerBlock; ++row) {
                if (distance[row] < threadMin[row].k) {
                    threadMin[row].k = distance[row];
                    threadMin[row].v = col;
                }
            }
        }

        // Reduce within the block
        blockReduceAll<kRowsPerBlock, Pair<T, int>, Min<Pair<T, int>>, false,
                       false>(threadMin, Min<Pair<T, int>>(), blockMin);

        if (threadIdx.x == 0) {
#pragma unroll
            for (int row = 0; row < kRowsPerBlock; ++row) {
                outDistances[rowStart + row + 0] = threadMin[row].k;
                outIndices[rowStart + row + 0] = threadMin[row].v;
            }
        }
    }
}

// L2 + select kernel for k > 1, no re-use of ||c||^2
template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void l2SelectMinK(T* productDistances,
                             T* centroidDistances,
                             T* outDistances,
                             int32_t* outIndices,
                             int k,
                             int dim,
                             int num_cols,
                             int tile_cols,
                             T initK) {
    // Each block handles a single row of the distances (results)
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ T smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];

    BlockSelect<T, int, false, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
            initK, -1, smemK, smemV, k);

    int row = blockIdx.x;

    // Whole warps must participate in the selection
    // int limit = utils::roundDown(dim, kWarpSize);
    int limit = (dim / kWarpSize) * kWarpSize;
    int i = threadIdx.x;

    for (; i < limit; i += blockDim.x) {
        T v = centroidDistances[i] + productDistances[row * tile_cols + i];
        heap.add(v, i);
    }

    if (i < dim) {
        T v = centroidDistances[i] + productDistances[row * tile_cols + i];
        heap.addThreadQ(v, i);
    }

    heap.reduce();
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outDistances[row * k * num_cols + i] = smemK[i];
        outIndices[row * k * num_cols + i] = smemV[i];
    }
}

template <typename T>
void runL2SelectMin(const cudaStream_t stream,
                    Tensor& productDistances,
                    Tensor& centroidDistances,
                    Tensor& outDistances,
                    Tensor& outIndices,
                    int k,
                    int num_cols,
                    int tile_cols) {
    OPEN3D_ASSERT(productDistances.GetShape(0) == outDistances.GetShape(0));
    OPEN3D_ASSERT(productDistances.GetShape(0) == outIndices.GetShape(0));
    OPEN3D_ASSERT(centroidDistances.GetShape(0) ==
                  productDistances.GetShape(1));
    OPEN3D_ASSERT(outDistances.GetShape(1) == k);
    OPEN3D_ASSERT(outIndices.GetShape(1) == k);
    OPEN3D_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (k == 1) {
        constexpr int kThreadsPerBlock = 256;
        constexpr int kRowsPerBlock = 8;

        auto block = dim3(kThreadsPerBlock);
        auto grid =
                dim3(utility::DivUp(outDistances.GetShape(0), kRowsPerBlock));

        l2SelectMin1<T, kRowsPerBlock, kThreadsPerBlock>
                <<<grid, block, 0, stream>>>(productDistances.GetDataPtr<T>(),
                                             centroidDistances.GetDataPtr<T>(),
                                             outDistances.GetDataPtr<T>(),
                                             outIndices.GetDataPtr<int32_t>(),
                                             (int)productDistances.GetShape(0),
                                             (int)productDistances.GetShape(1));
    } else {
        auto grid = dim3(outDistances.GetShape(0));

#define RUN_L2_SELECT(BLOCK, NUM_WARP_Q, NUM_THREAD_Q)                     \
    do {                                                                   \
        l2SelectMinK<T, NUM_WARP_Q, NUM_THREAD_Q, BLOCK>                   \
                <<<grid, BLOCK, 0, stream>>>(                              \
                        productDistances.GetDataPtr<T>(),                  \
                        centroidDistances.GetDataPtr<T>(),                 \
                        outDistances.GetDataPtr<T>(),                      \
                        outIndices.GetDataPtr<int32_t>(), k,               \
                        productDistances.GetShape(1), num_cols, tile_cols, \
                        Limits<T>::getMax());                              \
    } while (0)

        // block size 128 for everything <= 1024
        if (k <= 32) {
            RUN_L2_SELECT(128, 32, 2);
        } else if (k <= 64) {
            RUN_L2_SELECT(128, 64, 3);
        } else if (k <= 128) {
            RUN_L2_SELECT(128, 128, 3);
        } else if (k <= 256) {
            RUN_L2_SELECT(128, 256, 4);
        } else if (k <= 512) {
            RUN_L2_SELECT(128, 512, 8);
        } else if (k <= 1024) {
            RUN_L2_SELECT(128, 1024, 8);

#if GPU_MAX_SELECTION_K >= 2048
        } else if (k <= 2048) {
            // smaller block for less shared memory
            RUN_L2_SELECT(64, 2048, 8);
#endif

        } else {
            OPEN3D_ASSERT(false);
        }
    }

    // CUDA_TEST_ERROR();
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
