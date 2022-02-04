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
// original path: faiss/faiss/gpu/utils/BlockSelectKernel.cuh
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/kernel/Select.cuh"

namespace open3d {
namespace core {

template <typename K,
          typename IndexType,
          bool Dir,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
__global__ void blockSelect(K* in,
                            K* outK,
                            IndexType* outV,
                            K initK,
                            IndexType initV,
                            int k,
                            int dim,
                            int num_points) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<K, IndexType, Dir, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
            initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;

    int i = threadIdx.x;
    K* inStart = in + dim * row + i;

    // Whole warps must participate in the selection
    int limit = (dim / kWarpSize) * kWarpSize;

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inStart, (IndexType)i);
        inStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < dim) {
        heap.addThreadQ(*inStart, (IndexType)i);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        *(outK + row * dim + i) = smemK[i];
        *(outV + row * dim + i) = smemV[i];
    }
}

template <typename K,
          typename IndexType,
          bool Dir,
          int NumWarpQ,
          int NumThreadQ,
          int ThreadsPerBlock>
__global__ void blockSelectPair(K* inK,
                                IndexType* inV,
                                K* outK,
                                IndexType* outV,
                                K initK,
                                IndexType initV,
                                int k,
                                int dim,
                                int num_points) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ K smemK[kNumWarps * NumWarpQ];
    __shared__ IndexType smemV[kNumWarps * NumWarpQ];

    BlockSelect<K, IndexType, Dir, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(
            initK, initV, smemK, smemV, k);

    // Grid is exactly sized to rows available
    int row = blockIdx.x;

    int i = threadIdx.x;
    K* inKStart = &inK[row * dim + i];
    IndexType* inVStart = &inV[row * dim + i];

    // Whole warps must participate in the selection
    int limit = (dim / kWarpSize) * kWarpSize;

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(*inKStart, *inVStart);
        inKStart += ThreadsPerBlock;
        inVStart += ThreadsPerBlock;
    }

    // Handle last remainder fraction of a warp of elements
    if (i < dim) {
        heap.addThreadQ(*inKStart, *inVStart);
    }

    heap.reduce();

    for (int i = threadIdx.x; i < k; i += ThreadsPerBlock) {
        outK[row * k + i] = smemK[i];
        outV[row * k + i] = smemV[i];
    }
}

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int32_t* inV,
                        float* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        float* inK,
                        int64_t* inV,
                        float* outK,
                        int64_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        double* inK,
                        int32_t* inV,
                        double* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

void runBlockSelectPair(cudaStream_t stream,
                        double* inK,
                        int64_t* inV,
                        double* outK,
                        int64_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points);

}  // namespace core
}  // namespace open3d
