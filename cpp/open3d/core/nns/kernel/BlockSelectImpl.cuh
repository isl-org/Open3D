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
// original path: faiss/faiss/gpu/utils/blockselect/BlockSelectImpl.cuh
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/nns/kernel/BlockSelect.cuh"
#include "open3d/core/nns/kernel/Limits.cuh"

#define BLOCK_SELECT_IMPL(TYPE, TINDEX, DIR, WARP_Q, THREAD_Q)                 \
    void runBlockSelect_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_(               \
            cudaStream_t stream, TYPE* in, TYPE* outK, TINDEX* outV, bool dir, \
            int k, int dim, int num_points) {                                  \
        auto grid = dim3(num_points);                                          \
                                                                               \
        constexpr int kBlockSelectNumThreads =                                 \
                sizeof(TYPE) == 4 ? ((WARP_Q <= 1024) ? 128 : 64)              \
                                  : ((WARP_Q <= 512) ? 64 : 32);               \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        OPEN3D_ASSERT(k <= WARP_Q);                                            \
        OPEN3D_ASSERT(dir == DIR);                                             \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelect<TYPE, TINDEX, DIR, WARP_Q, THREAD_Q,                       \
                    kBlockSelectNumThreads><<<grid, block, 0, stream>>>(       \
                in, outK, outV, kInit, vInit, k, dim, num_points);             \
    }                                                                          \
                                                                               \
    void runBlockSelectPair_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_(           \
            cudaStream_t stream, TYPE* inK, TINDEX* inV, TYPE* outK,           \
            TINDEX* outV, bool dir, int k, int dim, int num_points) {          \
        auto grid = dim3(num_points);                                          \
                                                                               \
        constexpr int kBlockSelectNumThreads =                                 \
                sizeof(TYPE) == 4 ? ((WARP_Q <= 1024) ? 128 : 64)              \
                                  : ((WARP_Q <= 512) ? 64 : 32);               \
        auto block = dim3(kBlockSelectNumThreads);                             \
                                                                               \
        OPEN3D_ASSERT(k <= WARP_Q);                                            \
        OPEN3D_ASSERT(dir == DIR);                                             \
                                                                               \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();    \
        auto vInit = -1;                                                       \
                                                                               \
        blockSelectPair<TYPE, TINDEX, DIR, WARP_Q, THREAD_Q,                   \
                        kBlockSelectNumThreads><<<grid, block, 0, stream>>>(   \
                inK, inV, outK, outV, kInit, vInit, k, dim, num_points);       \
    }

#define BLOCK_SELECT_CALL(TYPE, TINDEX, DIR, WARP_Q)        \
    runBlockSelect_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_( \
            stream, in, outK, outV, dir, k, dim, num_points)

#define BLOCK_SELECT_PAIR_CALL(TYPE, TINDEX, DIR, WARP_Q)       \
    runBlockSelectPair_##TYPE##_##TINDEX##_##DIR##_##WARP_Q##_( \
            stream, inK, inV, outK, outV, dir, k, dim, num_points)
