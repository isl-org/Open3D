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
// original path: faiss/faiss/gpu/utils/MergeNetworkBlock.cuh
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/nns/kernel/DeviceDefs.cuh"
#include "open3d/core/nns/kernel/StaticUtils.cuh"

namespace open3d {
namespace core {

// Merge pairs of lists smaller than blockDim.x (NumThreads)
template <int NumThreads,
          typename K,
          typename V,
          int N,
          int L,
          bool AllThreads,
          bool Dir,
          bool FullMerge>
inline __device__ void blockMergeSmall(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L <= NumThreads, "merge list size must be <= NumThreads");

    // Which pair of lists we are merging
    int mergeId = threadIdx.x / L;

    // Which thread we are within the merge
    int tid = threadIdx.x % L;

    // listK points to a region of size N * 2 * L
    listK += 2 * L * mergeId;
    listV += 2 * L * mergeId;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
    int pos = L - 1 - tid;
    int stride = 2 * tid + 1;

    if (AllThreads || (threadIdx.x < N * L)) {
        K ka = listK[pos];
        K kb = listK[pos + stride];

        // bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        bool swap = Dir ? ka > kb : ka < kb;
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        V va = listV[pos];
        V vb = listV[pos + stride];
        listV[pos] = swap ? vb : va;
        listV[pos + stride] = swap ? va : vb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
        int pos = 2 * tid - (tid & (stride - 1));

        if (AllThreads || (threadIdx.x < N * L)) {
            K ka = listK[pos];
            K kb = listK[pos + stride];

            // bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            bool swap = Dir ? ka > kb : ka < kb;
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            V va = listV[pos];
            V vb = listV[pos + stride];
            listV[pos] = swap ? vb : va;
            listV[pos + stride] = swap ? va : vb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

// Merge pairs of sorted lists larger than blockDim.x (NumThreads)
template <int NumThreads,
          typename K,
          typename V,
          int L,
          bool Dir,
          bool FullMerge>
inline __device__ void blockMergeLarge(K* listK, V* listV) {
    static_assert(isPowerOf2(L), "L must be a power-of-2");
    static_assert(L >= kWarpSize, "merge list size must be >= 32");
    static_assert(isPowerOf2(NumThreads), "NumThreads must be a power-of-2");
    static_assert(L >= NumThreads, "merge list size must be >= NumThreads");

    // For L > NumThreads, each thread has to perform more work
    // per each stride.
    constexpr int kLoopPerThread = L / NumThreads;

    // It's not a bitonic merge, both lists are in the same direction,
    // so handle the first swap assuming the second list is reversed
#pragma unroll
    for (int loop = 0; loop < kLoopPerThread; ++loop) {
        int tid = loop * NumThreads + threadIdx.x;
        int pos = L - 1 - tid;
        int stride = 2 * tid + 1;

        K ka = listK[pos];
        K kb = listK[pos + stride];

        // bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        bool swap = Dir ? ka > kb : ka < kb;
        listK[pos] = swap ? kb : ka;
        listK[pos + stride] = swap ? ka : kb;

        V va = listV[pos];
        V vb = listV[pos + stride];
        listV[pos] = swap ? vb : va;
        listV[pos + stride] = swap ? va : vb;

        // FIXME: is this a CUDA 9 compiler bug?
        // K& ka = listK[pos];
        // K& kb = listK[pos + stride];

        // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
        // swap(s, ka, kb);

        // V& va = listV[pos];
        // V& vb = listV[pos + stride];
        // swap(s, va, vb);
    }

    __syncthreads();

    constexpr int kSecondLoopPerThread =
            FullMerge ? kLoopPerThread : kLoopPerThread / 2;

#pragma unroll
    for (int stride = L / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int loop = 0; loop < kSecondLoopPerThread; ++loop) {
            int tid = loop * NumThreads + threadIdx.x;
            int pos = 2 * tid - (tid & (stride - 1));

            K ka = listK[pos];
            K kb = listK[pos + stride];

            // bool swap = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            bool swap = Dir ? ka > kb : ka < kb;
            listK[pos] = swap ? kb : ka;
            listK[pos + stride] = swap ? ka : kb;

            V va = listV[pos];
            V vb = listV[pos + stride];
            listV[pos] = swap ? vb : va;
            listV[pos + stride] = swap ? va : vb;

            // FIXME: is this a CUDA 9 compiler bug?
            // K& ka = listK[pos];
            // K& kb = listK[pos + stride];

            // bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
            // swap(s, ka, kb);

            // V& va = listV[pos];
            // V& vb = listV[pos + stride];
            // swap(s, va, vb);
        }

        __syncthreads();
    }
}

/// Class template to prevent static_assert from firing for
/// mixing smaller/larger than block cases
template <int NumThreads,
          typename K,
          typename V,
          int N,
          int L,
          bool Dir,
          bool SmallerThanBlock,
          bool FullMerge>
struct BlockMerge {};

/// Merging lists smaller than a block
template <int NumThreads,
          typename K,
          typename V,
          int N,
          int L,
          bool Dir,
          bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, true, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        constexpr int kNumParallelMerges = NumThreads / L;
        constexpr int kNumIterations = N / kNumParallelMerges;

        static_assert(L <= NumThreads, "list must be <= NumThreads");
        static_assert((N < kNumParallelMerges) ||
                              (kNumIterations * kNumParallelMerges == N),
                      "improper selection of N and L");

        if (N < kNumParallelMerges) {
            // We only need L threads per each list to perform the merge
            blockMergeSmall<NumThreads, K, V, N, L, false, Dir, FullMerge>(
                    listK, listV);
        } else {
            // All threads participate
#pragma unroll
            for (int i = 0; i < kNumIterations; ++i) {
                int start = i * kNumParallelMerges * 2 * L;

                blockMergeSmall<NumThreads, K, V, N, L, true, Dir, FullMerge>(
                        listK + start, listV + start);
            }
        }
    }
};

/// Merging lists larger than a block
template <int NumThreads,
          typename K,
          typename V,
          int N,
          int L,
          bool Dir,
          bool FullMerge>
struct BlockMerge<NumThreads, K, V, N, L, Dir, false, FullMerge> {
    static inline __device__ void merge(K* listK, V* listV) {
        // Each pair of lists is merged sequentially
#pragma unroll
        for (int i = 0; i < N; ++i) {
            int start = i * 2 * L;

            blockMergeLarge<NumThreads, K, V, L, Dir, FullMerge>(listK + start,
                                                                 listV + start);
        }
    }
};

template <int NumThreads,
          typename K,
          typename V,
          int N,
          int L,
          bool Dir,
          bool FullMerge = true>
inline __device__ void blockMerge(K* listK, V* listV) {
    constexpr bool kSmallerThanBlock = (L <= NumThreads);

    BlockMerge<NumThreads, K, V, N, L, Dir, kSmallerThanBlock,
               FullMerge>::merge(listK, listV);
}

}  // namespace core
}  // namespace open3d
