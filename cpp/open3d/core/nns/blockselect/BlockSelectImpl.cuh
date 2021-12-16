#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/nns/BlockSelect.cuh"
#include "open3d/core/nns/Limits.cuh"

#define BLOCK_SELECT_DECL(TYPE, DIR, WARP_Q)                               \
    extern void runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(               \
            TYPE* in, TYPE* outK, int32_t* outV, bool dir, int k, int dim, \
            int num_points, cudaStream_t stream);                          \
                                                                           \
    extern void runBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_(           \
            TYPE* inK, int32_t* inV, TYPE* outK, int32_t* outV, bool dir,  \
            int k, int dim, int num_points, cudaStream_t stream);

#define BLOCK_SELECT_IMPL(TYPE, DIR, WARP_Q, THREAD_Q)                        \
    void runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(                         \
            TYPE* in, TYPE* outK, int32_t* outV, bool dir, int k, int dim,    \
            int num_points, cudaStream_t stream) {                            \
        auto grid = dim3(num_points);                                         \
                                                                              \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;   \
        auto block = dim3(kBlockSelectNumThreads);                            \
                                                                              \
        OPEN3D_ASSERT(k <= WARP_Q);                                           \
        OPEN3D_ASSERT(dir == DIR);                                            \
                                                                              \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();   \
        auto vInit = -1;                                                      \
                                                                              \
        blockSelect<TYPE, int, DIR, WARP_Q, THREAD_Q, kBlockSelectNumThreads> \
                <<<grid, block, 0, stream>>>(in, outK, outV, kInit, vInit, k, \
                                             dim, num_points);                \
    }                                                                         \
                                                                              \
    void runBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_(                     \
            TYPE* inK, int32_t* inV, TYPE* outK, int32_t* outV, bool dir,     \
            int k, int dim, int num_points, cudaStream_t stream) {            \
        auto grid = dim3(num_points);                                         \
                                                                              \
        constexpr int kBlockSelectNumThreads = (WARP_Q <= 1024) ? 128 : 64;   \
        auto block = dim3(kBlockSelectNumThreads);                            \
                                                                              \
        OPEN3D_ASSERT(k <= WARP_Q);                                           \
        OPEN3D_ASSERT(dir == DIR);                                            \
                                                                              \
        auto kInit = dir ? Limits<TYPE>::getMin() : Limits<TYPE>::getMax();   \
        auto vInit = -1;                                                      \
                                                                              \
        blockSelectPair<TYPE, int, DIR, WARP_Q, THREAD_Q,                     \
                        kBlockSelectNumThreads><<<grid, block, 0, stream>>>(  \
                inK, inV, outK, outV, kInit, vInit, k, dim, num_points);      \
    }

#define BLOCK_SELECT_CALL(TYPE, DIR, WARP_Q)                                 \
    runBlockSelect_##TYPE##_##DIR##_##WARP_Q##_(in, outK, outV, dir, k, dim, \
                                                num_points, stream)

#define BLOCK_SELECT_PAIR_CALL(TYPE, DIR, WARP_Q)    \
    runBlockSelectPair_##TYPE##_##DIR##_##WARP_Q##_( \
            inK, inV, outK, outV, dir, k, dim, num_points, stream)
