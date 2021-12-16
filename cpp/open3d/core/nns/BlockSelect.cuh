#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/Select.cuh"
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

    BlockSelect<K, IndexType, Dir, Comparator<K>, NumWarpQ, NumThreadQ,
                ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

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

    BlockSelect<K, IndexType, Dir, Comparator<K>, NumWarpQ, NumThreadQ,
                ThreadsPerBlock>
            heap(initK, initV, smemK, smemV, k);

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
        outK[row * dim + i] = smemK[i];
        outV[row * dim + i] = smemV[i];
    }
}

void runBlockSelectPair(float* inK,
                        int32_t* inV,
                        float* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points,
                        cudaStream_t stream);

void runBlockSelectPair(double* inK,
                        int32_t* inV,
                        double* outK,
                        int32_t* outV,
                        bool dir,
                        int k,
                        int dim,
                        int num_points,
                        cudaStream_t stream);

}  // namespace core
}  // namespace open3d