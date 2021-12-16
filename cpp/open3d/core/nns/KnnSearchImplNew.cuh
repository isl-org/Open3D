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

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace core {
namespace nns {
namespace impl {

// If the inner size (dim) of the vectors is small, we want a larger query tile
// size, like 1024
inline void chooseTileSize(int numQueries,
                           int numCentroids,
                           int dim,
                           int elementSize,
                           int &tileRows,
                           int &tileCols) {
    // The matrix multiplication should be large enough to be efficient, but if
    // it is too large, we seem to lose efficiency as opposed to
    // double-streaming. Each tile size here defines 1/2 of the memory use due
    // to double streaming. We ignore available temporary memory, as that is
    // adjusted independently by the user and can thus meet these requirements
    // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
    // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
    auto totalMem = GetCUDACurrentTotalMemSize();

    int targetUsage = 0;

    if (totalMem <= ((size_t)4) * 1024 * 1024 * 1024) {
        targetUsage = 512 * 1024 * 1024;
    } else if (totalMem <= ((size_t)8) * 1024 * 1024 * 1024) {
        targetUsage = 768 * 1024 * 1024;
    } else {
        targetUsage = 1024 * 1024 * 1024;
    }

    targetUsage /= 2 * elementSize;

    // 512 seems to be a batch size sweetspot for float32.
    // If we are on float16, increase to 512.
    // If the k size (vec dim) of the matrix multiplication is small (<= 32),
    // increase to 1024.
    int preferredTileRows = 512;
    if (dim <= 32) {
        preferredTileRows = 1024;
    }

    tileRows = std::min(preferredTileRows, numQueries);

    // tileCols is the remainder size
    tileCols = std::min(targetUsage / preferredTileRows, numCentroids);
}

template <class T, class TIndex, int NDIM>
void KnnQuery(const cudaStream_t &stream,
              TIndex *indices_ptr,
              T *distances_ptr,
              size_t num_points,
              const T *const points,
              size_t num_queries,
              const T *const queries,
              int knn) {
    // input: queries: (m, 3), points: (n, 3), idx: (m, knn)
}

template <class T, class OUTPUT_ALLOCATOR, int NDIM>
void KnnSearchCUDA(const cudaStream_t stream,
                   size_t num_points,
                   const T *const points,
                   size_t num_queries,
                   const T *const queries,
                   size_t points_row_splits_size,
                   const int64_t *const points_row_splits,
                   size_t queries_row_splits_size,
                   const int64_t *const queries_row_splits,
                   int knn,
                   OUTPUT_ALLOCATOR &output_allocator) {
    const int batch_size = points_row_splits_size - 1;

    const size_t num_indices = num_queries * knn;

    int32_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    for (int i = 0; i < batch_size; ++i) {
        const size_t num_queries_i =
                queries_row_splits[i + 1] - queries_row_splits[i];
        const size_t num_points_i =
                points_row_splits[i + 1] - points_row_splits[i];

        const T *const points_i = points + 3 * points_row_splits[i];
        const T *const queries_i = queries + 3 * queries_row_splits[i];
        int32_t *indices_ptr_i = indices_ptr + queries_row_splits[i] * knn;
        T *distances_ptr_i = distances_ptr + queries_row_splits[i] * knn;
        KnnQuery<T, int32_t, NDIM>(stream, indices_ptr_i, distances_ptr_i,
                                   num_points_i, points_i, num_queries_i,
                                   queries_i, knn);
    }
}

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d
