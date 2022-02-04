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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/KnnSearchImpl.cuh"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/kernel/BlockSelect.cuh"
#include "open3d/core/nns/kernel/DistancesUtils.cuh"
#include "open3d/core/nns/kernel/L2Select.cuh"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace core {
namespace nns {

#define CALL_KNN_BRUTE_FORCE(NDIM)                                  \
    impl::KnnQuery<T, TIndex, NDIM>(                                \
            stream, indices_ptr, distances_ptr, points.GetShape(0), \
            points.GetDataPtr<T>(), queries.GetShape(0),            \
            queries.GetDataPtr<T>(), knn);

template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void KnnSearchCUDABruteForce(const Tensor& points,
                             const Tensor& queries,
                             int knn,
                             OUTPUT_ALLOCATOR& output_allocator,
                             Tensor& query_neighbors_row_splits) {
    const cudaStream_t stream = cuda::GetStream();
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);

    // Return if input points are empty.
    if (num_points == 0 || num_queries == 0) {
        query_neighbors_row_splits.Fill(0);
        TIndex* indices_ptr;
        T* distances_ptr;

        output_allocator.AllocIndices(&indices_ptr, 0);
        output_allocator.AllocDistances(&distances_ptr, 0);
    }

    knn = num_points > knn ? knn : num_points;

    // Allocate output tensors.
    query_neighbors_row_splits.AsRvalue() =
            Tensor::Arange(0, num_queries * knn, knn);
    TIndex* indices_ptr;
    T* distances_ptr;
    const size_t num_indices = knn * num_queries;
    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    // Call kernel function.
    switch (points.GetShape(1)) {
        case 1:
            CALL_KNN_BRUTE_FORCE(1);
            break;
        case 2:
            CALL_KNN_BRUTE_FORCE(2);
            break;
        case 3:
            CALL_KNN_BRUTE_FORCE(3);
            break;
        case 4:
            CALL_KNN_BRUTE_FORCE(4);
            break;
        case 5:
            CALL_KNN_BRUTE_FORCE(5);
            break;
        case 6:
            CALL_KNN_BRUTE_FORCE(6);
            break;
        case 7:
            CALL_KNN_BRUTE_FORCE(7);
            break;
        case 8:
            CALL_KNN_BRUTE_FORCE(8);
            break;
        default:
            utility::LogError(
                    "KnnSearchCUDABruteForce only support data with dimension "
                    "1 to 8.");
            break;
    }
};

template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void KnnSearchCUDAOptimized(const Tensor& points,
                            const Tensor& queries,
                            int knn,
                            OUTPUT_ALLOCATOR& output_allocator,
                            Tensor& query_neighbors_row_splits) {
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    int dim = points.GetShape(1);
    Device device = points.GetDevice();
    Dtype dtype = points.GetDtype();

    // Return if input points are empty.
    if (num_points == 0 || num_queries == 0) {
        query_neighbors_row_splits.Fill(0);
        TIndex* indices_ptr;
        T* distances_ptr;

        output_allocator.AllocIndices(&indices_ptr, 0);
        output_allocator.AllocDistances(&distances_ptr, 0);
    }

    knn = num_points > knn ? knn : num_points;

    // Allocate output tensors.
    query_neighbors_row_splits.AsRvalue() =
            Tensor::Arange(0, num_queries * knn, knn);
    TIndex* indices_ptr;
    T* distances_ptr;
    const size_t num_indices = knn * num_queries;

    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    // Calculate norms, |p|^2, |q|^2.
    Tensor point_norms = points.Mul(points).Sum({1});
    Tensor query_norms = queries.Mul(queries).Sum({1});

    // Divide queries and points into chunks (rows and cols).
    int tile_rows = 0;
    int tile_cols = 0;
    chooseTileSize(num_queries, num_points, dim, sizeof(T), tile_rows,
                   tile_cols);
    int num_cols = utility::DivUp(num_points, tile_cols);

    // Allocate temporary memory space.
    Tensor temp_distances =
            Tensor::Empty({tile_rows, tile_cols}, dtype, device);
    Tensor buf_distances =
            Tensor::Empty({tile_rows, num_cols * knn}, dtype, device);
    Tensor buf_indices =
            Tensor::Empty({tile_rows, num_cols * knn}, Dtype::Int32, device);

    // Iterate row-wise.
    for (int i = 0; i < num_queries; i += tile_rows) {
        int num_queries_i = std::min(tile_rows, num_queries - i);
        Tensor queries_i = queries.Slice(0, i, i + num_queries_i);
        Tensor query_norms_i = query_norms.Slice(0, i, i + num_queries_i);
        Tensor buf_distances_row_view =
                buf_distances.Slice(0, 0, num_queries_i);
        Tensor buf_indices_row_view = buf_indices.Slice(0, 0, num_queries_i);
        {
            CUDAScopedStream scoped_stream(CUDAScopedStream::CreateNewStream);
            cudaStream_t cur_stream = cuda::GetStream();
            for (int j = 0; j < num_points; j += tile_cols) {
                int num_points_j = std::min(tile_cols, num_points - j);
                int col_j = j / tile_cols;
                Tensor points_j = points.Slice(0, j, j + num_points_j);
                Tensor point_norms_j =
                        point_norms.Slice(0, j, j + num_points_j);
                Tensor temp_distances_view =
                        temp_distances.Slice(0, 0, num_queries_i)
                                .Slice(1, 0, num_points_j);
                Tensor buf_distances_col_view = buf_distances_row_view.Slice(
                        1, knn * col_j, (col_j + 1) * knn);
                Tensor buf_indices_col_view = buf_indices_row_view.Slice(
                        1, knn * col_j, (col_j + 1) * knn);

                // Calculate -2*p*q
                AddMM(queries_i, points_j.T(), temp_distances_view, -2.0, 0.0);
                // Topk selection & Add |p|^2, |q|^2 with fused kernel
                if (tile_cols == num_points) {
                    Tensor out_indices_view =
                            output_allocator.NeighborsIndex_()
                                    .View({num_queries, knn})
                                    .Slice(0, i, i + num_queries_i);
                    Tensor out_distances_view =
                            output_allocator.NeighborsDistance_()
                                    .View({num_queries, knn})
                                    .Slice(0, i, i + num_queries_i);
                    runL2SelectMin<T>(cur_stream, temp_distances_view,
                                      point_norms_j, out_distances_view,
                                      out_indices_view, knn, num_cols,
                                      tile_cols);
                    out_distances_view.Add_(
                            query_norms_i.View({num_queries_i, 1}));
                } else {
                    runL2SelectMin<T>(cur_stream, temp_distances_view,
                                      point_norms_j, buf_distances_col_view,
                                      buf_indices_col_view, knn, num_cols,
                                      tile_cols);
                    buf_distances_col_view.Add_(
                            query_norms_i.View({num_queries_i, 1}));
                }
            }
            // Write results to output tensor.
            if (tile_cols != num_points) {
                runIncrementIndex<TIndex>(cur_stream, buf_indices_row_view, knn,
                                          tile_cols);
                runBlockSelectPair(
                        cur_stream, buf_distances_row_view.GetDataPtr<T>(),
                        buf_indices_row_view.GetDataPtr<TIndex>(),
                        distances_ptr + knn * i, indices_ptr + knn * i, false,
                        knn, buf_distances_row_view.GetShape(1),
                        buf_distances_row_view.GetShape(0));
            }
        }
    }
}

template <class T, class TIndex>
void KnnSearchCUDA(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_row_splits,
                   Tensor& neighbors_distance) {
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    Device device = points.GetDevice();
    bool brute_force = points.GetShape(1) < 8 && knn <= 32;

    const int batch_size = points_row_splits.GetShape(0) - 1;

    std::vector<NeighborSearchAllocator<T, TIndex>> batch_output_allocators(
            batch_size, NeighborSearchAllocator<T, TIndex>(device));

    int64_t last_neighbors_count = 0;
    for (int i = 0; i < batch_size; ++i) {
        const Tensor points_i =
                points.Slice(0, points_row_splits[i].Item<int64_t>(),
                             points_row_splits[i + 1].Item<int64_t>());
        const Tensor queries_i =
                queries.Slice(0, queries_row_splits[i].Item<int64_t>(),
                              queries_row_splits[i + 1].Item<int64_t>());
        int64_t num_queries_i = queries_i.GetShape(0);
        Tensor neighbors_row_splits_i = neighbors_row_splits.Slice(
                0, queries_row_splits[i].Item<int64_t>(),
                queries_row_splits[i + 1].Item<int64_t>());
        int64_t* neighbors_row_splits_i_ptr =
                neighbors_row_splits_i.GetDataPtr<int64_t>();

        if (brute_force) {
            KnnSearchCUDABruteForce<T, TIndex>(points_i, queries_i, knn,
                                               batch_output_allocators[i],
                                               neighbors_row_splits_i);
        } else {
            KnnSearchCUDAOptimized<T, TIndex>(points_i, queries_i, knn,
                                              batch_output_allocators[i],
                                              neighbors_row_splits_i);
        }

        if (i > 0) {
            for (int j = 0; j <= num_queries_i; ++j) {
                neighbors_row_splits_i_ptr[j] += last_neighbors_count;
            }
        }
        last_neighbors_count = neighbors_row_splits_i_ptr[num_queries_i];
    }

    if (batch_size == 1) {
        neighbors_index = batch_output_allocators[0].NeighborsIndex().View(
                {num_queries, -1});
        neighbors_distance =
                batch_output_allocators[0].NeighborsDistance().View(
                        {num_queries, -1});
        return;
    }

    // combine results
    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    int64_t neighbors_size = 0;
    for (const auto& a : batch_output_allocators) {
        neighbors_size += a.NeighborsIndex().GetShape(0);
    }
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, neighbors_size);
    output_allocator.AllocDistances(&neighbors_distance_ptr, neighbors_size);

    last_neighbors_count = 0;
    for (int i = 0; i < batch_size; ++i) {
        auto& a = batch_output_allocators[i];
        int64_t offset = points_row_splits[i].Item<int64_t>();
        int64_t num_neighbors_i = a.NeighborsIndex().GetShape(0);
        if (num_neighbors_i) {
            Tensor NeighborIndexAccumulated = a.NeighborsIndex().Add(offset);
            MemoryManager::Memcpy(neighbors_index_ptr + last_neighbors_count,
                                  device, a.IndicesPtr(), device,
                                  sizeof(TIndex) * num_neighbors_i);
            MemoryManager::Memcpy(neighbors_distance_ptr + last_neighbors_count,
                                  device, a.DistancesPtr(), device,
                                  sizeof(T) * num_neighbors_i);
            last_neighbors_count += num_neighbors_i;
        }
    }
    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                \
    template void KnnSearchCUDA<T, TIndex>(                                   \
            const Tensor& points, const Tensor& points_row_splits,            \
            const Tensor& queries, const Tensor& queries_row_splits, int knn, \
            Tensor& neighbors_index, Tensor& neighbors_row_splits,            \
            Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)

}  // namespace nns
}  // namespace core
}  // namespace open3d
