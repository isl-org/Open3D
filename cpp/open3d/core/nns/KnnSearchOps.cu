// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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

// CUDA kernel to mark selected points for masking in the next pass.
// Stores 1 for selected indices, 0 otherwise.
template <typename TIndex>
__global__ void MarkSelectedIndices(
        uint8_t* mask,                   // Shape: (num_queries, num_points)
        const TIndex* selected_indices,  // Shape: (num_queries_i, chunk_k)
        int num_queries_i,               // Number of queries in this batch
        int num_points,                  // Total number of points
        int chunk_k,                     // Number of neighbors selected
        int query_offset) {              // Starting query index
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries_i) return;

    int global_query_idx = query_offset + query_idx;
    const TIndex* selected_row = selected_indices + query_idx * chunk_k;
    uint8_t* mask_row = mask + global_query_idx * num_points;

    for (int k = 0; k < chunk_k; ++k) {
        TIndex idx = selected_row[k];
        if (idx >= 0 && idx < num_points) {
            mask_row[idx] = 1;
        }
    }
}

// CUDA kernel to apply mask by setting distances to infinity for masked
// points.
template <typename T>
__global__ void ApplyMaskToDistances(
        T* distances,         // Shape: (num_queries_tile, distance_row_stride)
        const uint8_t* mask,  // Shape: (num_queries, num_points)
        int num_queries_tile,
        int64_t distance_row_stride,  // Stride between rows
        int num_points_tile,  // Actual number of valid points in this tile
        int query_offset,     // Starting query index in the mask
        int point_offset,     // Starting point index in the mask
        int num_points) {     // Total number of points
    int query_local = blockIdx.y;
    int point_local = blockIdx.x * blockDim.x + threadIdx.x;

    if (query_local >= num_queries_tile || point_local >= num_points_tile)
        return;

    int query_global = query_offset + query_local;
    int point_global = point_offset + point_local;

    if (point_global >= num_points) return;

    if (mask[query_global * num_points + point_global]) {
        distances[query_local * distance_row_stride + point_local] =
                static_cast<T>(std::numeric_limits<float>::max());
    }
}

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
    CUDAScopedDevice scoped_device(points.GetDevice());
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
        return;
    }

    knn = num_points > knn ? knn : num_points;

    // Allocate output tensors.
    query_neighbors_row_splits.AsRvalue() =
            Tensor::Arange(0, num_queries * knn, knn);
    TIndex* indices_ptr;
    T* distances_ptr;
    const size_t num_indices = knn * num_queries;

    output_allocator.AllocIndices(&indices_ptr, num_indices, -1);
    output_allocator.AllocDistances(&distances_ptr, num_indices, 0);

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

// Single-pass KNN search (when k <= GPU_MAX_SELECTION_K).
template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void KnnSearchCUDASinglePass(const Tensor& points,
                             const Tensor& queries,
                             int knn,
                             int tile_rows,
                             int tile_cols,
                             OUTPUT_ALLOCATOR& output_allocator,
                             const Tensor& point_norms,
                             const Tensor& query_norms) {
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    Device device = points.GetDevice();
    Dtype dtype = points.GetDtype();
    Dtype index_dtype = Dtype::FromType<TIndex>();
    int num_cols = utility::DivUp(num_points, tile_cols);

    // Get pointers from allocator for use in runBlockSelectPair
    TIndex* indices_ptr = static_cast<TIndex*>(
            output_allocator.NeighborsIndex_().GetDataPtr());
    T* distances_ptr =
            static_cast<T*>(output_allocator.NeighborsDistance_().GetDataPtr());

    // Allocate temporary memory space.
    Tensor temp_distances =
            Tensor::Empty({tile_rows, tile_cols}, dtype, device);
    Tensor buf_distances =
            Tensor::Empty({tile_rows, num_cols * knn}, dtype, device);
    Tensor buf_indices =
            Tensor::Empty({tile_rows, num_cols * knn}, index_dtype, device);

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
                    runL2SelectMin<T, TIndex>(cur_stream, temp_distances_view,
                                              point_norms_j, out_distances_view,
                                              out_indices_view, knn, num_cols,
                                              tile_cols);
                    out_distances_view.Add_(
                            query_norms_i.View({num_queries_i, 1}));
                } else {
                    runL2SelectMin<T, TIndex>(
                            cur_stream, temp_distances_view, point_norms_j,
                            buf_distances_col_view, buf_indices_col_view, knn,
                            num_cols, tile_cols);
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

// Multi-pass KNN search (when knn > GPU_MAX_SELECTION_K).
// Processes neighbors in chunks, using a bitmask to avoid selecting
// already-found neighbors in subsequent passes.
template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void KnnSearchCUDAMultiPass(const Tensor& points,
                            const Tensor& queries,
                            int knn,
                            int tile_rows,
                            int tile_cols,
                            OUTPUT_ALLOCATOR& output_allocator,
                            const Tensor& point_norms,
                            const Tensor& query_norms) {
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    Device device = points.GetDevice();
    Dtype dtype = points.GetDtype();
    Dtype index_dtype = Dtype::FromType<TIndex>();
    int num_cols = utility::DivUp(num_points, tile_cols);

    // Allocate mask to track already-selected points across passes
    Tensor mask =
            Tensor::Zeros({num_queries, num_points}, Dtype::UInt8, device);

    int chunk_k = std::min(knn, GPU_MAX_SELECTION_K);
    Tensor temp_distances =
            Tensor::Empty({tile_rows, tile_cols}, dtype, device);

    // Allocate buffers for multi-tile case (single-tile uses intermediate
    // buffers per-iteration)
    Tensor buf_distances, buf_indices;
    if (num_cols > 1) {
        buf_distances =
                Tensor::Empty({tile_rows, num_cols * chunk_k}, dtype, device);
        buf_indices = Tensor::Empty({tile_rows, num_cols * chunk_k},
                                    index_dtype, device);
    }

    // Multi-pass loop: process chunk_k neighbors at a time
    int total_found = 0;
    while (total_found < knn) {
        int remaining_k = knn - total_found;
        chunk_k = std::min(remaining_k, GPU_MAX_SELECTION_K);

        // Resize buffers if chunk_k changed (only in last iteration)
        if (num_cols > 1 && chunk_k != buf_distances.GetShape(1) / num_cols) {
            buf_distances = Tensor::Empty({tile_rows, num_cols * chunk_k},
                                          dtype, device);
            buf_indices = Tensor::Empty({tile_rows, num_cols * chunk_k},
                                        index_dtype, device);
        }

        // Iterate row-wise over queries
        for (int i = 0; i < num_queries; i += tile_rows) {
            int num_queries_i = std::min(tile_rows, num_queries - i);
            Tensor queries_i = queries.Slice(0, i, i + num_queries_i);
            Tensor query_norms_i = query_norms.Slice(0, i, i + num_queries_i);

            // Intermediate buffers for single-tile multi-pass case
            Tensor chunk_out_distances, chunk_out_indices;
            Tensor buf_distances_row_view, buf_indices_row_view;

            if (tile_cols == num_points) {
                // Single-tile: allocate intermediate buffers
                chunk_out_distances =
                        Tensor::Empty({num_queries_i, chunk_k}, dtype, device);
                chunk_out_indices = Tensor::Empty({num_queries_i, chunk_k},
                                                  index_dtype, device);
            } else {
                // Multi-tile: use row views of buffer
                buf_distances_row_view =
                        buf_distances.Slice(0, 0, num_queries_i);
                buf_indices_row_view = buf_indices.Slice(0, 0, num_queries_i);
            }

            {
                CUDAScopedStream scoped_stream(
                        CUDAScopedStream::CreateNewStream);
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

                    // Calculate -2*p*q
                    AddMM(queries_i, points_j.T(), temp_distances_view, -2.0,
                          0.0);

                    // Apply mask: set already-selected distances to infinity
                    if (total_found > 0) {
                        int64_t distance_row_stride =
                                temp_distances_view.GetStride(0);
                        int block_size = 256;
                        dim3 block(block_size);
                        dim3 grid(utility::DivUp(num_points_j, block_size),
                                  num_queries_i);
                        ApplyMaskToDistances<T><<<grid, block, 0, cur_stream>>>(
                                temp_distances_view.GetDataPtr<T>(),
                                mask.GetDataPtr<uint8_t>(), num_queries_i,
                                distance_row_stride, num_points_j, i, j,
                                num_points);
                    }

                    // Top-k selection
                    if (tile_cols == num_points) {
                        // Single-tile case: output to intermediate buffers
                        runL2SelectMin<T, TIndex>(
                                cur_stream, temp_distances_view, point_norms_j,
                                chunk_out_distances, chunk_out_indices, chunk_k,
                                1, tile_cols);
                        chunk_out_distances.Add_(
                                query_norms_i.View({num_queries_i, 1}));
                    } else {
                        // Multi-tile case: output to buffer
                        Tensor buf_distances_col_view =
                                buf_distances_row_view.Slice(
                                        1, chunk_k * col_j,
                                        (col_j + 1) * chunk_k);
                        Tensor buf_indices_col_view =
                                buf_indices_row_view.Slice(
                                        1, chunk_k * col_j,
                                        (col_j + 1) * chunk_k);
                        runL2SelectMin<T, TIndex>(
                                cur_stream, temp_distances_view, point_norms_j,
                                buf_distances_col_view, buf_indices_col_view,
                                chunk_k, num_cols, tile_cols);
                        buf_distances_col_view.Add_(
                                query_norms_i.View({num_queries_i, 1}));
                    }
                }

                // Write results and update mask
                if (tile_cols != num_points) {
                    // Multi-tile case
                    runIncrementIndex<TIndex>(cur_stream, buf_indices_row_view,
                                              chunk_k, tile_cols);

                    Tensor chunk_out_dist_multi = Tensor::Empty(
                            {num_queries_i, chunk_k}, dtype, device);
                    Tensor chunk_out_idx_multi = Tensor::Empty(
                            {num_queries_i, chunk_k}, index_dtype, device);

                    runBlockSelectPair(
                            cur_stream, buf_distances_row_view.GetDataPtr<T>(),
                            buf_indices_row_view.GetDataPtr<TIndex>(),
                            chunk_out_dist_multi.GetDataPtr<T>(),
                            chunk_out_idx_multi.GetDataPtr<TIndex>(), false,
                            chunk_k, buf_distances_row_view.GetShape(1),
                            num_queries_i);

                    // Copy to final output
                    TIndex* indices_ptr =
                            static_cast<TIndex*>(
                                    output_allocator.NeighborsIndex_()
                                            .GetDataPtr()) +
                            (i * knn + total_found);
                    T* distances_ptr =
                            static_cast<T*>(
                                    output_allocator.NeighborsDistance_()
                                            .GetDataPtr()) +
                            (i * knn + total_found);

                    for (int q = 0; q < num_queries_i; ++q) {
                        MemoryManager::Memcpy(
                                distances_ptr + q * knn, device,
                                chunk_out_dist_multi.GetDataPtr<T>() +
                                        q * chunk_k,
                                device, sizeof(T) * chunk_k);
                        MemoryManager::Memcpy(
                                indices_ptr + q * knn, device,
                                chunk_out_idx_multi.GetDataPtr<TIndex>() +
                                        q * chunk_k,
                                device, sizeof(TIndex) * chunk_k);
                    }

                    // Update mask for next pass
                    {
                        int block_size = 256;
                        dim3 block(block_size);
                        dim3 grid(utility::DivUp(num_queries_i, block_size));
                        MarkSelectedIndices<TIndex>
                                <<<grid, block, 0, cur_stream>>>(
                                        mask.GetDataPtr<uint8_t>(),
                                        chunk_out_idx_multi
                                                .GetDataPtr<TIndex>(),
                                        num_queries_i, num_points, chunk_k, i);
                    }

                } else {
                    // Single-tile case: copy from intermediate buffers to
                    // non-contiguous output slice
                    Tensor out_indices_full =
                            output_allocator.NeighborsIndex_().View(
                                    {num_queries, knn});
                    Tensor out_distances_full =
                            output_allocator.NeighborsDistance_().View(
                                    {num_queries, knn});

                    for (int q = 0; q < num_queries_i; ++q) {
                        int global_query_idx = i + q;

                        Tensor src_dist = chunk_out_distances.Slice(0, q, q + 1)
                                                  .Flatten();
                        Tensor src_idx =
                                chunk_out_indices.Slice(0, q, q + 1).Flatten();

                        Tensor dst_dist = out_distances_full
                                                  .Slice(0, global_query_idx,
                                                         global_query_idx + 1)
                                                  .Slice(1, total_found,
                                                         total_found + chunk_k)
                                                  .Flatten();
                        dst_dist.AsRvalue() = src_dist;

                        Tensor dst_idx = out_indices_full
                                                 .Slice(0, global_query_idx,
                                                        global_query_idx + 1)
                                                 .Slice(1, total_found,
                                                        total_found + chunk_k)
                                                 .Flatten();
                        dst_idx.AsRvalue() = src_idx;
                    }

                    // Update mask for next pass
                    {
                        int block_size = 256;
                        dim3 block(block_size);
                        dim3 grid(utility::DivUp(num_queries_i, block_size));
                        MarkSelectedIndices<TIndex>
                                <<<grid, block, 0, cur_stream>>>(
                                        mask.GetDataPtr<uint8_t>(),
                                        chunk_out_indices.GetDataPtr<TIndex>(),
                                        num_queries_i, num_points, chunk_k, i);
                    }
                }
            }
        }

        total_found += chunk_k;
    }
}

template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void KnnSearchCUDAOptimized(const Tensor& points,
                            const Tensor& queries,
                            int knn,
                            OUTPUT_ALLOCATOR& output_allocator,
                            Tensor& query_neighbors_row_splits) {
    CUDAScopedDevice scoped_device(points.GetDevice());
    int num_points = points.GetShape(0);
    int num_queries = queries.GetShape(0);
    int dim = points.GetShape(1);

    // Return if input points are empty.
    if (num_points == 0 || num_queries == 0) {
        query_neighbors_row_splits.Fill(0);
        TIndex* indices_ptr;
        T* distances_ptr;

        output_allocator.AllocIndices(&indices_ptr, 0);
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    knn = num_points > knn ? knn : num_points;

    // Allocate output tensors.
    query_neighbors_row_splits.AsRvalue() =
            Tensor::Arange(0, num_queries * knn, knn);
    TIndex* indices_ptr;
    T* distances_ptr;
    const size_t num_indices = knn * num_queries;

    output_allocator.AllocIndices(&indices_ptr, num_indices, -1);
    output_allocator.AllocDistances(&distances_ptr, num_indices, 0);

    // Calculate norms, |p|^2, |q|^2.
    Tensor point_norms = points.Mul(points).Sum({1});
    Tensor query_norms = queries.Mul(queries).Sum({1});

    // Divide queries and points into chunks (rows and cols).
    int tile_rows = 0;
    int tile_cols = 0;
    chooseTileSize(num_queries, num_points, dim, sizeof(T), tile_rows,
                   tile_cols);

    // Dispatch to appropriate algorithm based on knn
    if (knn <= GPU_MAX_SELECTION_K) {
        KnnSearchCUDASinglePass<T, TIndex>(points, queries, knn, tile_rows,
                                           tile_cols, output_allocator,
                                           point_norms, query_norms);
    } else {
        KnnSearchCUDAMultiPass<T, TIndex>(points, queries, knn, tile_rows,
                                          tile_cols, output_allocator,
                                          point_norms, query_norms);
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
    CUDAScopedDevice scoped_device(points.GetDevice());
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
