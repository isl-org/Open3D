// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL NNS driver: tiled AddMM distance evaluation with fused top-K selection.
//
// Three search variants are implemented here:
//   KnnSearchSYCL       – brute-force K nearest neighbors
//   FixedRadiusSearchSYCL – all neighbors within radius (count + gather)
//   HybridSearchSYCL    – neighbors within radius, keep best max_knn
//
// Common tiling strategy
// ──────────────────────
// All three variants loop over (query-tile, point-tile) pairs.  For each pair
// AddMM computes the −2*q*p tile once (O(n·m·d) work, highly optimised by
// oneMKL).  The remaining passes read the tile (O(n·m)) at most twice, keeping
// computation well below the GEMM cost.
//
// KNN small-k path (k ≤ kSYCLKnnMidKMax)
// ────────────────────────────────────────
// UpdateTopKFromTileSYCL<K> fuses: Add_(point_norms) + SelectTopK + Merge
// into a single per-query scan over the tile, maintaining a running max-heap
// in global memory (fits in GRF for k ≤ kSYCLKnnSmallKMax).  No intermediate
// tile_top or merged tensors are needed.  FinalizeTopKSYCL adds |q|² once (P2)
// and heap-sorts into ascending order.
//
// KNN large-k path (k > kSYCLKnnMidKMax)
// ────────────────────────────────────────
// Falls back to the legacy SelectTopKQueriesSYCL + MergeTopKQueriesSYCL pair.
// P2 applies: |q|² is NOT added to the tile; AddQueryNormsToDistancesSYCL adds
// it once at the end.  P8 note: for k > kSYCLKnnMidKMax the SelectTopK path
// launches one oneDPL partial_sort per query sequentially.
//
// Radius / Hybrid
// ───────────────
// P2 applies: CountWithinThresholdQueriesSYCL and
// GatherWithinThresholdQueriesSYCL take the per-query adjusted threshold
// (radius² − |q|²), avoiding the Add_(query_norms) tile pass.  Returned
// distances are the full L2 (partial + |q|²), clamped ≥ 0 (C1).

#include <algorithm>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/kernel/KnnSearchSYCLImpl.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

// Batched KNN search.
//
// For k ≤ kSYCLKnnMidKMax: one fused kernel per (query-tile, point-tile) pair
//   replaces three separate passes (Add_, SelectTopK, Merge).  The running
//   max-heap is maintained in global memory between tile iterations and
//   finalized (sorted + |q|² added) once per query batch.
//
// For k > kSYCLKnnMidKMax: legacy Select + Merge path with P2 fix.
//
// C5 fix: batch_knn = min(knn, num_points_i) per batch.
template <class T, class TIndex>
void KnnSearchSYCL(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_row_splits,
                   Tensor& neighbors_distance,
                   int64_t tile_bytes) {
    const Device device = points.GetDevice();
    const Dtype dtype = points.GetDtype();
    const Dtype index_dtype = Dtype::FromType<TIndex>();
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    const int batch_size = points_row_splits.GetShape(0) - 1;
    std::vector<NeighborSearchAllocator<T, TIndex>> batch_output_allocators(
            batch_size, NeighborSearchAllocator<T, TIndex>(device));

    int64_t* neighbors_row_splits_ptr =
            neighbors_row_splits.GetDataPtr<int64_t>();
    int64_t last_neighbors_count = 0;
    int64_t batch_knn = 0;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();

        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);

        // C5: clamp knn to the number of available points.
        batch_knn = std::min<int64_t>(knn, num_points_i);

        // Populate row_splits for this batch.
        neighbors_row_splits_ptr[query_begin] = last_neighbors_count;
        for (int64_t q = 0; q < num_queries_i; ++q) {
            neighbors_row_splits_ptr[query_begin + q + 1] =
                    last_neighbors_count + (q + 1) * batch_knn;
        }
        last_neighbors_count += num_queries_i * batch_knn;

        TIndex* indices_ptr;
        T* distances_ptr;
        batch_output_allocators[batch_idx].AllocIndices(
                &indices_ptr, num_queries_i * batch_knn, TIndex(-1));
        batch_output_allocators[batch_idx].AllocDistances(
                &distances_ptr, num_queries_i * batch_knn, T(0));

        Tensor out_indices =
                batch_output_allocators[batch_idx].NeighborsIndex().View(
                        {num_queries_i, batch_knn});
        Tensor out_distances =
                batch_output_allocators[batch_idx].NeighborsDistance().View(
                        {num_queries_i, batch_knn});

        // |p|² and |q|² for distance tiling.
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        int64_t tile_queries = 0, tile_points = 0;
        ChooseTileSizeSYCL(num_queries_i, num_points_i, sizeof(T), tile_bytes,
                           tile_queries, tile_points);

        // Shared tile buffer for AddMM output (−2*q*p).
        Tensor temp_distances =
                Tensor::Empty({tile_queries, tile_points}, dtype, device);

        if (batch_knn <= kSYCLKnnMidKMax) {
            // ── Fused small/mid-k path ─────────────────────────────────────
            // K-bucket ≥ batch_knn; running best allocated at K-bucket width.
            const int64_t k_bucket = KBucket(batch_knn);

            // Running best (max-heap, not sorted until Finalize).
            Tensor running_best_dist =
                    Tensor::Full({tile_queries, k_bucket},
                                 std::numeric_limits<T>::max(), dtype, device);
            Tensor running_best_idx = Tensor::Full(
                    {tile_queries, k_bucket}, TIndex(-1), index_dtype, device);

            for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
                const int64_t num_queries_iter =
                        std::min(tile_queries, num_queries_i - q);
                Tensor queries_tile =
                        queries_i.Slice(0, q, q + num_queries_iter);
                Tensor query_norms_tile =
                        query_norms.Slice(0, q, q + num_queries_iter);

                // Reset running best for this query tile.
                Tensor rb_dist =
                        running_best_dist.Slice(0, 0, num_queries_iter);
                Tensor rb_idx = running_best_idx.Slice(0, 0, num_queries_iter);
                rb_dist.Fill(std::numeric_limits<T>::max());
                rb_idx.Fill(TIndex(-1));

                for (int64_t p = 0; p < num_points_i; p += tile_points) {
                    const int64_t num_points_iter =
                            std::min(tile_points, num_points_i - p);
                    Tensor points_tile =
                            points_i.Slice(0, p, p + num_points_iter);
                    Tensor point_norms_tile =
                            point_norms.Slice(0, p, p + num_points_iter);
                    Tensor temp_view =
                            temp_distances.Slice(0, 0, num_queries_iter)
                                    .Slice(1, 0, num_points_iter);

                    // Step 1: AddMM → temp_view = −2*q*p
                    AddMM(queries_tile, points_tile.T(), temp_view, -2.0, 0.0);

                    // Step 2: Fused |p|² add + top-K update (P2, C1, C4)
                    DispatchUpdateTopKFromTile<T, TIndex>(
                            queue, temp_view.GetDataPtr<T>(),
                            temp_view.GetStride(0),
                            point_norms_tile.GetDataPtr<T>(), num_queries_iter,
                            num_points_iter, k_bucket, TIndex(p),
                            rb_dist.GetDataPtr<T>(),
                            rb_idx.GetDataPtr<TIndex>(),
                            /*use_threshold=*/false, T(0));
                }

                // Step 3: Heap-sort + add |q|² → write to final output.
                DispatchFinalizeTopK<T, TIndex>(
                        queue, num_queries_iter, rb_dist.GetDataPtr<T>(),
                        rb_idx.GetDataPtr<TIndex>(),
                        out_distances.Slice(0, q, q + num_queries_iter)
                                .GetDataPtr<T>(),
                        out_indices.Slice(0, q, q + num_queries_iter)
                                .GetDataPtr<TIndex>(),
                        batch_knn, k_bucket, query_norms_tile.GetDataPtr<T>());
            }
        } else {
            // ── Large-k path (k > kSYCLKnnMidKMax): legacy Select + Merge ──
            // P2: |q|² not in tile; AddQueryNormsToDistancesSYCL adds it once.
            Tensor tile_sort_indices = Tensor::Empty(
                    {tile_queries, tile_points}, index_dtype, device);
            Tensor tile_top_indices = Tensor::Empty({tile_queries, batch_knn},
                                                    index_dtype, device);
            Tensor tile_top_distances =
                    Tensor::Empty({tile_queries, batch_knn}, dtype, device);
            Tensor best_indices = Tensor::Empty({tile_queries, batch_knn},
                                                index_dtype, device);
            Tensor best_distances =
                    Tensor::Empty({tile_queries, batch_knn}, dtype, device);
            Tensor merged_indices = Tensor::Empty({tile_queries, batch_knn},
                                                  index_dtype, device);
            Tensor merged_distances =
                    Tensor::Empty({tile_queries, batch_knn}, dtype, device);
            Tensor merge_scratch = Tensor::Empty({tile_queries, 2 * batch_knn},
                                                 index_dtype, device);

            for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
                const int64_t num_queries_iter =
                        std::min(tile_queries, num_queries_i - q);
                Tensor queries_tile =
                        queries_i.Slice(0, q, q + num_queries_iter);
                Tensor query_norms_tile =
                        query_norms.Slice(0, q, q + num_queries_iter);

                Tensor biv = best_indices.Slice(0, 0, num_queries_iter);
                Tensor bdv = best_distances.Slice(0, 0, num_queries_iter);
                biv.Fill(TIndex(-1));
                bdv.Fill(std::numeric_limits<T>::max());

                for (int64_t p = 0; p < num_points_i; p += tile_points) {
                    const int64_t num_points_iter =
                            std::min(tile_points, num_points_i - p);
                    Tensor points_tile =
                            points_i.Slice(0, p, p + num_points_iter);
                    Tensor point_norms_tile =
                            point_norms.Slice(0, p, p + num_points_iter);
                    Tensor temp_view =
                            temp_distances.Slice(0, 0, num_queries_iter)
                                    .Slice(1, 0, num_points_iter);

                    // AddMM → partial dist (−2qp)
                    AddMM(queries_tile, points_tile.T(), temp_view, -2.0, 0.0);
                    // P2: add |p|² only (skip |q|²)
                    temp_view.Add_(point_norms_tile.View({1, num_points_iter}));

                    Tensor ttiv =
                            tile_top_indices.Slice(0, 0, num_queries_iter);
                    Tensor ttdv =
                            tile_top_distances.Slice(0, 0, num_queries_iter);
                    Tensor miv = merged_indices.Slice(0, 0, num_queries_iter);
                    Tensor mdv = merged_distances.Slice(0, 0, num_queries_iter);

                    SelectTopKQueriesSYCL<T, TIndex>(
                            device, temp_view.GetDataPtr<T>(),
                            temp_view.GetStride(0), num_queries_iter,
                            num_points_iter, batch_knn, TIndex(p),
                            tile_sort_indices.GetDataPtr<TIndex>(),
                            tile_sort_indices.GetStride(0),
                            ttiv.GetDataPtr<TIndex>(), ttdv.GetDataPtr<T>(),
                            batch_knn);
                    MergeTopKQueriesSYCL<T, TIndex>(
                            device, bdv.GetDataPtr<T>(),
                            biv.GetDataPtr<TIndex>(), batch_knn,
                            ttdv.GetDataPtr<T>(), ttiv.GetDataPtr<TIndex>(),
                            batch_knn, num_queries_iter, batch_knn,
                            merge_scratch.GetDataPtr<TIndex>(),
                            merge_scratch.GetStride(0),
                            miv.GetDataPtr<TIndex>(), mdv.GetDataPtr<T>(),
                            batch_knn);

                    biv.AsRvalue() = miv;
                    bdv.AsRvalue() = mdv;
                }

                // Write partial results to output; |q|² added after all tiles.
                out_indices.Slice(0, q, q + num_queries_iter).AsRvalue() = biv;
                out_distances.Slice(0, q, q + num_queries_iter).AsRvalue() =
                        bdv;
            }

            // P2: add |q|² once per query to the final distances.
            AddQueryNormsToDistancesSYCL<T, TIndex>(
                    device, num_queries_i, batch_knn, indices_ptr,
                    distances_ptr, query_norms.GetDataPtr<T>());
        }

        queue.wait_and_throw();
    }

    // Assemble the final output tensors from per-batch allocators.
    if (batch_size == 1) {
        neighbors_index = batch_output_allocators[0].NeighborsIndex().View(
                {queries.GetShape(0), batch_knn});
        neighbors_distance =
                batch_output_allocators[0].NeighborsDistance().View(
                        {queries.GetShape(0), batch_knn});
        return;
    }

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    int64_t neighbors_size = 0;
    for (const auto& alloc : batch_output_allocators)
        neighbors_size += alloc.NeighborsIndex().GetShape(0);

    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, neighbors_size);
    output_allocator.AllocDistances(&neighbors_distance_ptr, neighbors_size);

    int64_t offset = 0;
    for (const auto& alloc : batch_output_allocators) {
        const int64_t sz = alloc.NeighborsIndex().GetShape(0);
        if (sz == 0) continue;
        MemoryManager::Memcpy(neighbors_index_ptr + offset, device,
                              alloc.IndicesPtr(), device, sizeof(TIndex) * sz);
        MemoryManager::Memcpy(neighbors_distance_ptr + offset, device,
                              alloc.DistancesPtr(), device, sizeof(T) * sz);
        offset += sz;
    }
    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

// Fixed-radius search.
//
// Two tiled passes: count (to size the output) then gather.
// P2: the query-norm Add_ pass is dropped.  CountWithinThreshold and
//     GatherWithinThreshold compute per-query adjusted thresholds internally
//     (threshold_q = radius² − |q|²) and add |q|² back to returned distances.
template <class T, class TIndex>
void FixedRadiusSearchSYCL(const Tensor& points,
                           const Tensor& queries,
                           double radius,
                           const Tensor& points_row_splits,
                           const Tensor& queries_row_splits,
                           const Tensor& /*hash_table_splits*/,
                           const Tensor& /*hash_table_index*/,
                           const Tensor& /*hash_table_cell_splits*/,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           const bool sort,
                           Tensor& neighbors_index,
                           Tensor& neighbors_row_splits,
                           Tensor& neighbors_distance,
                           int64_t tile_bytes) {
    if (metric != Metric::L2) {
        utility::LogError("SYCL fixed radius search only supports L2 metric.");
    }
    if (ignore_query_point) {
        utility::LogError(
                "SYCL fixed radius search does not support "
                "ignore_query_point.");
    }

    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const Dtype dtype = points.GetDtype();
    const T radius_sq = static_cast<T>(radius * radius);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    Tensor counts = Tensor::Zeros({num_queries}, Int64, device);

    int64_t tile_queries = 0, tile_points = 0;
    ChooseTileSizeSYCL(num_queries, points.GetShape(0), sizeof(T), tile_bytes,
                       tile_queries, tile_points);
    Tensor temp_distances =
            Tensor::Empty({tile_queries, tile_points}, dtype, device);

    const int num_batches = static_cast<int>(points_row_splits.GetShape(0)) - 1;

    // ── Pass 1: count ────────────────────────────────────────────────────────
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            int64_t* counts_ptr =
                    counts.GetDataPtr<int64_t>() + query_begin + q;

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_view = temp_distances.Slice(0, 0, num_queries_iter)
                                           .Slice(1, 0, num_points_iter);

                // Partial dist = −2qp + |p|² (no |q|²).
                AddMM(queries_tile, points_tile.T(), temp_view, -2.0, 0.0);
                temp_view.Add_(point_norms_tile.View({1, num_points_iter}));

                // P2: CountWithinThreshold uses per-query adjusted threshold.
                CountWithinThresholdQueriesSYCL<T>(
                        device, temp_view.GetDataPtr<T>(),
                        temp_view.GetStride(0), num_queries_iter,
                        num_points_iter, query_norms_tile.GetDataPtr<T>(),
                        radius_sq, counts_ptr);
            }
        }
    }

    queue.wait_and_throw();

    // Build row_splits from counts.
    Tensor counts_cpu = counts.To(Device("CPU:0"));
    const int64_t* counts_cpu_ptr = counts_cpu.GetDataPtr<int64_t>();
    std::vector<int64_t> row_splits(num_queries + 1, 0);
    for (int64_t q = 0; q < num_queries; ++q)
        row_splits[q + 1] = row_splits[q] + counts_cpu_ptr[q];
    neighbors_row_splits =
            Tensor(row_splits, {num_queries + 1}, Int64).To(device);

    std::vector<int64_t> row_offsets(row_splits.begin(), row_splits.end() - 1);
    Tensor write_offsets = Tensor(row_offsets, {num_queries}, Int64).To(device);

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, row_splits.back());
    if (return_distances || sort) {
        output_allocator.AllocDistances(&neighbors_distance_ptr,
                                        row_splits.back());
    } else {
        output_allocator.AllocDistances(&neighbors_distance_ptr, 0);
    }

    // ── Pass 2: gather ───────────────────────────────────────────────────────
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            int64_t* offsets_ptr =
                    write_offsets.GetDataPtr<int64_t>() + query_begin + q;

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_view = temp_distances.Slice(0, 0, num_queries_iter)
                                           .Slice(1, 0, num_points_iter);

                AddMM(queries_tile, points_tile.T(), temp_view, -2.0, 0.0);
                temp_view.Add_(point_norms_tile.View({1, num_points_iter}));

                GatherWithinThresholdQueriesSYCL<T, TIndex>(
                        device, temp_view.GetDataPtr<T>(),
                        temp_view.GetStride(0), num_queries_iter,
                        num_points_iter, query_norms_tile.GetDataPtr<T>(),
                        radius_sq, TIndex(point_begin + p), offsets_ptr,
                        neighbors_index_ptr,
                        (return_distances || sort) ? neighbors_distance_ptr
                                                   : nullptr);
            }
        }
    }

    queue.wait_and_throw();

    if (sort && row_splits.back() > 0) {
        // Per-query insertion sort on distances (already full L2 from Gather).
        const int64_t* rs_ptr = neighbors_row_splits.GetDataPtr<int64_t>();
        queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
            const int64_t q = id[0];
            const int64_t start = rs_ptr[q], end = rs_ptr[q + 1];
            const int64_t len = end - start;
            if (len <= 1) return;
            TIndex* seg_i = neighbors_index_ptr + start;
            T* seg_d = neighbors_distance_ptr + start;
            for (int64_t i = 1; i < len; ++i) {
                T kd = seg_d[i];
                TIndex ki = seg_i[i];
                int64_t j = i - 1;
                while (j >= 0 &&
                       (seg_d[j] > kd || (seg_d[j] == kd && seg_i[j] > ki))) {
                    seg_d[j + 1] = seg_d[j];
                    seg_i[j + 1] = seg_i[j];
                    --j;
                }
                seg_d[j + 1] = kd;
                seg_i[j + 1] = ki;
            }
        });
        queue.wait_and_throw();
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    if (!return_distances)
        neighbors_distance = Tensor({0}, Dtype::FromType<T>(), device);
}

// Hybrid search: count all neighbors within radius while keeping only the
// best max_knn per query in fixed-size output tensors.
//
// P2: |q|² is deferred to AddQueryNormsToHybridDistancesSYCL.
// SelectTopKQueriesSYCL receives per-query adjusted thresholds.
template <class T, class TIndex>
void HybridSearchSYCL(const Tensor& points,
                      const Tensor& queries,
                      double radius,
                      int max_knn,
                      const Tensor& points_row_splits,
                      const Tensor& queries_row_splits,
                      const Tensor& /*hash_table_splits*/,
                      const Tensor& /*hash_table_index*/,
                      const Tensor& /*hash_table_cell_splits*/,
                      const Metric metric,
                      Tensor& neighbors_index,
                      Tensor& neighbors_count,
                      Tensor& neighbors_distance,
                      int64_t tile_bytes) {
    if (metric != Metric::L2) {
        utility::LogError("SYCL hybrid search only supports L2 metric.");
    }

    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const Dtype dtype = points.GetDtype();
    const T radius_sq = static_cast<T>(radius * radius);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, num_queries * max_knn,
                                  TIndex(-1));
    output_allocator.AllocDistances(&neighbors_distance_ptr,
                                    num_queries * max_knn,
                                    std::numeric_limits<T>::max());

    Tensor counts = Tensor::Zeros({num_queries}, Int64, device);
    Tensor out_indices =
            output_allocator.NeighborsIndex().View({num_queries, max_knn});
    Tensor out_distances =
            output_allocator.NeighborsDistance().View({num_queries, max_knn});

    int64_t tile_queries = 0, tile_points = 0;
    ChooseTileSizeSYCL(num_queries, points.GetShape(0), sizeof(T), tile_bytes,
                       tile_queries, tile_points);
    Tensor temp_distances =
            Tensor::Empty({tile_queries, tile_points}, dtype, device);
    const Dtype idx_dtype = Dtype::FromType<TIndex>();
    Tensor tile_sort_indices =
            Tensor::Empty({tile_queries, tile_points}, idx_dtype, device);
    Tensor tile_top_indices =
            Tensor::Empty({tile_queries, max_knn}, idx_dtype, device);
    Tensor tile_top_distances =
            Tensor::Empty({tile_queries, max_knn}, dtype, device);
    Tensor merged_indices =
            Tensor::Empty({tile_queries, max_knn}, idx_dtype, device);
    Tensor merged_distances =
            Tensor::Empty({tile_queries, max_knn}, dtype, device);
    Tensor merge_scratch =
            Tensor::Empty({tile_queries, 2 * max_knn}, idx_dtype, device);

    // Per-query adjusted threshold tensor (device memory, one T per query
    // tile).
    Tensor adj_threshold = Tensor::Empty({tile_queries}, dtype, device);

    const int num_batches = static_cast<int>(points_row_splits.GetShape(0)) - 1;
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            Tensor best_iv = out_indices.Slice(
                    0, query_begin + q, query_begin + q + num_queries_iter);
            Tensor best_dv = out_distances.Slice(
                    0, query_begin + q, query_begin + q + num_queries_iter);
            int64_t* counts_ptr =
                    counts.GetDataPtr<int64_t>() + query_begin + q;

            // Per-query adjusted threshold: radius² − |q|².
            // Compute on device and slice to query tile size.
            {
                const T rsq = radius_sq;
                const T* qn = query_norms_tile.GetDataPtr<T>();
                T* at = adj_threshold.GetDataPtr<T>();
                queue.parallel_for(
                        sycl::range<1>(num_queries_iter),
                        [=](sycl::id<1> id) { at[id[0]] = rsq - qn[id[0]]; });
            }

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_view = temp_distances.Slice(0, 0, num_queries_iter)
                                           .Slice(1, 0, num_points_iter);

                // Partial dist = −2qp + |p|² (no |q|²).
                AddMM(queries_tile, points_tile.T(), temp_view, -2.0, 0.0);
                temp_view.Add_(point_norms_tile.View({1, num_points_iter}));

                // Count: uses per-query adjusted threshold (P2).
                CountWithinThresholdQueriesSYCL<T>(
                        device, temp_view.GetDataPtr<T>(),
                        temp_view.GetStride(0), num_queries_iter,
                        num_points_iter, query_norms_tile.GetDataPtr<T>(),
                        radius_sq, counts_ptr);

                // Select top-max_knn within threshold.
                Tensor ttiv = tile_top_indices.Slice(0, 0, num_queries_iter);
                Tensor ttdv = tile_top_distances.Slice(0, 0, num_queries_iter);
                Tensor miv = merged_indices.Slice(0, 0, num_queries_iter);
                Tensor mdv = merged_distances.Slice(0, 0, num_queries_iter);

                SelectTopKQueriesSYCL<T, TIndex>(
                        device, temp_view.GetDataPtr<T>(),
                        temp_view.GetStride(0), num_queries_iter,
                        num_points_iter, max_knn, TIndex(point_begin + p),
                        tile_sort_indices.GetDataPtr<TIndex>(),
                        tile_sort_indices.GetStride(0),
                        ttiv.GetDataPtr<TIndex>(), ttdv.GetDataPtr<T>(),
                        max_knn,
                        /*use_threshold=*/true, adj_threshold.GetDataPtr<T>(),
                        /*scalar_threshold=*/T(0));
                MergeTopKQueriesSYCL<T, TIndex>(
                        device, best_dv.GetDataPtr<T>(),
                        best_iv.GetDataPtr<TIndex>(), max_knn,
                        ttdv.GetDataPtr<T>(), ttiv.GetDataPtr<TIndex>(),
                        max_knn, num_queries_iter, max_knn,
                        merge_scratch.GetDataPtr<TIndex>(),
                        merge_scratch.GetStride(0), miv.GetDataPtr<TIndex>(),
                        mdv.GetDataPtr<T>(), max_knn);

                best_iv.AsRvalue() = miv;
                best_dv.AsRvalue() = mdv;
            }
        }
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    neighbors_count =
            Tensor::Empty({num_queries}, Dtype::FromType<TIndex>(), device);
    FinalizeHybridResultsSYCL<T, TIndex>(device, counts.GetDataPtr<int64_t>(),
                                         num_queries, max_knn,
                                         neighbors_index.GetDataPtr<TIndex>(),
                                         neighbors_distance.GetDataPtr<T>(),
                                         neighbors_count.GetDataPtr<TIndex>());

    // P2: add |q|² back to the partial distances stored in neighbors_distance.
    AddQueryNormsToHybridDistancesSYCL<T, TIndex>(
            device, num_queries, max_knn, neighbors_index.GetDataPtr<TIndex>(),
            neighbors_distance.GetDataPtr<T>(),
            queries.Mul(queries).Sum({1}).GetDataPtr<T>());

    queue.wait_and_throw();
}

#define INSTANTIATE(T, TIndex)                                                 \
    template void KnnSearchSYCL<T, TIndex>(                                    \
            const Tensor& points, const Tensor& points_row_splits,             \
            const Tensor& queries, const Tensor& queries_row_splits, int knn,  \
            Tensor& neighbors_index, Tensor& neighbors_row_splits,             \
            Tensor& neighbors_distance, int64_t tile_bytes);                   \
    template void FixedRadiusSearchSYCL<T, TIndex>(                            \
            const Tensor& points, const Tensor& queries, double radius,        \
            const Tensor& points_row_splits, const Tensor& queries_row_splits, \
            const Tensor&, const Tensor&, const Tensor&, const Metric metric,  \
            const bool ignore_query_point, const bool return_distances,        \
            const bool sort, Tensor& neighbors_index,                          \
            Tensor& neighbors_row_splits, Tensor& neighbors_distance,          \
            int64_t tile_bytes);                                               \
    template void HybridSearchSYCL<T, TIndex>(                                 \
            const Tensor& points, const Tensor& queries, double radius,        \
            int max_knn, const Tensor& points_row_splits,                      \
            const Tensor& queries_row_splits, const Tensor&, const Tensor&,    \
            const Tensor&, const Metric metric, Tensor& neighbors_index,       \
            Tensor& neighbors_count, Tensor& neighbors_distance,               \
            int64_t tile_bytes);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)

}  // namespace nns
}  // namespace core
}  // namespace open3d
