// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL NNS driver.
//
// Three search variants are implemented here:
//   KnnSearchSYCL       – brute-force K nearest neighbors: tiled AddMM
//                         distance evaluation with fused top-K selection.
//   FixedRadiusSearchSYCL – all neighbors within radius (count + gather),
//                         via the uniform-grid (cell-list) algorithm.
//   HybridSearchSYCL    – neighbors within radius, keep best max_knn, via
//                         the same uniform grid.
//
// There are three SYCL KNN paths inside KnnSearchSYCL, chosen per batch after
// batch_knn = min(knn, num_points).
//
// Decision tree
//
// force_addmm_path == false
//   AND 1 ≤ dim ≤ 8
//   AND batch_knn ≤ 32          →  Direct |p−q|²
// else
//   center data (once)
//   if batch_knn ≤ 512          →  AddMM + fused top-K
//   else                        →  AddMM + Select/Merge (large-k)
//
// | Path           | When                                             |
// Thresholds                                    |
// |----------------|--------------------------------------------------|-----------------------------------------------|
// | Direct         | !force_addmm_path && UseKnnDirect(dim, batch_knn) | dim ∈
// [1,8], k ≤ 32 (kSYCLKnnSmallKMax)      | | AddMM fused    | Not direct,
// batch_knn ≤ 512                      | k ≤ kSYCLKnnMidKMax | | AddMM large-k
// | Not direct, batch_knn > 512                      | k > kSYCLKnnMidKMax | |
// Fixed-radius SYCL | Uniform grid: CountNeighbors -> scan -> WriteNeighbors (+
// optional device sort) (radius-only) | | Hybrid SYCL       | Uniform grid:
// single pass, running top-max_knn + count (radius + capped top-k) |
//
// (Note)  force_addmm_path=true (tuning benchmarks only) always skips Direct.
//
// 1. Direct path (DispatchKnnDirect)
//
// Idea: Brute-force L2 as sum_d (p_d − q_d)² — no AddMM, no centering, no |q|²
// deferral. Algorithm:
// - One sub-group handles one query; many queries share a work-group.
// - Point tiles are staged in SLM (double-buffered).
// - Each lane keeps a private sorted top-K; lanes shuffle-merge within the
// sub-group; lane 0 writes the result.
// - Compile-time NDIM (1…8) and K bucket (1…32); float uses SG=16, double
// prefers SG=8 when supported. Typical use: 3D point clouds, small k (fastest
// path on Xe).
//
// 2. AddMM fused path (k ≤ 512)
//
// Idea: Expand L2 as |q|² − 2q·p + |p|². oneMKL AddMM builds the −2q·p tile;
// selection fuses the rest. Algorithm:
// - Center points/queries (first data row) to reduce float32 cancellation.
// - Precompute |p|², |q|².
// - Loop over (query-tile × point-tile):
//   - AddMM → tile of −2q·p
//   - UpdateTopKFromTile: add |p|², clamp ≥ 0, update a running max-heap per
//   query (K = KBucket(k))
// - FinalizeTopK: heap-sort, add |q|², write first batch_knn neighbors.
// Heap storage: K ≤ 32 → GRF; K ∈ {64…512} → scratch.
//
// 3. AddMM large-k path (k > 512)
//
// Idea: Same AddMM tiling, but Select + Merge instead of a fused running heap.
// Algorithm:
// - Same centering + norms + AddMM tiles.
// - Per point-tile: SelectTopKQueries → tile top-k.
// - Merge into running best via MergeTopKQueries.
// - Once: AddQueryNormsToDistances (|q|² + clamp).
// P8 caveat: for k > 512, merge/select can fall back to per-query serial
// partial_sort (correct, slow).
//
// 4. Fixed-radius (FixedRadiusSearchSYCL) — uniform grid (cell list)
//
// Idea: Find every point within radius; neighbor list length varies per query.
// Ported from FixedRadiusSearchImpl.cuh (CUDA); see
// FixedRadiusSearchSYCLImpl.h. Algorithm:
// - Dataset is bucketed once into a spatial-hash grid with cell size 2*radius
//   (BuildSpatialHashTableSYCL, called from FixedRadiusIndex::SetTensorData):
//   count points per cell -> device inclusive scan (oneDPL) -> scatter into
//   CSR (hash_table_cell_splits, hash_table_index).
// - Pass 1 — count: CountNeighborsSYCL visits each query's 8 corner-adjacent
//   bins and counts points within radius.
// - Device inclusive scan (oneDPL) of counts -> neighbors_row_splits; allocate
//   index (and optional distance) buffers sized by the scanned total.
// - Pass 2 — gather: WriteNeighborsSYCL revisits the same 8 bins and writes
//   matching indices and squared L2 distances.
// - Optional sort: SortNeighborsByDistanceSYCL, a device-only segmented sort
//   (oneDPL sort_by_key) per query segment; ties are not secondarily ordered
//   by index (matches CUDA's cub::DeviceSegmentedRadixSort::SortPairs).
// Typical use: radius neighborhood queries when you need the full set, not a
// fixed-k cap.
//
// 5. Hybrid (HybridSearchSYCL) — uniform grid (cell list)
//
// Idea: Count everyone in radius, but only keep the closest max_knn in
// fixed-size outputs (like KNN capped by radius). Ported from
// FixedRadiusSearchImpl.cuh (CUDA); see FixedRadiusSearchSYCLImpl.h. Algorithm:
// - Reuses the grid built by BuildSpatialHashTableSYCL (shared with
// fixed-radius).
// - Allocate fixed (num_queries, max_knn) index/distance buffers (sentinel
// empty slots).
// - WriteNeighborsHybridSYCL: single pass over the 8 corner-adjacent bins per
//   query, maintaining a running top-max_knn (replace-the-current-max) plus
//   the full in-radius count, then a small per-query bubble sort (bounded by
//   max_knn, so cheap) to return results in ascending-distance order.
// Typical use: features / normals / covariances that want “up to K neighbors
// inside radius.”
//
// Shared conventions (AddMM KNN paths)
// P2: tiles store partial dist −2qp + |p|²; |q|² added at finalize.
// C1: clamp distances ≥ 0.
// C4: equal distance → smaller point index wins.
// C5: batch_knn = min(knn, num_points).
//
// TODO:
// ====
// - Only DirectKNN path is well optimized. AddMM based KNN paths can be slow.
// - Hybrid search could be recast as KNN and then filter (deferred; the grid
//   port already removes the O(N*M) brute force).

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/kernel/FixedRadiusSearchSYCLImpl.h"
#include "open3d/core/nns/kernel/KnnSearchSYCLImpl.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

namespace {

// Translate points and queries by a shared reference point to avoid float32
// catastrophic cancellation in the tiled distance formula |q|^2 - 2q*p + |p|^2.
// Squared L2 distance is translation-invariant, so the shift does not
// change any distance or neighbor index, but it shrinks |p|^2 and |q|^2 from
// O(max_coord^2 * dim) down to O(variance * dim). Without this, high-norm data
// with small inter-point distances (FPFH features, room-scale scan coordinates)
// loses all precision when the large near-equal terms are subtracted, returning
// wrong nearest neighbors. CPU nanoflann computes sum((q-p)^2) directly and is
// unaffected, which is why only the SYCL matmul path needed this.
template <class T>
void CenterPointsAndQueries(Tensor& points, Tensor& queries) {
    if (points.GetShape(0) == 0) return;
    // Use the first point as the shift reference: it is an actual data row (so
    // it always lies in the data range and can never be NaN/uninitialized) and
    // needs no reduction. A centroid (Mean) would shrink norms slightly more,
    // but is not worth a reduction kernel here.
    const Tensor center = points.Slice(0, 0, 1);  // (1, D)
    points = points.Sub(center);
    queries = queries.Sub(center);
}

template <class T>
T FixedRadiusThreshold(Metric metric, T radius) {
    if (metric == L2) {
        return radius * radius;
    }
    return radius;
}

}  // namespace

// Batched KNN search.
///
/// For k ≤ kSYCLKnnMidKMax: one fused kernel per (query-tile, point-tile) pair.
///     The running max-heap is maintained in global memory between tile
///     iterations and finalized (sorted + |q|² added) once per query batch.
///
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
                   int64_t tile_bytes,
                   int64_t max_tile_queries,
                   int64_t tile_points_alignment,
                   bool force_addmm_path) {
    const Device device = points.GetDevice();
    const Dtype dtype = points.GetDtype();
    const Dtype index_dtype = Dtype::FromType<TIndex>();
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    const int batch_size = points_row_splits.GetShape(0) - 1;
    std::vector<NeighborSearchAllocator<T, TIndex>> batch_output_allocators(
            batch_size, NeighborSearchAllocator<T, TIndex>(device));

    // Centering is only needed for the AddMM |q|²−2q·p+|p|² paths. The direct
    // |p−q|² path does not use it — defer until the first AddMM batch.
    Tensor points_c, queries_c;
    bool centered = false;

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

        const Tensor points_raw = points.Slice(0, point_begin, point_end);
        const Tensor queries_raw = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_raw.GetShape(0);
        const int64_t num_queries_i = queries_raw.GetShape(0);
        const int64_t dim = points_raw.GetShape(1);

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

        if (!force_addmm_path && UseKnnDirect(dim, batch_knn)) {
            // Direct |p−q|² path: no AddMM and no centering.
            DispatchKnnDirect<T, TIndex>(queue, points_raw.GetDataPtr<T>(),
                                         queries_raw.GetDataPtr<T>(), dim,
                                         num_points_i, num_queries_i, batch_knn,
                                         out_distances.GetDataPtr<T>(),
                                         out_indices.GetDataPtr<TIndex>());
            queue.wait_and_throw();
            continue;
        }

        if (!centered) {
            points_c = points;
            queries_c = queries;
            CenterPointsAndQueries<T>(points_c, queries_c);
            centered = true;
        }
        const Tensor points_i = points_c.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries_c.Slice(0, query_begin, query_end);

        // |p|² and |q|² for distance tiling.
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        int64_t tile_queries = 0, tile_points = 0;
        ChooseTileSize(num_queries_i, num_points_i, sizeof(T), tile_bytes,
                       tile_queries, tile_points, max_tile_queries,
                       tile_points_alignment);

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
            // P2: |q|² not in tile; AddQueryNormsToDistances adds it once.
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

                    SelectTopKQueries<T, TIndex>(
                            device, temp_view.GetDataPtr<T>(),
                            temp_view.GetStride(0), num_queries_iter,
                            num_points_iter, batch_knn, TIndex(p),
                            tile_sort_indices.GetDataPtr<TIndex>(),
                            tile_sort_indices.GetStride(0),
                            ttiv.GetDataPtr<TIndex>(), ttdv.GetDataPtr<T>(),
                            batch_knn);
                    MergeTopKQueries<T, TIndex>(
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
            AddQueryNormsToDistances<T, TIndex>(
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

// Fixed-radius search: uniform-grid (cell-list) algorithm, ported from
// FixedRadiusSearchImpl.cuh (CUDA). Two passes over the grid: count (to size
// the output) then gather; both visit only the 8 corner-adjacent hash bins
// per query (see FixedRadiusSearchSYCLImpl.h for the kernels).
template <class T, class TIndex>
void FixedRadiusSearchSYCL(const Tensor& points,
                           const Tensor& queries,
                           double radius,
                           const Tensor& points_row_splits,
                           const Tensor& queries_row_splits,
                           const Tensor& hash_table_splits,
                           const Tensor& hash_table_index,
                           const Tensor& hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           const bool sort,
                           Tensor& neighbors_index,
                           Tensor& neighbors_row_splits,
                           Tensor& neighbors_distance,
                           int64_t /*tile_bytes*/) {
    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const T radius_t = static_cast<T>(radius);
    const T threshold = FixedRadiusThreshold<T>(metric, radius_t);
    const T voxel_size = T(2) * radius_t;
    const T inv_voxel_size = T(1) / voxel_size;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    const T* points_ptr = points.GetDataPtr<T>();
    const T* queries_ptr = queries.GetDataPtr<T>();
    const uint32_t* hash_index_ptr = hash_table_index.GetDataPtr<uint32_t>();

    Tensor counts = Tensor::Empty({num_queries}, UInt32, device);
    uint32_t* counts_ptr = counts.GetDataPtr<uint32_t>();

    const int num_batches = static_cast<int>(points_row_splits.GetShape(0)) - 1;

    // ── Pass 1: count ────────────────────────────────────────────────────────
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const uint32_t first_cell_idx =
                hash_table_splits[batch_idx].Item<uint32_t>();
        const uint32_t hash_table_size =
                hash_table_splits[batch_idx + 1].Item<uint32_t>() -
                first_cell_idx;
        const uint32_t* cell_splits_i =
                hash_table_cell_splits.GetDataPtr<uint32_t>() + first_cell_idx;

        CountNeighborsSYCL<T>(queue, counts_ptr + query_begin, hash_index_ptr,
                              cell_splits_i, hash_table_size,
                              queries_ptr + 3 * query_begin,
                              query_end - query_begin, points_ptr,
                              inv_voxel_size, radius_t, metric,
                              ignore_query_point, threshold);
    }
    queue.wait_and_throw();

    // Build row_splits from counts (device inclusive scan; no host fallback).
    neighbors_row_splits = Tensor::Zeros({num_queries + 1}, Int64, device);
    int64_t* row_splits_ptr = neighbors_row_splits.GetDataPtr<int64_t>();
    {
        auto policy = oneapi::dpl::execution::make_device_policy(queue);
        // counts_ptr is uint32_t; scan directly into the int64_t row-splits
        // tail (offset by 1) so element 0 stays the required leading zero.
        // The explicit int64_t init forces 64-bit accumulation (avoids
        // overflow for very large neighbor counts).
        std::inclusive_scan(policy, counts_ptr, counts_ptr + num_queries,
                            row_splits_ptr + 1, std::plus<int64_t>(),
                            int64_t(0));
        queue.wait_and_throw();
    }
    int64_t total_neighbors = 0;
    if (num_queries > 0) {
        queue.memcpy(&total_neighbors, row_splits_ptr + num_queries,
                     sizeof(int64_t))
                .wait_and_throw();
    }

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, total_neighbors);
    if (return_distances || sort) {
        output_allocator.AllocDistances(&neighbors_distance_ptr,
                                        total_neighbors);
    } else {
        output_allocator.AllocDistances(&neighbors_distance_ptr, 0);
    }

    // ── Pass 2: gather ───────────────────────────────────────────────────────
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const uint32_t first_cell_idx =
                hash_table_splits[batch_idx].Item<uint32_t>();
        const uint32_t hash_table_size =
                hash_table_splits[batch_idx + 1].Item<uint32_t>() -
                first_cell_idx;
        const uint32_t* cell_splits_i =
                hash_table_cell_splits.GetDataPtr<uint32_t>() + first_cell_idx;

        WriteNeighborsSYCL<T, TIndex>(
                queue, neighbors_index_ptr, neighbors_distance_ptr,
                row_splits_ptr + query_begin, hash_index_ptr, cell_splits_i,
                hash_table_size, queries_ptr + 3 * query_begin,
                query_end - query_begin, points_ptr, inv_voxel_size, radius_t,
                metric, ignore_query_point, threshold,
                return_distances || sort);
    }
    queue.wait_and_throw();

    if (sort && total_neighbors > 0) {
        SortNeighborsByDistanceSYCL<T, TIndex>(
                device, neighbors_index_ptr, neighbors_distance_ptr,
                row_splits_ptr, num_queries, total_neighbors);
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    if (!return_distances)
        neighbors_distance = Tensor({0}, Dtype::FromType<T>(), device);
}

// Hybrid search: uniform-grid (cell-list) algorithm, ported from
// FixedRadiusSearchImpl.cuh (CUDA). A single pass over the grid's 8
// corner-adjacent hash bins per query counts all in-radius neighbors while
// keeping a running top-max_knn (see WriteNeighborsHybridSYCL in
// FixedRadiusSearchSYCLImpl.h, including its final per-query bubble sort).
template <class T, class TIndex>
void HybridSearchSYCL(const Tensor& points,
                      const Tensor& queries,
                      double radius,
                      int max_knn,
                      const Tensor& points_row_splits,
                      const Tensor& queries_row_splits,
                      const Tensor& hash_table_splits,
                      const Tensor& hash_table_index,
                      const Tensor& hash_table_cell_splits,
                      const Metric metric,
                      Tensor& neighbors_index,
                      Tensor& neighbors_count,
                      Tensor& neighbors_distance,
                      int64_t /*tile_bytes*/) {
    if (metric != Metric::L2) {
        utility::LogError("SYCL hybrid search only supports L2 metric.");
    }
    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const T radius_t = static_cast<T>(radius);
    const T threshold = radius_t * radius_t;  // L2: compare squared distances.
    const T voxel_size = T(2) * radius_t;
    const T inv_voxel_size = T(1) / voxel_size;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    const T* points_ptr = points.GetDataPtr<T>();
    const T* queries_ptr = queries.GetDataPtr<T>();
    const uint32_t* hash_index_ptr = hash_table_index.GetDataPtr<uint32_t>();

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    TIndex* neighbors_count_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, num_queries * max_knn,
                                  TIndex(-1));
    output_allocator.AllocDistances(&neighbors_distance_ptr,
                                    num_queries * max_knn, T(0));
    output_allocator.AllocCounts(&neighbors_count_ptr, num_queries, TIndex(0));

    const int num_batches = static_cast<int>(points_row_splits.GetShape(0)) - 1;
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const uint32_t first_cell_idx =
                hash_table_splits[batch_idx].Item<uint32_t>();
        const uint32_t hash_table_size =
                hash_table_splits[batch_idx + 1].Item<uint32_t>() -
                first_cell_idx;
        const uint32_t* cell_splits_i =
                hash_table_cell_splits.GetDataPtr<uint32_t>() + first_cell_idx;

        WriteNeighborsHybridSYCL<T, TIndex>(
                queue, neighbors_index_ptr + max_knn * query_begin,
                neighbors_distance_ptr + max_knn * query_begin,
                neighbors_count_ptr + query_begin, hash_index_ptr,
                cell_splits_i, hash_table_size, queries_ptr + 3 * query_begin,
                query_end - query_begin, points_ptr, inv_voxel_size, radius_t,
                threshold, max_knn);
    }
    queue.wait_and_throw();

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    neighbors_count = output_allocator.NeighborsCount();
}

#define INSTANTIATE(T, TIndex)                                                 \
    template void KnnSearchSYCL<T, TIndex>(                                    \
            const Tensor& points, const Tensor& points_row_splits,             \
            const Tensor& queries, const Tensor& queries_row_splits, int knn,  \
            Tensor& neighbors_index, Tensor& neighbors_row_splits,             \
            Tensor& neighbors_distance, int64_t tile_bytes,                    \
            int64_t max_tile_queries, int64_t tile_points_alignment,           \
            bool force_addmm_path);                                            \
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

template void BuildSpatialHashTableSYCL<float>(const Tensor& points,
                                               double radius,
                                               const Tensor& points_row_splits,
                                               const Tensor& hash_table_splits,
                                               Tensor& hash_table_index,
                                               Tensor& hash_table_cell_splits);
template void BuildSpatialHashTableSYCL<double>(const Tensor& points,
                                                double radius,
                                                const Tensor& points_row_splits,
                                                const Tensor& hash_table_splits,
                                                Tensor& hash_table_index,
                                                Tensor& hash_table_cell_splits);

}  // namespace nns
}  // namespace core
}  // namespace open3d
