// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file KnnSearchOpsSYCL.cpp
/// \brief Host driver for SYCL nearest-neighbor search (KNN, fixed-radius,
/// hybrid).
///
/// **What lives here**
/// - \ref KnnSearchSYCL — batched **k**-NN (L2); three internal paths (Direct /
/// AddMM).
/// - \ref FixedRadiusSearchSYCL — variable-length neighbors within `radius`
/// (grid).
/// - \ref HybridSearchSYCL — capped top-`max_knn` within `radius` + counts
/// (grid).
///
/// **Callers:** \ref NearestNeighborSearch / \ref KnnIndex / \ref
/// FixedRadiusIndex when dataset and query tensors are on a SYCL device. CPU
/// tensors use NanoFlann.
///
/// **Includes:** KNN device code in `kernel/KnnSearchSYCLImpl.h`; uniform-grid
/// kernels in `kernel/FixedRadiusSearchSYCLImpl.h`. Short index:
/// `nns/SYCL_DESIGN.md`.
/// **This file header** is the maintainer reference for path selection and
/// end-to-end flow.
///
/// \section KnnSyclOverview Three search modes in this file
///
/// | Function | Purpose |
/// |----------|---------|
/// | \ref KnnSearchSYCL | Fixed **k** nearest neighbors (L2), batched queries ×
/// dataset | | \ref FixedRadiusSearchSYCL | **All** neighbors within `radius`
/// (variable output size per query) | | \ref HybridSearchSYCL | Up to
/// **max_knn** closest neighbors within `radius` + in-radius count |
///
/// Fixed-radius and hybrid share the same spatial-hash **grid index** built in
/// \ref FixedRadiusIndex::SetTensorData (see \ref FixedRadiusSearchSYCLImpl.h).
/// The sections below focus on **KNN**; grid search is summarized at the end.
///
/// \section KnnSyclDecision KNN path selection
///
/// For each batch, `batch_knn = min(knn, num_points)` (convention **C5**).
/// CUDA KNN uses two strategies (small brute + GEMM + FAISS warp/block select);
/// SYCL uses **three** paths:
///
/// **Decision tree**
/// - If `force_addmm_path == false` **and** `1 ≤ dim ≤ 8` **and** `batch_knn ≤
/// 32`
///   → **Direct** brute-force L2 (`UseKnnDirect` / `DispatchKnnDirect`).
/// - Else center points and queries once (AddMM paths only; see below).
///   - If `batch_knn ≤ 512` → **AddMM fused** (running heap per query tile).
///   - Else → **AddMM large-k** (select + merge; may use oneDPL
///   `partial_sort`).
///
/// `force_addmm_path=true` is for benchmarks only — always skips Direct.
///
/// | Path | When | CUDA analogue | SYCL implementation |
/// |------|------|---------------|---------------------|
/// | **Direct** | `!force_addmm` ∧ dim∈[1,8] ∧ k≤32 | Small dim/k brute kernel
/// | Sub-group SLM tiles + shuffle-merge top-k | | **AddMM fused** | else,
/// k≤512 | GEMM + FAISS `BlockSelect` / heap | oneMKL AddMM + fused
/// `UpdateTopKFromTile` | | **AddMM large-k** | else, k>512 | Multi-pass FAISS
/// select | Tile select/merge + serial `partial_sort` (P8) |
///
/// Constants: `kSYCLKnnSmallKMax = 32`, `kSYCLKnnMidKMax = 512` (see
/// `KnnSearchSYCLImpl.h`).
///
/// \subsection KnnSyclDirect Direct path (`DispatchKnnDirect`)
///
/// **Idea:** Compute L2 as Σ_d (p_d − q_d)² directly — **no** AddMM, **no**
/// data centering, **no** deferred ‖q‖² term.
///
/// **Algorithm**
/// - One **sub-group** owns one query; several queries share a work-group.
/// - Dataset points are loaded in **SLM tiles** (double-buffered where
/// applicable).
/// - Each lane maintains a private sorted top-K; sub-group **shuffle-merge**;
///   lane 0 writes indices and distances.
/// - **Template params:** compile-time `NDIM` (1…8) and K bucket (1…32).
///   Float typically uses sub-group size 16; double prefers 8 when supported.
///
/// **Typical use:** 3D point clouds, small k — often the fastest path on Intel
/// Xe.
///
/// \subsection KnnSyclAddMMFused AddMM fused path (k ≤ 512)
///
/// **Idea:** Expand squared L2 as ‖q‖² − 2 q·p + ‖p‖². oneMKL \ref AddMM builds
/// the **−2 q·p** tile for each (query-tile × point-tile) pair; top-k selection
/// fuses ‖p‖² and running heap updates (`UpdateTopKFromTile`, `FinalizeTopK`).
///
/// **Algorithm**
/// 1. **Center** points and queries (subtract first dataset row) — reduces
/// float32
///    cancellation in ‖p‖² and ‖q‖² when coordinates or features have large
///    norm (CUDA GEMM KNN does not center; Direct SYCL path does not need it).
/// 2. Precompute ‖p‖² per point and ‖q‖² per query (or equivalent norms).
/// 3. For each tile pair: AddMM → partial tile **−2qp**; add ‖p‖², clamp ≥ 0,
///    update per-query **max-heap** (size = `KBucket(batch_knn)`).
/// 4. `FinalizeTopK`: heap-sort, add ‖q‖², write `batch_knn` neighbors.
///
/// **Heap storage:** k ≤ 32 → GRF-resident; k ∈ {64…512} → per-work-item
/// scratch.
///
/// \subsection KnnSyclAddMMLarge AddMM large-k path (k > 512)
///
/// **Idea:** Same tiled AddMM and centering as fused path, but
/// **SelectTopKQueries**
/// + **MergeTopKQueries** instead of one fused running heap across all tiles.
///
/// **Algorithm**
/// - Same centering, norms, and AddMM tile loop.
/// - Per point-tile: select tile top-k per query; merge into global
/// best-so-far.
/// - Once: `AddQueryNormsToDistances` (add ‖q‖², clamp).
///
/// **P8 caveat:** For very large k, merge/select may fall back to **per-query
/// serial `partial_sort`** (oneDPL has no cheap segmented top-k) — correct but
/// slow.
///
/// \section KnnSyclVsCuda KNN: SYCL vs CUDA (design differences)
///
/// | Topic | CUDA | SYCL (this file) |
/// |-------|------|------------------|
/// | Small k, low dim | Brute + GEMM paths | **+ Direct** sub-group path (no
/// GEMM) | | GEMM | cuBLAS | oneMKL via \ref AddMM | | Top-k on GEMM tiles |
/// FAISS warp/block select | Custom heap / select-merge / oneDPL | | Float32
/// stability | Uses norms as-is on GEMM path | **Centers** data before AddMM |
/// | Large k | Multi-pass masking (1024/2048) | Select/merge + `partial_sort`
/// per query |
///
/// \section KnnSyclConv AddMM KNN conventions (all non-Direct paths)
///
/// | Id | Rule |
/// |----|------|
/// | P2 | Tile stores partial distance **−2qp + ‖p‖²**; **‖q‖²** added once in
/// finalize | | C1 | Clamp partial / final distance **≥ 0** | | C4 | Equal
/// distance → **smaller point index** wins | | C5 | `batch_knn = min(knn,
/// num_points)` (caller / driver) |
///
/// \section KnnSyclFrs Fixed-radius (\ref FixedRadiusSearchSYCL) — summary
///
/// **Idea:** Return **every** dataset point within `radius` of each query;
/// output length varies per query (CSR layout). Ported from CUDA
/// `FixedRadiusSearchImpl.cuh`; device detail in \ref
/// FixedRadiusSearchSYCLImpl.h.
///
/// **Grid build** (once per index, `FixedRadiusIndex::SetTensorData`):
/// - Uniform spatial hash with **cell size `2×radius`**.
/// - `BuildSpatialHashTableSYCL`: count points per cell → oneDPL inclusive scan
/// →
///   scatter into CSR (`hash_table_cell_splits`, `hash_table_index`).
///
/// **Query algorithm**
/// 1. **Count:** `CountNeighborsSYCL` — for each query, visit **8
/// corner-adjacent**
///    bins; count points with squared L2 ≤ radius².
/// 2. Inclusive scan of counts → `neighbors_row_splits`; allocate index (and
/// optional
///    distance) buffers for the total neighbor count.
/// 3. **Gather:** `WriteNeighborsSYCL` — same 8 bins; write indices and squared
/// distances.
/// 4. Optional **`sort=true`:** `SortNeighborsByDistanceSYCL` — segmented
/// oneDPL
///    `sort_by_key` per query (ties not secondarily ordered by index; matches
///    CUDA `cub::DeviceSegmentedRadixSort::SortPairs`).
///
/// Typical use: full neighborhood inside a ball, not a fixed-k cap.
///
/// \section KnnSyclHybrid Hybrid (\ref HybridSearchSYCL) — summary
///
/// **Idea:** Count all in-radius neighbors, but store only the **closest
/// `max_knn`** in fixed `(num_queries, max_knn)` tensors plus per-query
/// in-radius **counts**. Same grid as fixed-radius
/// (`BuildSpatialHashTableSYCL`).
///
/// **Algorithm**
/// - Allocate fixed-size index/distance outputs (empty slots use sentinels).
/// - `WriteNeighborsHybridSYCL`: one pass over 8 bins per query — running
/// top-`max_knn`
///   (replace-current-max), full in-radius count, then bounded **bubble sort**
///   (≤ `max_knn`) for ascending distance order.
///
/// Typical use: FPFH, normals, covariances — “up to K neighbors inside radius.”
///
/// **Note:** Could be expressed as KNN-then-filter; deferred because the grid
/// port already avoids O(queries × points) brute force.
///
/// \section KnnSyclPrimitives Shared primitive map (CUDA → SYCL)
///
/// | Role | CUDA | SYCL |
/// |------|------|------|
/// | Dense GEMM | cuBLAS | oneMKL |
/// | Prefix sum | CUB | oneDPL / work-group scan |
/// | Segmented sort | CUB radix | oneDPL `sort_by_key` |
/// | Parallel loops | CUDA kernels | SYCL queues / \ref ParallelFor |
///
/// \section KnnSyclTodo Maintenance notes
///
/// - **Direct KNN** is the most tuned path; **AddMM KNN** paths may need
/// further optimization.
/// - `force_addmm_path` forces GEMM paths for regression / benchmark parity
/// testing.
/// - Hybrid could theoretically be “KNN then filter by radius”; deferred — grid
/// port
///   already avoids O(queries × points) brute force for radius-limited search.
///
/// \section KnnSyclRelated Related source files
///
/// | File | Role |
/// |------|------|
/// | `KnnSearchSYCLImpl.h` | Direct / AddMM SYCL kernels, `KBucket`, thresholds
/// | | `FixedRadiusSearchSYCLImpl.h` | Grid build, count/write/sort/hybrid
/// kernels | | `NearestNeighborSearch.cpp` | Device dispatch to this driver |
/// | `AddMM.h` | oneMKL batched GEMM for KNN tiles |

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

        CountNeighborsSYCL<T>(
                queue, counts_ptr + query_begin, hash_index_ptr, cell_splits_i,
                hash_table_size, queries_ptr + 3 * query_begin,
                query_end - query_begin, points_ptr, inv_voxel_size, radius_t,
                metric, ignore_query_point, threshold);
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
