// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Device-side SYCL implementations for KNN, fixed-radius, and hybrid search.
// Included only by KnnSearchOpsSYCL.cpp – not part of the public API.
//
// Design overview
// ───────────────
// The outer driver (KnnSearchOpsSYCL.cpp) uses oneMKL AddMM to compute the
// −2*q*p distance tile for each (query-tile × point-tile) pair.
//
// P1+P9 – Fused per-tile kernel (small k ≤ kSYCLKnnMidKMax):
//   UpdateTopKFromTileSYCL<K> runs one work-item per query.  It reads the
//   AddMM output, adds |p|² on the fly (P2 – |q|² deferred to Finalize), and
//   maintains a running max-heap in global memory.  Template K is the
//   compile-time heap capacity (batch_knn rounded up to the dispatch bucket).
//   For K ≤ kSYCLKnnSmallKMax (32) the d[K]/idx[K] arrays fit in GRF
//   registers, eliminating the old fixed local_dists[256] scratch spill that
//   occurred even when knn=3.  For K in {64,128,256,512} the arrays spill to
//   per-work-item scratch but the allocation is proportional to k, not fixed.
//
//   FinalizeTopKSYCL<K> heap-sorts the result and adds |q|² (C1 clamped) in
//   one pass, replacing the old separate query-norm Add_ pass.
//
// Legacy path (mid k, threshold search):
//   SelectTopKQueriesSYCL (K-dispatched heap) and MergeTopKQueriesSYCL serve
//   hybrid search and k > kSYCLKnnMidKMax.  Hybrid tiles use
//   CountAndSelectTopKQueriesSYCL to fuse radius counting with top-k.
//
// P2 – Drop |q|² from the distance tile:
//   All selection/count/gather kernels compare partial_dist = −2qp + |p|²
//   against an adjusted per-query threshold = radius² − |q|².  KNN callers
//   add |q|² back once in Finalize / AddQueryNormsToDistancesSYCL.
//
// C1 – Negative distance clamp: partial_dist = max(0, −2qp + |p|²).
// C4 – Tie-break: equal distances resolved by smaller global point index.
// C5 – Handled by caller: actual_k = min(knn, num_points).
//
// P8 NOTE: oneDPL has no segmented top-k.  For k > kSYCLKnnMidKMax the code
// falls back to a per-query serial partial_sort (correct but sequential).
// A SYCL port of Faiss BlockSelect (gpu/impl/L2Select.cu) would remove this.

#pragma once

#include <algorithm>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

// ─── Tile sizing ──────────────────────────────────────────────────────────

/// Compute tile dimensions that bound the −2*q*p tile to \p tile_bytes.
/// tile_queries is capped at 128 (a good oneMKL GEMM row-tile width).
inline void ChooseTileSizeSYCL(int64_t num_queries,
                               int64_t num_points,
                               int64_t element_size,
                               int64_t tile_bytes,
                               int64_t& tile_queries,
                               int64_t& tile_points) {
    tile_queries = std::min<int64_t>(num_queries, 128);
    tile_queries = std::max<int64_t>(tile_queries, 1);
    tile_points = std::max<int64_t>(tile_bytes / (tile_queries * element_size),
                                    int64_t(256));
    tile_points = std::min<int64_t>(tile_points, num_points);
    tile_points = std::max<int64_t>(tile_points, 1);
}

// ─── Compile-time heap helpers ────────────────────────────────────────────
// C4 tie-break: smaller global index wins on equal distance.

/// Heapify-down for a compile-time max-heap of size K.  K is a constant so
/// the compiler can unroll the loop and keep d[]/idx[] in GRF for K ≤ 32.
template <typename T, typename TIndex, int K>
inline void HeapifyDown(T* d, TIndex* idx, int root) {
    while (true) {
        int left = 2 * root + 1, right = 2 * root + 2, largest = root;
        if (left < K && (d[left] > d[largest] ||
                         (d[left] == d[largest] && idx[left] > idx[largest])))
            largest = left;
        if (right < K && (d[right] > d[largest] || (d[right] == d[largest] &&
                                                    idx[right] > idx[largest])))
            largest = right;
        if (largest == root) break;
        T td = d[root];
        d[root] = d[largest];
        d[largest] = td;
        TIndex ti = idx[root];
        idx[root] = idx[largest];
        idx[largest] = ti;
        root = largest;
    }
}

/// Heap-sort a compile-time max-heap of size K into ascending order.
template <typename T, typename TIndex, int K>
inline void HeapSort(T* d, TIndex* idx) {
    for (int end = K - 1; end > 0; --end) {
        T td = d[0];
        d[0] = d[end];
        d[end] = td;
        TIndex ti = idx[0];
        idx[0] = idx[end];
        idx[end] = ti;
        int root = 0;
        while (true) {
            int left = 2 * root + 1, right = 2 * root + 2, largest = root;
            if (left < end &&
                (d[left] > d[largest] ||
                 (d[left] == d[largest] && idx[left] > idx[largest])))
                largest = left;
            if (right < end &&
                (d[right] > d[largest] ||
                 (d[right] == d[largest] && idx[right] > idx[largest])))
                largest = right;
            if (largest == root) break;
            T td2 = d[root];
            d[root] = d[largest];
            d[largest] = td2;
            TIndex ti2 = idx[root];
            idx[root] = idx[largest];
            idx[largest] = ti2;
            root = largest;
        }
    }
}

// ─── Fused per-tile top-K update (compile-time K, small/mid k path) ───────

/// Update the per-query running top-K heap using one point-column tile.
///
/// One work-item per query; K is the compile-time heap capacity (= dispatch
/// bucket ≥ actual knn).
///
///  K ≤ kSYCLKnnSmallKMax (32): d[K]/idx[K] live in GRF – no scratch spill.
///  K ≤ kSYCLKnnMidKMax  (512): spill to scratch, but proportional to K,
///                               not the old fixed 256.
///
/// Fuses three passes from the old code: Add_(point_norms) +
/// SelectTopKQueriesSYCL + MergeTopKQueriesSYCL.
///
/// P2: |q|² NOT added; callers add it once in FinalizeTopKSYCL.
/// C1: partial_dist = max(0, −2qp + |p|²).
/// C4: equal distances resolved by smaller global point index.
///
/// @param neg2qp_ptr      −2*q*p tile from AddMM, shape (num_q, dist_stride).
/// @param point_norms_ptr |p|² for this tile, shape (num_points,).
/// @param point_offset    Global index of this tile's first point.
/// @param best_dist_ptr   Running top-K distances (max-heap), (num_q, K).
/// @param best_idx_ptr    Running top-K global indices,       (num_q, K).
/// @param use_threshold   True for radius / hybrid search.
/// @param threshold       Adjusted per-query threshold = radius² − |q|².
///                        Caller passes the per-query array value for each
///                        work-item's own query; the scalar path is used by
///                        the hybrid SelectTopKQueriesSYCL variant.
template <typename T, typename TIndex, int K>
void UpdateTopKFromTileSYCL(sycl::queue& queue,
                            const T* neg2qp_ptr,
                            int64_t distance_stride,
                            const T* point_norms_ptr,
                            int64_t num_queries,
                            int64_t num_points,
                            TIndex point_offset,
                            T* best_dist_ptr,
                            TIndex* best_idx_ptr,
                            bool use_threshold,
                            T threshold) {
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
                const T* qrow = neg2qp_ptr + q * distance_stride;
                T* qd = best_dist_ptr + q * K;
                TIndex* qi = best_idx_ptr + q * K;

                // Load running best into private registers (or scratch for
                // large K).
                T d[K];
                TIndex idx[K];
                for (int i = 0; i < K; ++i) {
                    d[i] = qd[i];
                    idx[i] = qi[i];
                }

                // Scan: fused |p|² add, heap insert.
                // Note: partial_dist = −2qp + |p|² may be negative (|q|² not
                // yet added).  Do NOT clamp here; C1 clamping is applied in
                // FinalizeTopKSYCL / GatherWithinThreshold once |q|² is added
                // back.
                for (int64_t p = 0; p < num_points; ++p) {
                    const T dist = qrow[p] + point_norms_ptr[p];
                    if (use_threshold && dist > threshold) continue;
                    const TIndex gp = point_offset + static_cast<TIndex>(p);
                    // d[0] = heap root = current k-th worst; insert if better.
                    if (dist < d[0] || (dist == d[0] && gp < idx[0])) {
                        d[0] = dist;
                        idx[0] = gp;
                        HeapifyDown<T, TIndex, K>(d, idx, 0);
                    }
                }

                for (int i = 0; i < K; ++i) {
                    qd[i] = d[i];
                    qi[i] = idx[i];
                }
            });
}

/// Heap-sort the running top-K, add |q|² (P2), clamp ≥ 0 (C1), and write
/// the first actual_k entries to the output buffers.  Called once after all
/// point tiles (small-k KNN path only).
///
/// @param running_dist_ptr  (num_q, K) – max-heap from UpdateTopKFromTile.
/// @param running_idx_ptr   (num_q, K) – corresponding global indices.
/// @param out_dist_ptr      (num_q, actual_k) – final sorted distances.
/// @param out_idx_ptr       (num_q, actual_k) – final sorted indices.
/// @param actual_k          Real knn value (≤ K).
/// @param query_norms_ptr   |q|² per query (nullptr to skip, e.g. threshold).
template <typename T, typename TIndex, int K>
void FinalizeTopKSYCL(sycl::queue& queue,
                      int64_t num_queries,
                      const T* running_dist_ptr,
                      const TIndex* running_idx_ptr,
                      T* out_dist_ptr,
                      TIndex* out_idx_ptr,
                      int64_t actual_k,
                      const T* query_norms_ptr) {
    queue.parallel_for(sycl::range<1>(num_queries),
                       [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                           const int64_t q = id[0];
                           T d[K];
                           TIndex idx[K];
                           for (int i = 0; i < K; ++i) {
                               d[i] = running_dist_ptr[q * K + i];
                               idx[i] = running_idx_ptr[q * K + i];
                           }
                           HeapSort<T, TIndex, K>(d, idx);

                           const T qnorm =
                                   query_norms_ptr ? query_norms_ptr[q] : T(0);
                           T* qout_d = out_dist_ptr + q * actual_k;
                           TIndex* qout_i = out_idx_ptr + q * actual_k;
                           for (int64_t i = 0; i < actual_k; ++i) {
                               T dist = d[i];
                               if (query_norms_ptr)
                                   dist = sycl::fmax(T(0), dist + qnorm);
                               qout_d[i] = dist;
                               qout_i[i] = idx[i];
                           }
                       });
}

// ─── K-dispatch helpers ───────────────────────────────────────────────────
// K buckets: 1, 2, 4, 8, 16, 32 (GRF register path, no spill) and
//            64, 128, 256, 512  (scratch-resident, proportional alloc).
// Dispatch is on the bucket size, not the raw knn, so callers round up first.

/// Return the smallest dispatch-bucket value ≥ k.
inline int64_t KBucket(int64_t k) {
    if (k <= 1) return 1;
    if (k <= 2) return 2;
    if (k <= 4) return 4;
    if (k <= 8) return 8;
    if (k <= 16) return 16;
    if (k <= 32) return 32;
    if (k <= 64) return 64;
    if (k <= 128) return 128;
    if (k <= 256) return 256;
    return 512;
}

template <typename T, typename TIndex>
void DispatchUpdateTopKFromTile(sycl::queue& queue,
                                const T* neg2qp_ptr,
                                int64_t distance_stride,
                                const T* point_norms_ptr,
                                int64_t num_queries,
                                int64_t num_points,
                                int64_t k_bucket,
                                TIndex point_offset,
                                T* best_dist_ptr,
                                TIndex* best_idx_ptr,
                                bool use_threshold,
                                T threshold) {
#define CALL_UPDATE(Kval)                                                     \
    UpdateTopKFromTileSYCL<T, TIndex, Kval>(                                  \
            queue, neg2qp_ptr, distance_stride, point_norms_ptr, num_queries, \
            num_points, point_offset, best_dist_ptr, best_idx_ptr,            \
            use_threshold, threshold)
    if (k_bucket <= 1)
        CALL_UPDATE(1);
    else if (k_bucket <= 2)
        CALL_UPDATE(2);
    else if (k_bucket <= 4)
        CALL_UPDATE(4);
    else if (k_bucket <= 8)
        CALL_UPDATE(8);
    else if (k_bucket <= 16)
        CALL_UPDATE(16);
    else if (k_bucket <= 32)
        CALL_UPDATE(32);
    else if (k_bucket <= 64)
        CALL_UPDATE(64);
    else if (k_bucket <= 128)
        CALL_UPDATE(128);
    else if (k_bucket <= 256)
        CALL_UPDATE(256);
    else
        CALL_UPDATE(512);
#undef CALL_UPDATE
}

template <typename T, typename TIndex>
void DispatchFinalizeTopK(sycl::queue& queue,
                          int64_t num_queries,
                          const T* running_dist_ptr,
                          const TIndex* running_idx_ptr,
                          T* out_dist_ptr,
                          TIndex* out_idx_ptr,
                          int64_t actual_k,
                          int64_t k_bucket,
                          const T* query_norms_ptr) {
#define CALL_FINALIZE(Kval)                                                 \
    FinalizeTopKSYCL<T, TIndex, Kval>(queue, num_queries, running_dist_ptr, \
                                      running_idx_ptr, out_dist_ptr,        \
                                      out_idx_ptr, actual_k, query_norms_ptr)
    if (k_bucket <= 1)
        CALL_FINALIZE(1);
    else if (k_bucket <= 2)
        CALL_FINALIZE(2);
    else if (k_bucket <= 4)
        CALL_FINALIZE(4);
    else if (k_bucket <= 8)
        CALL_FINALIZE(8);
    else if (k_bucket <= 16)
        CALL_FINALIZE(16);
    else if (k_bucket <= 32)
        CALL_FINALIZE(32);
    else if (k_bucket <= 64)
        CALL_FINALIZE(64);
    else if (k_bucket <= 128)
        CALL_FINALIZE(128);
    else if (k_bucket <= 256)
        CALL_FINALIZE(256);
    else
        CALL_FINALIZE(512);
#undef CALL_FINALIZE
}

// ─── Direct-distance KNN (no AddMM, no centering) ────────────────────────
// Cooperative, SLM-tiled brute-force KNN for low-dimension, low-k queries
// (the dominant point-cloud regime: D ≤ kSYCLKnnDirectMaxDim,
// K ≤ kSYCLKnnSmallKMax). This bypasses the oneMKL AddMM tiling used by the
// rest of this file entirely: |p-q|² is accumulated directly as a sum of
// squared per-dimension differences, so there is no |q|²-2qp+|p|²
// cancellation risk and therefore neither data centering nor the P2
// query-norm deferral above are needed for this path.
//
// Work distribution: one sub-group of compile-time width SG handles one
// query. `subgroups_per_wg` sub-groups (queries) share a work-group and
// cooperatively cache each `tile_points`-sized chunk of the dataset in SLM
// once, amortizing that load across every query in the work-group instead
// of re-reading points per query. Tiles are double-buffered in SLM so the
// next tile's cooperative load is issued (software-pipelined prefetch)
// before the current tile is consumed, letting the GPU's load pipeline
// overlap with the distance-compute loop instead of stalling on it.
//
// Each lane keeps a private ascending-sorted top-K (register-resident for
// K ≤ kSYCLKnnSmallKMax) while scanning its strided share of each tile.
// Once all tiles are scanned, the SG lanes combine their per-lane top-K via
// a register/shuffle all-reduce merge (sycl::select_from_group): at each of
// the log2(SG) rounds, a lane exchanges its whole sorted K-array with its
// XOR partner and 2-way-merges the two sorted arrays, keeping only the
// smallest K. After the loop every lane holds the identical final top-K for
// the query (no SLM/global traffic for the merge itself) — lane 0 writes
// it out.
//
// `subgroups_per_wg` and `tile_points` are plain runtime parameters (not
// template parameters), so launch geometry can be retuned by benchmarking
// without recompiling; SG, NDIM and K stay compile-time so the per-lane
// arrays remain register-resident.

// Performance note: This direct path is 2-3 orders of magnitude faster than the
// indirect path on an A770! Both paths can benefit from tuning.

constexpr int64_t kSYCLKnnDirectSubgroupSize = 16;
constexpr int64_t kSYCLKnnDirectSubgroupsPerWG = 8;
constexpr int64_t kSYCLKnnDirectTilePoints = 256;
constexpr int64_t kSYCLKnnDirectMaxDim = 8;

template <typename T, typename TIndex, int NDIM, int K, int SG>
void KnnDirectSYCL(sycl::queue& queue,
                   const T* points_ptr,
                   const T* queries_ptr,
                   int64_t num_points,
                   int64_t num_queries,
                   int64_t actual_k,
                   T* out_dist_ptr,
                   TIndex* out_idx_ptr,
                   int64_t subgroups_per_wg,
                   int64_t tile_points) {
    if (num_points <= 0 || num_queries <= 0) return;

    const int64_t wg_size = subgroups_per_wg * SG;
    const int64_t num_wgs =
            (num_queries + subgroups_per_wg - 1) / subgroups_per_wg;
    const int64_t global_size = num_wgs * wg_size;
    const int64_t tp = std::min<int64_t>(tile_points, num_points);
    const int64_t num_tiles = (num_points + tp - 1) / tp;

    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> slm(sycl::range<1>(2 * tp * NDIM), h);
        h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global_size),
                                  sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> it)
                        [[sycl::reqd_sub_group_size(SG)]]
                        [[intel::kernel_args_restrict]] {
                    const auto sg = it.get_sub_group();
                    const int64_t lane = sg.get_local_id()[0];
                    const int64_t sg_id_in_wg = sg.get_group_id()[0];
                    const int64_t wg_id = it.get_group(0);
                    const int64_t local_lin = it.get_local_linear_id();
                    const int64_t local_range = it.get_local_range(0);

                    const int64_t query_idx =
                            wg_id * subgroups_per_wg + sg_id_in_wg;
                    const bool active_query = query_idx < num_queries;

                    // Load this sub-group's query once. Inactive sub-groups
                    // (tail of the last work-group) load row 0 so every
                    // lane in the work-group stays in lock-step for the
                    // shared SLM tile loads / barriers below.
                    T q[NDIM];
                    {
                        const int64_t qrow = active_query ? query_idx : 0;
                        for (int d = 0; d < NDIM; ++d) {
                            q[d] = queries_ptr[qrow * NDIM + d];
                        }
                    }

                    // Private ascending-sorted top-K, sentinel-filled.
                    T d[K];
                    TIndex idx[K];
                    for (int i = 0; i < K; ++i) {
                        d[i] = std::numeric_limits<T>::max();
                        idx[i] = TIndex(-1);
                    }

                    for (int64_t t = 0; t < num_tiles; ++t) {
                        const int64_t cur = t & 1;
                        const int64_t cur_start = t * tp;
                        const int64_t cur_n =
                                std::min<int64_t>(tp, num_points - cur_start);

                        if (t == 0) {
                            // Cooperative whole-work-group load of tile 0.
                            for (int64_t e = local_lin; e < cur_n * NDIM;
                                 e += local_range) {
                                const int64_t p = e / NDIM, dd = e % NDIM;
                                slm[cur * tp * NDIM + e] =
                                        points_ptr[(cur_start + p) * NDIM +
                                                  dd];
                            }
                            sycl::group_barrier(it.get_group());
                        }

                        // Prefetch: cooperatively load the NEXT tile into
                        // the other SLM buffer before computing on the
                        // current one, so its global-memory loads are
                        // issued early and can overlap with this tile's
                        // compute below.
                        if (t + 1 < num_tiles) {
                            const int64_t nxt = 1 - cur;
                            const int64_t nxt_start = (t + 1) * tp;
                            const int64_t nxt_n = std::min<int64_t>(
                                    tp, num_points - nxt_start);
                            for (int64_t e = local_lin; e < nxt_n * NDIM;
                                 e += local_range) {
                                const int64_t p = e / NDIM, dd = e % NDIM;
                                slm[nxt * tp * NDIM + e] =
                                        points_ptr[(nxt_start + p) * NDIM +
                                                  dd];
                            }
                        }

                        if (active_query) {
                            for (int64_t p_local = lane; p_local < cur_n;
                                 p_local += SG) {
                                T dist = T(0);
                                const int64_t base =
                                        cur * tp * NDIM + p_local * NDIM;
                                for (int dd = 0; dd < NDIM; ++dd) {
                                    const T diff = q[dd] - slm[base + dd];
                                    dist += diff * diff;
                                }
                                const TIndex gp = static_cast<TIndex>(
                                        cur_start + p_local);
                                if (dist < d[K - 1] ||
                                    (dist == d[K - 1] && gp < idx[K - 1])) {
                                    int pos = K - 1;
                                    d[pos] = dist;
                                    idx[pos] = gp;
                                    while (pos > 0 &&
                                          (d[pos - 1] > d[pos] ||
                                           (d[pos - 1] == d[pos] &&
                                            idx[pos - 1] > idx[pos]))) {
                                        T td = d[pos - 1];
                                        d[pos - 1] = d[pos];
                                        d[pos] = td;
                                        TIndex ti = idx[pos - 1];
                                        idx[pos - 1] = idx[pos];
                                        idx[pos] = ti;
                                        --pos;
                                    }
                                }
                            }
                        }

                        // Bottom barrier: (a) the next-tile load issued
                        // above must finish before the following iteration
                        // treats it as "current"; (b) every lane must be
                        // done reading the current buffer before it is
                        // overwritten two iterations from now.
                        sycl::group_barrier(it.get_group());
                    }

                    if (!active_query) return;

                    // Sub-group all-reduce merge: after log2(SG)
                    // shuffle/merge rounds every lane holds the identical
                    // final top-K for this query, entirely register
                    // resident.
                    for (int step = 1; step < SG; step <<= 1) {
                        const int64_t partner = lane ^ step;
                        T od[K];
                        TIndex oidx[K];
                        for (int i = 0; i < K; ++i) {
                            od[i] = sycl::select_from_group(sg, d[i],
                                                            partner);
                            oidx[i] = sycl::select_from_group(sg, idx[i],
                                                              partner);
                        }
                        T md[K];
                        TIndex mi[K];
                        int a = 0, b = 0;
                        for (int o = 0; o < K; ++o) {
                            const bool take_a =
                                    (b >= K) ||
                                    (a < K &&
                                     (d[a] < od[b] ||
                                      (d[a] == od[b] && idx[a] <= oidx[b])));
                            if (take_a) {
                                md[o] = d[a];
                                mi[o] = idx[a];
                                ++a;
                            } else {
                                md[o] = od[b];
                                mi[o] = oidx[b];
                                ++b;
                            }
                        }
                        for (int o = 0; o < K; ++o) {
                            d[o] = md[o];
                            idx[o] = mi[o];
                        }
                    }

                    if (lane == 0) {
                        T* od = out_dist_ptr + query_idx * actual_k;
                        TIndex* oi = out_idx_ptr + query_idx * actual_k;
                        for (int64_t i = 0; i < actual_k; ++i) {
                            od[i] = sycl::fmax(T(0), d[i]);  // C1
                            oi[i] = idx[i];
                        }
                    }
                });
    });
}

template <typename T, typename TIndex, int NDIM>
void DispatchKnnDirectK(sycl::queue& queue,
                       const T* points_ptr,
                       const T* queries_ptr,
                       int64_t num_points,
                       int64_t num_queries,
                       int64_t actual_k,
                       T* out_dist_ptr,
                       TIndex* out_idx_ptr,
                       int64_t subgroups_per_wg,
                       int64_t tile_points) {
    const int64_t k_bucket = KBucket(actual_k);
#define CALL_DIRECT(Kval)                                                  \
    KnnDirectSYCL<T, TIndex, NDIM, Kval, kSYCLKnnDirectSubgroupSize>(      \
            queue, points_ptr, queries_ptr, num_points, num_queries,       \
            actual_k, out_dist_ptr, out_idx_ptr, subgroups_per_wg,         \
            tile_points)
    if (k_bucket <= 1)
        CALL_DIRECT(1);
    else if (k_bucket <= 2)
        CALL_DIRECT(2);
    else if (k_bucket <= 4)
        CALL_DIRECT(4);
    else if (k_bucket <= 8)
        CALL_DIRECT(8);
    else if (k_bucket <= 16)
        CALL_DIRECT(16);
    else
        CALL_DIRECT(32);
#undef CALL_DIRECT
}

/// Dispatch the direct-distance KNN path by compile-time dimension (1..8)
/// and K-bucket (≤ 32). Writes directly into the caller-allocated output
/// buffers; no public API / build plumbing changes are required.
template <typename T, typename TIndex>
void DispatchKnnDirectSYCL(
        sycl::queue& queue,
        const T* points_ptr,
        const T* queries_ptr,
        int64_t dim,
        int64_t num_points,
        int64_t num_queries,
        int64_t actual_k,
        T* out_dist_ptr,
        TIndex* out_idx_ptr,
        int64_t subgroups_per_wg = kSYCLKnnDirectSubgroupsPerWG,
        int64_t tile_points = kSYCLKnnDirectTilePoints) {
#define CALL_DIM(NDIMVAL)                                                \
    DispatchKnnDirectK<T, TIndex, NDIMVAL>(                              \
            queue, points_ptr, queries_ptr, num_points, num_queries,     \
            actual_k, out_dist_ptr, out_idx_ptr, subgroups_per_wg,       \
            tile_points)
    switch (dim) {
        case 1:
            CALL_DIM(1);
            break;
        case 2:
            CALL_DIM(2);
            break;
        case 3:
            CALL_DIM(3);
            break;
        case 4:
            CALL_DIM(4);
            break;
        case 5:
            CALL_DIM(5);
            break;
        case 6:
            CALL_DIM(6);
            break;
        case 7:
            CALL_DIM(7);
            break;
        case 8:
            CALL_DIM(8);
            break;
        default:
            utility::LogError(
                    "DispatchKnnDirectSYCL only supports dim 1 to {}.",
                    kSYCLKnnDirectMaxDim);
    }
#undef CALL_DIM
}

/// True if (dim, knn) qualifies for the direct-distance SYCL KNN path.
inline bool UseKnnDirectSYCL(int64_t dim, int64_t knn) {
    return dim >= 1 && dim <= kSYCLKnnDirectMaxDim && knn <= kSYCLKnnSmallKMax;
}

// ─── Legacy Select / Merge (mid-k and large-k paths) ─────────────────────
// These keep the existing algorithm but fix C1 (clamp), C4 (tie-break), and
// add P2 support (no |q|² in distances_ptr; per-query threshold for hybrid).

namespace detail {

/// Heapify-down for a max-heap of runtime size active_k (≤ compile-time K).
template <typename T, typename TIndex, int K>
inline void HeapifyDownActive(T* local_d,
                              TIndex* local_i,
                              int root,
                              int active_k) {
    int i = root;
    while (true) {
        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
        if (left < active_k && (local_d[left] > local_d[largest] ||
                                (local_d[left] == local_d[largest] &&
                                 local_i[left] > local_i[largest])))
            largest = left;
        if (right < active_k && (local_d[right] > local_d[largest] ||
                                 (local_d[right] == local_d[largest] &&
                                  local_i[right] > local_i[largest])))
            largest = right;
        if (largest == i) break;
        T td = local_d[i];
        local_d[i] = local_d[largest];
        local_d[largest] = td;
        TIndex ti = local_i[i];
        local_i[i] = local_i[largest];
        local_i[largest] = ti;
        i = largest;
    }
}

/// Mid-k heap select with compile-time bucket K (≥ knn, from KBucket).
template <typename T, typename TIndex, int K>
void SelectTopKQueriesHeapSYCL(sycl::queue& queue,
                               const T* distances_ptr,
                               int64_t distance_query_stride,
                               int64_t num_queries,
                               int64_t num_points,
                               int64_t knn,
                               TIndex index_offset,
                               TIndex* out_indices_ptr,
                               T* out_distances_ptr,
                               int64_t out_query_stride,
                               bool use_threshold,
                               const T* query_norms_ptr,
                               T radius_sq,
                               T scalar_threshold) {
    const T inf = std::numeric_limits<T>::max();
    const int64_t actual_knn = std::min(knn, num_points);

    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
                const T* qd = distances_ptr + q * distance_query_stride;
                TIndex* qout_i = out_indices_ptr + q * out_query_stride;
                T* qout_d = out_distances_ptr + q * out_query_stride;

                const T thr = (use_threshold && query_norms_ptr)
                                      ? (radius_sq - query_norms_ptr[q])
                                      : scalar_threshold;

                T local_d[K];
                TIndex local_i[K];
                for (int k = 0; k < actual_knn; ++k) {
                    local_d[k] = inf;
                    local_i[k] = TIndex(-1);
                }

                for (TIndex p = 0; p < static_cast<TIndex>(num_points); ++p) {
                    const T dist = qd[p];
                    if (use_threshold && dist > thr) continue;
                    if (dist < local_d[0] ||
                        (dist == local_d[0] &&
                         index_offset + p < index_offset + local_i[0])) {
                        local_d[0] = dist;
                        local_i[0] = p;
                        HeapifyDownActive<T, TIndex, K>(
                                local_d, local_i, 0,
                                static_cast<int>(actual_knn));
                    }
                }

                for (int i = 1; i < actual_knn; ++i) {
                    T key_d = local_d[i];
                    TIndex key_i = local_i[i];
                    int j = i - 1;
                    while (j >= 0 &&
                           (local_d[j] > key_d ||
                            (local_d[j] == key_d && local_i[j] > key_i))) {
                        local_d[j + 1] = local_d[j];
                        local_i[j + 1] = local_i[j];
                        j--;
                    }
                    local_d[j + 1] = key_d;
                    local_i[j + 1] = key_i;
                }

                for (int64_t k = 0; k < knn; ++k) {
                    if (k >= actual_knn || local_i[k] == TIndex(-1)) {
                        qout_i[k] = TIndex(-1);
                        qout_d[k] = inf;
                    } else {
                        qout_i[k] = index_offset + local_i[k];
                        qout_d[k] = local_d[k];
                    }
                }
            });
}

/// Hybrid tile pass: count points within radius and select top-k in one scan.
template <typename T, typename TIndex, int K>
void CountAndSelectTopKQueriesHeapSYCL(sycl::queue& queue,
                                       const T* partial_dist_ptr,
                                       int64_t distance_stride,
                                       int64_t num_queries,
                                       int64_t num_points,
                                       int64_t knn,
                                       TIndex index_offset,
                                       const T* query_norms_ptr,
                                       T radius_sq,
                                       int64_t* counts_ptr,
                                       TIndex* out_indices_ptr,
                                       T* out_distances_ptr,
                                       int64_t out_query_stride) {
    const T inf = std::numeric_limits<T>::max();
    const int64_t actual_knn = std::min(knn, num_points);

    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
                const T thresh = radius_sq - query_norms_ptr[q];
                const T* qd = partial_dist_ptr + q * distance_stride;
                TIndex* qout_i = out_indices_ptr + q * out_query_stride;
                T* qout_d = out_distances_ptr + q * out_query_stride;

                T local_d[K];
                TIndex local_i[K];
                for (int k = 0; k < actual_knn; ++k) {
                    local_d[k] = inf;
                    local_i[k] = TIndex(-1);
                }

                int64_t cnt = 0;
                for (int64_t p = 0; p < num_points; ++p) {
                    const T dist = qd[p];
                    if (dist <= thresh) {
                        ++cnt;
                        if (dist < local_d[0] ||
                            (dist == local_d[0] &&
                             index_offset + static_cast<TIndex>(p) <
                                     index_offset + local_i[0])) {
                            local_d[0] = dist;
                            local_i[0] = static_cast<TIndex>(p);
                            HeapifyDownActive<T, TIndex, K>(
                                    local_d, local_i, 0,
                                    static_cast<int>(actual_knn));
                        }
                    }
                }
                counts_ptr[q] += cnt;

                for (int i = 1; i < actual_knn; ++i) {
                    T key_d = local_d[i];
                    TIndex key_i = local_i[i];
                    int j = i - 1;
                    while (j >= 0 &&
                           (local_d[j] > key_d ||
                            (local_d[j] == key_d && local_i[j] > key_i))) {
                        local_d[j + 1] = local_d[j];
                        local_i[j + 1] = local_i[j];
                        j--;
                    }
                    local_d[j + 1] = key_d;
                    local_i[j + 1] = key_i;
                }

                for (int64_t k = 0; k < knn; ++k) {
                    if (k >= actual_knn || local_i[k] == TIndex(-1)) {
                        qout_i[k] = TIndex(-1);
                        qout_d[k] = inf;
                    } else {
                        qout_i[k] = index_offset + local_i[k];
                        qout_d[k] = local_d[k];
                    }
                }
            });
}

}  // namespace detail

template <typename T, typename TIndex>
void DispatchSelectTopKQueriesSYCL(sycl::queue& queue,
                                   const T* distances_ptr,
                                   int64_t distance_query_stride,
                                   int64_t num_queries,
                                   int64_t num_points,
                                   int64_t knn,
                                   int64_t k_bucket,
                                   TIndex index_offset,
                                   TIndex* out_indices_ptr,
                                   T* out_distances_ptr,
                                   int64_t out_query_stride,
                                   bool use_threshold,
                                   const T* query_norms_ptr,
                                   T radius_sq,
                                   T scalar_threshold) {
#define CALL_SELECT(Kval)                                                      \
    detail::SelectTopKQueriesHeapSYCL<T, TIndex, Kval>(                        \
            queue, distances_ptr, distance_query_stride, num_queries,          \
            num_points, knn, index_offset, out_indices_ptr, out_distances_ptr, \
            out_query_stride, use_threshold, query_norms_ptr, radius_sq,       \
            scalar_threshold)
    if (k_bucket <= 1)
        CALL_SELECT(1);
    else if (k_bucket <= 2)
        CALL_SELECT(2);
    else if (k_bucket <= 4)
        CALL_SELECT(4);
    else if (k_bucket <= 8)
        CALL_SELECT(8);
    else if (k_bucket <= 16)
        CALL_SELECT(16);
    else if (k_bucket <= 32)
        CALL_SELECT(32);
    else if (k_bucket <= 64)
        CALL_SELECT(64);
    else if (k_bucket <= 128)
        CALL_SELECT(128);
    else if (k_bucket <= 256)
        CALL_SELECT(256);
    else
        CALL_SELECT(512);
#undef CALL_SELECT
}

template <typename T, typename TIndex>
void DispatchCountAndSelectTopKQueriesSYCL(sycl::queue& queue,
                                           const T* partial_dist_ptr,
                                           int64_t distance_stride,
                                           int64_t num_queries,
                                           int64_t num_points,
                                           int64_t knn,
                                           int64_t k_bucket,
                                           TIndex index_offset,
                                           const T* query_norms_ptr,
                                           T radius_sq,
                                           int64_t* counts_ptr,
                                           TIndex* out_indices_ptr,
                                           T* out_distances_ptr,
                                           int64_t out_query_stride) {
#define CALL_COUNT_SELECT(Kval)                                                \
    detail::CountAndSelectTopKQueriesHeapSYCL<T, TIndex, Kval>(                \
            queue, partial_dist_ptr, distance_stride, num_queries, num_points, \
            knn, index_offset, query_norms_ptr, radius_sq, counts_ptr,         \
            out_indices_ptr, out_distances_ptr, out_query_stride)
    if (k_bucket <= 1)
        CALL_COUNT_SELECT(1);
    else if (k_bucket <= 2)
        CALL_COUNT_SELECT(2);
    else if (k_bucket <= 4)
        CALL_COUNT_SELECT(4);
    else if (k_bucket <= 8)
        CALL_COUNT_SELECT(8);
    else if (k_bucket <= 16)
        CALL_COUNT_SELECT(16);
    else if (k_bucket <= 32)
        CALL_COUNT_SELECT(32);
    else if (k_bucket <= 64)
        CALL_COUNT_SELECT(64);
    else if (k_bucket <= 128)
        CALL_COUNT_SELECT(128);
    else if (k_bucket <= 256)
        CALL_COUNT_SELECT(256);
    else
        CALL_COUNT_SELECT(512);
#undef CALL_COUNT_SELECT
}

/// Select the smallest knn partial distances per query.
/// P2: distances_ptr contains partial dists (no |q|²); callers add it after.
/// C1: clamp each distance to max(0, ...) before comparing.
/// C4: equal distances resolved by global index.
///
/// @param query_norms_ptr  Per-query |q|² (size num_queries). When
///   use_threshold is true and this is non-null, threshold is
///   radius_sq − query_norms_ptr[q] (P2 hybrid). Otherwise scalar_threshold.
template <typename T, typename TIndex>
void SelectTopKQueriesSYCL(const Device& device,
                           const T* distances_ptr,
                           int64_t distance_query_stride,
                           int64_t num_queries,
                           int64_t num_points,
                           int64_t knn,
                           TIndex index_offset,
                           TIndex* scratch_indices_ptr,
                           int64_t scratch_query_stride,
                           TIndex* out_indices_ptr,
                           T* out_distances_ptr,
                           int64_t out_query_stride,
                           bool use_threshold = false,
                           const T* query_norms_ptr = nullptr,
                           T radius_sq = T(0),
                           T scalar_threshold = T(0)) {
    if (num_queries == 0 || num_points == 0 || knn <= 0) return;

    const T inf = std::numeric_limits<T>::max();
    const int64_t actual_knn = std::min(knn, num_points);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    if (knn <= kSYCLKnnMidKMax) {
        const int64_t k_bucket = KBucket(knn);
        DispatchSelectTopKQueriesSYCL<T, TIndex>(
                queue, distances_ptr, distance_query_stride, num_queries,
                num_points, knn, k_bucket, index_offset, out_indices_ptr,
                out_distances_ptr, out_query_stride, use_threshold,
                query_norms_ptr, radius_sq, scalar_threshold);
    } else {
        // oneDPL partial_sort fallback (P8: serial per query).
        auto policy = oneapi::dpl::execution::make_device_policy(queue);
        queue.parallel_for(
                sycl::range<2>(num_queries, num_points),
                [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                    scratch_indices_ptr[id[0] * scratch_query_stride + id[1]] =
                            static_cast<TIndex>(id[1]);
                });
        queue.wait_and_throw();

        for (int64_t qi = 0; qi < num_queries; ++qi) {
            TIndex* q_scratch = scratch_indices_ptr + qi * scratch_query_stride;
            const T* q_dist = distances_ptr + qi * distance_query_stride;
            std::partial_sort(policy, q_scratch, q_scratch + actual_knn,
                              q_scratch + num_points,
                              [q_dist](TIndex lhs, TIndex rhs) {
                                  const T ld = sycl::fmax(T(0), q_dist[lhs]);
                                  const T rd = sycl::fmax(T(0), q_dist[rhs]);
                                  if (ld < rd) return true;
                                  if (rd < ld) return false;
                                  return lhs < rhs;  // C4
                              });
        }

        queue.parallel_for(
                sycl::range<2>(num_queries, knn),
                [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                    const int64_t qi = id[0], k = id[1];
                    TIndex* qout_i = out_indices_ptr + qi * out_query_stride;
                    T* qout_d = out_distances_ptr + qi * out_query_stride;
                    if (k >= actual_knn) {
                        qout_i[k] = TIndex(-1);
                        qout_d[k] = inf;
                        return;
                    }
                    const TIndex li =
                            scratch_indices_ptr[qi * scratch_query_stride + k];
                    const T dist = sycl::fmax(
                            T(0),
                            distances_ptr[qi * distance_query_stride + li]);
                    const T thr = (use_threshold && query_norms_ptr)
                                          ? (radius_sq - query_norms_ptr[qi])
                                          : scalar_threshold;
                    if (use_threshold && dist > thr) {
                        qout_i[k] = TIndex(-1);
                        qout_d[k] = inf;
                        return;
                    }
                    qout_i[k] = index_offset + li;
                    qout_d[k] = dist;
                });
    }
}

/// Merge two sorted (ascending) per-query top-K arrays.
/// Identical to the old MergeTopKQueriesSYCL but split into the two paths
/// based on kSYCLKnnMidKMax (was fixed at 256).
template <typename T, typename TIndex>
void MergeTopKQueriesSYCL(const Device& device,
                          const T* curr_dist_ptr,
                          const TIndex* curr_idx_ptr,
                          int64_t curr_stride,
                          const T* cand_dist_ptr,
                          const TIndex* cand_idx_ptr,
                          int64_t cand_stride,
                          int64_t num_queries,
                          int64_t knn,
                          TIndex* scratch_ptr,
                          int64_t scratch_stride,
                          TIndex* out_idx_ptr,
                          T* out_dist_ptr,
                          int64_t out_stride) {
    if (num_queries == 0 || knn <= 0) return;
    const T inf = std::numeric_limits<T>::max();
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    if (knn <= kSYCLKnnMidKMax) {
        queue.parallel_for(
                sycl::range<1>(num_queries),
                [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                    const int64_t q = id[0];
                    const T* qcd = curr_dist_ptr + q * curr_stride;
                    const TIndex* qci = curr_idx_ptr + q * curr_stride;
                    const T* qad = cand_dist_ptr + q * cand_stride;
                    const TIndex* qai = cand_idx_ptr + q * cand_stride;
                    TIndex* qout_i = out_idx_ptr + q * out_stride;
                    T* qout_d = out_dist_ptr + q * out_stride;

                    int64_t ic = 0, ia = 0;
                    for (int64_t k = 0; k < knn; ++k) {
                        const TIndex ci = (ic < knn) ? qci[ic] : TIndex(-1);
                        const TIndex ai = (ia < knn) ? qai[ia] : TIndex(-1);
                        if (ci < 0 && ai < 0) {
                            qout_i[k] = TIndex(-1);
                            qout_d[k] = inf;
                            continue;
                        }
                        bool take_curr;
                        if (ci < 0) {
                            take_curr = false;
                        } else if (ai < 0) {
                            take_curr = true;
                        } else {
                            const T cd = qcd[ic], ad = qad[ia];
                            if (cd < ad)
                                take_curr = true;
                            else if (ad < cd)
                                take_curr = false;
                            else
                                take_curr = (ci < ai);  // C4
                        }
                        if (take_curr) {
                            qout_d[k] = qcd[ic];
                            qout_i[k] = ci;
                            ++ic;
                        } else {
                            qout_d[k] = qad[ia];
                            qout_i[k] = ai;
                            ++ia;
                        }
                    }
                });
    } else {
        // oneDPL merge sort fallback for large knn (P8).
        const int64_t combined = 2 * knn;
        auto policy = oneapi::dpl::execution::make_device_policy(queue);
        queue.parallel_for(sycl::range<2>(num_queries, combined),
                           [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                               scratch_ptr[id[0] * scratch_stride + id[1]] =
                                       static_cast<TIndex>(id[1]);
                           });
        queue.wait_and_throw();

        for (int64_t qi = 0; qi < num_queries; ++qi) {
            TIndex* qs = scratch_ptr + qi * scratch_stride;
            const T* qcd = curr_dist_ptr + qi * curr_stride;
            const TIndex* qci = curr_idx_ptr + qi * curr_stride;
            const T* qad = cand_dist_ptr + qi * cand_stride;
            const TIndex* qai = cand_idx_ptr + qi * cand_stride;
            std::partial_sort(
                    policy, qs, qs + knn, qs + combined,
                    [qcd, qci, qad, qai, knn](TIndex lhs, TIndex rhs) {
                        const bool lc = (lhs < knn), rc = (rhs < knn);
                        const TIndex li = lc ? qci[lhs] : qai[lhs - knn];
                        const TIndex ri = rc ? qci[rhs] : qai[rhs - knn];
                        const T ld = lc ? qcd[lhs] : qad[lhs - knn];
                        const T rd = rc ? qcd[rhs] : qad[rhs - knn];
                        if ((li >= 0) != (ri >= 0)) return li >= 0;
                        if (ld < rd) return true;
                        if (rd < ld) return false;
                        return li < ri;  // C4
                    });
        }

        queue.parallel_for(
                sycl::range<2>(num_queries, knn),
                [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                    const int64_t qi = id[0], k = id[1];
                    const TIndex src = scratch_ptr[qi * scratch_stride + k];
                    const bool is_curr = (src < knn);
                    const int64_t off = is_curr ? src : src - knn;
                    const TIndex ii =
                            is_curr ? curr_idx_ptr[qi * curr_stride + off]
                                    : cand_idx_ptr[qi * cand_stride + off];
                    const T dd =
                            is_curr ? curr_dist_ptr[qi * curr_stride + off]
                                    : cand_dist_ptr[qi * cand_stride + off];
                    TIndex* qout_i = out_idx_ptr + qi * out_stride;
                    T* qout_d = out_dist_ptr + qi * out_stride;
                    if (ii < 0) {
                        qout_i[k] = TIndex(-1);
                        qout_d[k] = inf;
                    } else {
                        qout_i[k] = ii;
                        qout_d[k] = dd;
                    }
                });
    }
}

// ─── P2 finalisation for mid-/large-K paths ───────────────────────────────

/// Add |q|² to partial distances and clamp ≥ 0 (C1).  Called once after all
/// point tiles for the SelectTopK / Merge path (mid-K and large-K).
template <typename T, typename TIndex>
void AddQueryNormsToDistancesSYCL(const Device& device,
                                  int64_t num_queries,
                                  int64_t knn,
                                  const TIndex* indices_ptr,
                                  T* distances_ptr,
                                  const T* query_norms_ptr) {
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<2>(num_queries, knn),
                       [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                           const int64_t q = id[0], k = id[1];
                           if (indices_ptr[q * knn + k] < 0) return;
                           distances_ptr[q * knn + k] =
                                   sycl::fmax(T(0), distances_ptr[q * knn + k] +
                                                            query_norms_ptr[q]);
                       });
}

// ─── Radius / hybrid helpers ──────────────────────────────────────────────

/// Count points within the adjusted per-query threshold (P2).
/// threshold_q = radius² − |q|², so we compare partial_dist directly.
template <typename T>
void CountWithinThresholdQueriesSYCL(const Device& device,
                                     const T* partial_dist_ptr,
                                     int64_t distance_stride,
                                     int64_t num_queries,
                                     int64_t num_points,
                                     const T* query_norms_ptr,
                                     T radius_sq,
                                     int64_t* counts_ptr) {
    if (num_queries == 0 || num_points == 0) return;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<1>(num_queries),
                       [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                           const int64_t q = id[0];
                           const T thresh = radius_sq - query_norms_ptr[q];
                           const T* qd = partial_dist_ptr + q * distance_stride;
                           int64_t cnt = 0;
                           for (int64_t p = 0; p < num_points; ++p) {
                               if (qd[p] <= thresh) ++cnt;
                           }
                           counts_ptr[q] += cnt;
                       });
}

/// Hybrid tile: accumulate radius counts and select top-k partial distances
/// in one pass per query (mid-k only; k ≤ kSYCLKnnMidKMax).
template <typename T, typename TIndex>
void CountAndSelectTopKQueriesSYCL(const Device& device,
                                   const T* partial_dist_ptr,
                                   int64_t distance_stride,
                                   int64_t num_queries,
                                   int64_t num_points,
                                   int64_t knn,
                                   TIndex index_offset,
                                   const T* query_norms_ptr,
                                   T radius_sq,
                                   int64_t* counts_ptr,
                                   TIndex* scratch_indices_ptr,
                                   int64_t scratch_query_stride,
                                   TIndex* out_indices_ptr,
                                   T* out_distances_ptr,
                                   int64_t out_query_stride) {
    if (num_queries == 0 || num_points == 0 || knn <= 0) return;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    if (knn <= kSYCLKnnMidKMax) {
        const int64_t k_bucket = KBucket(knn);
        DispatchCountAndSelectTopKQueriesSYCL<T, TIndex>(
                queue, partial_dist_ptr, distance_stride, num_queries,
                num_points, knn, k_bucket, index_offset, query_norms_ptr,
                radius_sq, counts_ptr, out_indices_ptr, out_distances_ptr,
                out_query_stride);
    } else {
        CountWithinThresholdQueriesSYCL<T>(
                device, partial_dist_ptr, distance_stride, num_queries,
                num_points, query_norms_ptr, radius_sq, counts_ptr);
        SelectTopKQueriesSYCL<T, TIndex>(
                device, partial_dist_ptr, distance_stride, num_queries,
                num_points, knn, index_offset, scratch_indices_ptr,
                scratch_query_stride, out_indices_ptr, out_distances_ptr,
                out_query_stride,
                /*use_threshold=*/true, query_norms_ptr, radius_sq);
    }
}

/// Gather indices (and full L2 distances) within the per-query threshold.
/// Returned distances = partial + |q|², clamped ≥ 0 (C1).
template <typename T, typename TIndex>
void GatherWithinThresholdQueriesSYCL(const Device& device,
                                      const T* partial_dist_ptr,
                                      int64_t distance_stride,
                                      int64_t num_queries,
                                      int64_t num_points,
                                      const T* query_norms_ptr,
                                      T radius_sq,
                                      TIndex index_offset,
                                      int64_t* offsets_ptr,
                                      TIndex* out_indices_ptr,
                                      T* out_distances_ptr) {
    if (num_queries == 0 || num_points == 0) return;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
                const T qnorm = query_norms_ptr[q];
                const T thresh = radius_sq - qnorm;
                const T* qd = partial_dist_ptr + q * distance_stride;
                int64_t offset = offsets_ptr[q];
                for (int64_t p = 0; p < num_points; ++p) {
                    const T partial = qd[p];
                    if (partial <= thresh) {
                        out_indices_ptr[offset] =
                                index_offset + static_cast<TIndex>(p);
                        if (out_distances_ptr != nullptr)
                            out_distances_ptr[offset] =
                                    sycl::fmax(T(0), partial + qnorm);  // C1
                        ++offset;
                    }
                }
                offsets_ptr[q] = offset;
            });
}

/// Clip hybrid neighbor counts to max_knn and zero-fill the unused tail.
template <typename T, typename TIndex>
void FinalizeHybridResultsSYCL(const Device& device,
                               const int64_t* counts_ptr,
                               int64_t num_queries,
                               int64_t max_knn,
                               TIndex* neighbors_index_ptr,
                               T* neighbors_distance_ptr,
                               TIndex* neighbors_count_ptr) {
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
                const int64_t cnt = std::min<int64_t>(counts_ptr[q], max_knn);
                neighbors_count_ptr[q] = static_cast<TIndex>(cnt);
                for (int64_t k = cnt; k < max_knn; ++k) {
                    neighbors_index_ptr[q * max_knn + k] = TIndex(-1);
                    neighbors_distance_ptr[q * max_knn + k] = T(0);
                }
            });
}

/// Per-query final distances for hybrid search need |q|² added and clamped.
/// Called after the SelectTopK/Merge pass that produced partial distances.
template <typename T, typename TIndex>
void AddQueryNormsToHybridDistancesSYCL(const Device& device,
                                        int64_t num_queries,
                                        int64_t max_knn,
                                        const TIndex* indices_ptr,
                                        T* distances_ptr,
                                        const T* query_norms_ptr) {
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<2>(num_queries, max_knn),
                       [=](sycl::id<2> id) [[intel::kernel_args_restrict]] {
                           const int64_t q = id[0], k = id[1];
                           if (indices_ptr[q * max_knn + k] < 0) return;
                           distances_ptr[q * max_knn + k] = sycl::fmax(
                                   T(0), distances_ptr[q * max_knn + k] +
                                                 query_norms_ptr[q]);
                       });
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
