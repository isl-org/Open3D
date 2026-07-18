// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file KnnSearchSYCLImpl.h
/// \brief Device-side SYCL kernels for KNN, fixed-radius, and hybrid search.
///
/// Included only by KnnSearchOpsSYCL.cpp; not part of the public API.
///
/// The driver uses oneMKL AddMM to compute the −2*q*p distance tile for each
/// (query-tile × point-tile) pair.
///
/// Fused per-tile path (small k ≤ kSYCLKnnMidKMax):
/// - \ref UpdateTopKFromTile maintains a running max-heap per query from AddMM
///   tiles; \ref FinalizeTopK heap-sorts and adds |q|² (P2, C1).
/// - For K ≤ kSYCLKnnSmallKMax (32), heap arrays are GRF-resident; larger
///   buckets spill to per-work-item scratch.
///
/// Legacy path (mid/large k):
/// - \ref SelectTopKQueries, \ref MergeTopKQueries (used by the KNN
///   large-k AddMM path only; fixed-radius/hybrid now use the uniform-grid
///   kernels in FixedRadiusSearchSYCLImpl.h instead).
///
/// Direct-distance path (low D, low k):
/// - \ref KnnDirect / \ref DispatchKnnDirect bypass AddMM; see section below.
///
/// Conventions:
/// - **P2**: partial_dist = −2qp + |p|²; |q|² added once in finalize.
/// - **C1**: clamp partial_dist ≥ 0.
/// - **C4**: tie-break by smaller global point index.
/// - **C5**: actual_k = min(knn, num_points) handled by caller.
///
/// **P8**: For k > kSYCLKnnMidKMax, merge uses per-query serial partial_sort
/// (oneDPL has no segmented top-k).

#pragma once

#include <algorithm>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

/// \addtogroup nns_sycl_knn Tile sizing (AddMM path)
/// @{

/// Compute tile dimensions that bound the −2*q*p tile to \p tile_bytes.
/// tile_queries is capped at \p max_tile_queries (a good oneMKL GEMM row-tile
/// width; 128 by default, see
/// benchmarks/core/NearestNeighborSearchSYCLAddMMTuning.cpp for the sweep that
/// validated this value). tile_points is rounded down to a multiple of \p
/// tile_points_alignment (128 by default) once it exceeds that alignment,
/// keeping the column tile GEMM/cache-line friendly.
inline void ChooseTileSize(int64_t num_queries,
                           int64_t num_points,
                           int64_t element_size,
                           int64_t tile_bytes,
                           int64_t& tile_queries,
                           int64_t& tile_points,
                           int64_t max_tile_queries = 128,
                           int64_t tile_points_alignment = 128) {
    tile_queries = std::min<int64_t>(num_queries, max_tile_queries);
    tile_queries = std::max<int64_t>(tile_queries, 1);
    tile_points = std::max<int64_t>(tile_bytes / (tile_queries * element_size),
                                    int64_t(256));
    if (tile_points > tile_points_alignment) {
        tile_points =
                (tile_points / tile_points_alignment) * tile_points_alignment;
    }
    tile_points = std::min<int64_t>(tile_points, num_points);
    tile_points = std::max<int64_t>(tile_points, 1);
}

/// @}

/// \addtogroup nns_sycl_knn Compile-time heap helpers
/// @{
/// C4 tie-break: smaller global index wins on equal distance.

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

/// @}

/// \addtogroup nns_sycl_knn Fused per-tile top-K (small/mid k)
/// @{

/// Update the per-query running top-K heap using one point-column tile.
///
/// One work-item per query; K is the compile-time heap capacity (= dispatch
/// bucket ≥ actual knn).
///
///  K ≤ kSYCLKnnSmallKMax (32): d[K]/idx[K] live in GRF – no scratch spill.
///  K ≤ kSYCLKnnMidKMax  (512): spill to scratch, proportional to K.
///
/// Fuses three passes into one: Add_(point_norms) + SelectTopKQueries +
/// MergeTopKQueries.
///
/// P2: |q|² NOT added; callers add it once in FinalizeTopK.
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
///                        the hybrid SelectTopKQueries variant.
template <typename T, typename TIndex, int K>
void UpdateTopKFromTile(sycl::queue& queue,
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
                // FinalizeTopK / GatherWithinThresholdQueries once |q|² is
                // added back.
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
void FinalizeTopK(sycl::queue& queue,
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

/// @}

/// \addtogroup nns_sycl_knn K-bucket dispatch (fused path)
/// @{
/// Dispatch buckets: 1, 2, 4, …, 512 (round knn up with \ref KBucket first).

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

/// Instantiate \ref UpdateTopKFromTile for the given \p k_bucket.
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
    UpdateTopKFromTile<T, TIndex, Kval>(                                      \
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

/// Instantiate \ref FinalizeTopK for the given \p k_bucket.
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
#define CALL_FINALIZE(Kval)                                                   \
    FinalizeTopK<T, TIndex, Kval>(queue, num_queries, running_dist_ptr,       \
                                  running_idx_ptr, out_dist_ptr, out_idx_ptr, \
                                  actual_k, query_norms_ptr)
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

/// @}

/// \addtogroup nns_sycl_knn Direct-distance KNN (no AddMM)
/// @{
///
/// Cooperative SLM-tiled brute-force KNN for low dimension and low k
/// (D ≤ \ref kKnnDirectMaxDim, K ≤ kSYCLKnnSmallKMax). Accumulates |p−q|²
/// directly (no |q|²−2qp+|p|² cancellation); centering and P2 deferral are
/// not required on this path.
///
/// One sub-group of width SG handles one query; \p subgroups_per_wg queries
/// share a work-group and cooperatively cache each point tile in SLM.
/// Double-buffered tiles hide load latency. Per-lane top-K arrays merge via
/// sub-group shuffle (\c select_from_group); lane 0 writes the result.
///
/// Tuned defaults for dim=3 on Intel Xe: \ref kKnnDirectSubgroupsPerWG and
/// \ref kKnnDirectTilePoints (see NearestNeighborSearchSYCLTuning.cpp).

/// Default sub-group width for the direct KNN kernel (float path).
constexpr int64_t kKnnDirectSubgroupSize = 16;
/// Default sub-groups per work-group (512 work-items at SG=16).
constexpr int64_t kKnnDirectSubgroupsPerWG = 32;
/// Default point tile size for SLM staging.
constexpr int64_t kKnnDirectTilePoints = 2048;
/// Maximum point dimension compiled for \ref DispatchKnnDirect.
constexpr int64_t kKnnDirectMaxDim = 8;

/// Named kernel tag for \ref KnnDirect (SYCL kernel naming).
template <typename T, typename TIndex, int NDIM, int K, int SG>
class KnnDirectKernel;

/// Launch direct-distance KNN for fixed compile-time NDIM, K, and SG.
template <typename T, typename TIndex, int NDIM, int K, int SG>
void KnnDirect(sycl::queue& queue,
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
        h.parallel_for<KnnDirectKernel<T, TIndex, NDIM, K, SG>>(
                sycl::nd_range<1>(sycl::range<1>(global_size),
                                  sycl::range<1>(wg_size)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(
                        SG)]] [[intel::kernel_args_restrict]] {
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
                                        points_ptr[(cur_start + p) * NDIM + dd];
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
                                        points_ptr[(nxt_start + p) * NDIM + dd];
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
                            od[i] = sycl::select_from_group(sg, d[i], partner);
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

/// K-bucket dispatch for a fixed compile-time sub-group width \p SG. Kept
/// separate from DispatchKnnDirectK so the same K-bucket switch can be
/// instantiated at two different SG widths (see DispatchKnnDirectK) without
/// duplicating the bucket logic.
template <typename T, typename TIndex, int NDIM, int SG>
void DispatchKnnDirectKForSG(sycl::queue& queue,
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
#define CALL_DIRECT(Kval)                                                      \
    KnnDirect<T, TIndex, NDIM, Kval, SG>(                                      \
            queue, points_ptr, queries_ptr, num_points, num_queries, actual_k, \
            out_dist_ptr, out_idx_ptr, subgroups_per_wg, tile_points)
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

/// Choose sub-group width for direct KNN: float uses 16; double uses 8 when
/// supported, else 16 (see file comment in implementation).
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
    if constexpr (std::is_same_v<T, double>) {
        const auto sg_sizes =
                queue.get_device()
                        .get_info<sycl::info::device::sub_group_sizes>();
        const bool supports_subgroup_8 =
                std::find(sg_sizes.begin(), sg_sizes.end(), size_t(8)) !=
                sg_sizes.end();
        if (supports_subgroup_8) {
            DispatchKnnDirectKForSG<T, TIndex, NDIM, 8>(
                    queue, points_ptr, queries_ptr, num_points, num_queries,
                    actual_k, out_dist_ptr, out_idx_ptr, subgroups_per_wg,
                    tile_points);
        } else {
            DispatchKnnDirectKForSG<T, TIndex, NDIM, 16>(
                    queue, points_ptr, queries_ptr, num_points, num_queries,
                    actual_k, out_dist_ptr, out_idx_ptr, subgroups_per_wg,
                    tile_points);
        }
    } else {
        DispatchKnnDirectKForSG<T, TIndex, NDIM, 16>(
                queue, points_ptr, queries_ptr, num_points, num_queries,
                actual_k, out_dist_ptr, out_idx_ptr, subgroups_per_wg,
                tile_points);
    }
}

/// Dispatch the direct-distance KNN path by compile-time dimension (1..8)
/// and K-bucket (≤ 32). Writes directly into the caller-allocated output
/// buffers; no public API / build plumbing changes are required. The
/// double-precision sub-group width (8 vs 16) is chosen at runtime inside
/// DispatchKnnDirectK by querying the device; float always uses width 16.
template <typename T, typename TIndex>
void DispatchKnnDirect(sycl::queue& queue,
                       const T* points_ptr,
                       const T* queries_ptr,
                       int64_t dim,
                       int64_t num_points,
                       int64_t num_queries,
                       int64_t actual_k,
                       T* out_dist_ptr,
                       TIndex* out_idx_ptr,
                       int64_t subgroups_per_wg = kKnnDirectSubgroupsPerWG,
                       int64_t tile_points = kKnnDirectTilePoints) {
    // kKnnDirectTilePoints is tuned for the common case (dim ≤ 3, see
    // benchmarks/core/NearestNeighborSearchSYCLTuning.cpp), where the
    // resulting per-work-group SLM usage (2 * tile_points * dim * sizeof(T))
    // is well inside typical device budgets. For larger `dim` (up to
    // kKnnDirectMaxDim) or double precision, that same tile_points could
    // exceed the device's actual local memory size, so clamp it down here
    // using the real device limit (queried once, cheap) rather than baking a
    // dim/dtype-specific constant into the caller.
    {
        const size_t local_mem_bytes =
                queue.get_device()
                        .get_info<sycl::info::device::local_mem_size>();
        // Leave 10% headroom for other local allocations / runtime overhead.
        const int64_t max_tile_points_by_slm = static_cast<int64_t>(
                (local_mem_bytes * 9 / 10) / (2 * dim * sizeof(T)));
        tile_points = std::min(tile_points,
                               std::max<int64_t>(max_tile_points_by_slm, 1));
    }
#define CALL_DIM(NDIMVAL)                                                      \
    DispatchKnnDirectK<T, TIndex, NDIMVAL>(                                    \
            queue, points_ptr, queries_ptr, num_points, num_queries, actual_k, \
            out_dist_ptr, out_idx_ptr, subgroups_per_wg, tile_points)
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
            utility::LogError("DispatchKnnDirect only supports dim 1 to {}.",
                              kKnnDirectMaxDim);
    }
#undef CALL_DIM
}

/// True if (dim, knn) qualifies for the direct-distance SYCL KNN path.
inline bool UseKnnDirect(int64_t dim, int64_t knn) {
    return dim >= 1 && dim <= kKnnDirectMaxDim && knn <= kSYCLKnnSmallKMax;
}

/// @}

/// \addtogroup nns_sycl_knn Legacy select/merge (mid and large k)
/// @{
/// Mid/large-k paths apply C1 clamp, C4 tie-break, and P2 (no |q|² in tiles).

namespace {

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
void SelectTopKQueriesHeap(sycl::queue& queue,
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

}  // namespace

/// K-bucket dispatch to \ref SelectTopKQueriesHeap (file-local, anonymous
/// namespace).
template <typename T, typename TIndex>
void DispatchSelectTopKQueries(sycl::queue& queue,
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
    SelectTopKQueriesHeap<T, TIndex, Kval>(                                    \
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

/// Select the smallest knn partial distances per query.
/// P2: distances_ptr contains partial dists (no |q|²); callers add it after.
/// C1: clamp each distance to max(0, ...) before comparing.
/// C4: equal distances resolved by global index.
///
/// @param query_norms_ptr  Per-query |q|² (size num_queries). When
///   use_threshold is true and this is non-null, threshold is
///   radius_sq − query_norms_ptr[q] (P2 hybrid). Otherwise scalar_threshold.
template <typename T, typename TIndex>
void SelectTopKQueries(const Device& device,
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
        DispatchSelectTopKQueries<T, TIndex>(
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
            // C1's clamp is for the *reported* distance only (applied below,
            // when writing qout_d). Clamping here in the comparator would
            // tie together every point whose true partial distance is
            // slightly negative from P2 cancellation (common for
            // widely-spread float32 data), corrupting the selected/sorted
            // *set* of neighbors -- not just their reported distance value.
            // Comparing the raw (unclamped) values preserves the true
            // relative order even when cancellation makes some values
            // slightly negative.
            std::partial_sort(policy, q_scratch, q_scratch + actual_knn,
                              q_scratch + num_points,
                              [q_dist](TIndex lhs, TIndex rhs) {
                                  const T ld = q_dist[lhs];
                                  const T rd = q_dist[rhs];
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
                    // P2/C1: this is the *partial* distance (−2qp+|p|², |q|²
                    // not yet added by the caller). Do not clamp ≥ 0 here --
                    // the partial value can be legitimately very negative
                    // (missing +|q|²), especially for widely-spread float32
                    // data; clamping it here (before |q|² is added) ties
                    // together every such point at exactly 0, corrupting the
                    // reported distance for many neighbors at once. The
                    // final clamp is applied once |q|² has been added (see
                    // AddQueryNormsToDistances / FinalizeTopK's C1).
                    const T dist =
                            distances_ptr[qi * distance_query_stride + li];
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

/// Merge two sorted (ascending) per-query top-K arrays. Uses a linear merge
/// for knn ≤ kSYCLKnnMidKMax, else an oneDPL partial_sort fallback (P8).
template <typename T, typename TIndex>
void MergeTopKQueries(const Device& device,
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

/// @}

/// \addtogroup nns_sycl_knn P2 finalization (KNN large-k path)
/// @{

/// Add |q|² to partial distances and clamp ≥ 0 (C1).  Called once after all
/// point tiles for the SelectTopK / Merge path (mid-K and large-K).
template <typename T, typename TIndex>
void AddQueryNormsToDistances(const Device& device,
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

/// @}

}  // namespace nns
}  // namespace core
}  // namespace open3d
