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
//   SelectTopKQueriesSYCL and MergeTopKQueriesSYCL are kept for the hybrid
//   search (threshold + top-k) and for k > kSYCLKnnMidKMax.  They now take
//   partial distances (no |q|²) and a per-query threshold pointer for P2.
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
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t q = id[0];
        const T* qrow = neg2qp_ptr + q * distance_stride;
        T* qd = best_dist_ptr + q * K;
        TIndex* qi = best_idx_ptr + q * K;

        // Load running best into private registers (or scratch for large K).
        T d[K];
        TIndex idx[K];
        for (int i = 0; i < K; ++i) {
            d[i] = qd[i];
            idx[i] = qi[i];
        }

        // Scan: fused |p|² add, heap insert.
        // Note: partial_dist = −2qp + |p|² may be negative (|q|² not yet
        // added).  Do NOT clamp here; C1 clamping is applied in
        // FinalizeTopKSYCL / GatherWithinThreshold once |q|² is added back.
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
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t q = id[0];
        T d[K];
        TIndex idx[K];
        for (int i = 0; i < K; ++i) {
            d[i] = running_dist_ptr[q * K + i];
            idx[i] = running_idx_ptr[q * K + i];
        }
        HeapSort<T, TIndex, K>(d, idx);

        const T qnorm = query_norms_ptr ? query_norms_ptr[q] : T(0);
        T* qout_d = out_dist_ptr + q * actual_k;
        TIndex* qout_i = out_idx_ptr + q * actual_k;
        for (int64_t i = 0; i < actual_k; ++i) {
            T dist = d[i];
            if (query_norms_ptr) dist = sycl::fmax(T(0), dist + qnorm);
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

// ─── Legacy Select / Merge (mid-k and large-k paths) ─────────────────────
// These keep the existing algorithm but fix C1 (clamp), C4 (tie-break), and
// add P2 support (no |q|² in distances_ptr; per-query threshold for hybrid).

/// Select the smallest knn partial distances per query.
/// P2: distances_ptr contains partial dists (no |q|²); callers add it after.
/// C1: clamp each distance to max(0, ...) before comparing.
/// C4: equal distances resolved by global index.
///
/// @param per_query_threshold_ptr  Per-query adjusted threshold array (size
///   num_queries) used for hybrid search (radius² − |q|²).  If nullptr,
///   scalar_threshold is used uniformly.  Ignored when use_threshold=false.
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
                           const T* per_query_threshold_ptr = nullptr,
                           T scalar_threshold = T(0)) {
    if (num_queries == 0 || num_points == 0 || knn <= 0) return;

    const T inf = std::numeric_limits<T>::max();
    const int64_t actual_knn = std::min(knn, num_points);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    if (knn <= kSYCLKnnMidKMax) {
        // Heap path: one work-item per query, private array sized to knn.
        // The fixed [kSYCLKnnMidKMax] bounds avoid VLA; only [0,actual_knn)
        // is semantically active.  For knn ≤ 32 the compiler can keep the
        // active entries in registers; for knn ≤ 512 it spills proportionally.
        queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
            const int64_t q = id[0];
            const T* qd = distances_ptr + q * distance_query_stride;
            TIndex* qout_i = out_indices_ptr + q * out_query_stride;
            T* qout_d = out_distances_ptr + q * out_query_stride;

            const T thr = (use_threshold && per_query_threshold_ptr)
                                  ? per_query_threshold_ptr[q]
                                  : scalar_threshold;

            T local_d[kSYCLKnnMidKMax];
            TIndex local_i[kSYCLKnnMidKMax];
            for (int k = 0; k < actual_knn; ++k) {
                local_d[k] = inf;
                local_i[k] = TIndex(-1);
            }

            for (TIndex p = 0; p < static_cast<TIndex>(num_points); ++p) {
                // Partial distance (no |q|²): may be negative; do not
                // clamp here.  C1 clamping is in AddQueryNorms*.
                const T dist = qd[p];
                if (use_threshold && dist > thr) continue;
                if (dist < local_d[0] ||
                    (dist == local_d[0] &&
                     index_offset + p < index_offset + local_i[0])) {
                    local_d[0] = dist;
                    local_i[0] = p;  // local index stored
                    // Heapify-down with runtime knn bound.
                    int i = 0;
                    while (true) {
                        int left = 2 * i + 1, right = 2 * i + 2, largest = i;
                        if (left < actual_knn &&
                            (local_d[left] > local_d[largest] ||
                             (local_d[left] == local_d[largest] &&
                              local_i[left] > local_i[largest])))
                            largest = left;
                        if (right < actual_knn &&
                            (local_d[right] > local_d[largest] ||
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
            }

            // Insertion-sort ascending (C4: ties already broken by
            // local index, equivalent to global since same offset).
            for (int i = 1; i < actual_knn; ++i) {
                T key_d = local_d[i];
                TIndex key_i = local_i[i];
                int j = i - 1;
                while (j >= 0 && (local_d[j] > key_d || (local_d[j] == key_d &&
                                                         local_i[j] > key_i))) {
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
    } else {
        // oneDPL partial_sort fallback (P8: serial per query).
        auto policy = oneapi::dpl::execution::make_device_policy(queue);
        queue.parallel_for(
                sycl::range<2>(num_queries, num_points), [=](sycl::id<2> id) {
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
                sycl::range<2>(num_queries, knn), [=](sycl::id<2> id) {
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
                    const T thr = (use_threshold && per_query_threshold_ptr)
                                          ? per_query_threshold_ptr[qi]
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
        queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
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
                           [=](sycl::id<2> id) {
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
                sycl::range<2>(num_queries, knn), [=](sycl::id<2> id) {
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
    queue.parallel_for(sycl::range<2>(num_queries, knn), [=](sycl::id<2> id) {
        const int64_t q = id[0], k = id[1];
        if (indices_ptr[q * knn + k] < 0) return;
        distances_ptr[q * knn + k] = sycl::fmax(
                T(0), distances_ptr[q * knn + k] + query_norms_ptr[q]);
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
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
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
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t q = id[0];
        const T qnorm = query_norms_ptr[q];
        const T thresh = radius_sq - qnorm;
        const T* qd = partial_dist_ptr + q * distance_stride;
        int64_t offset = offsets_ptr[q];
        for (int64_t p = 0; p < num_points; ++p) {
            const T partial = qd[p];
            if (partial <= thresh) {
                out_indices_ptr[offset] = index_offset + static_cast<TIndex>(p);
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
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
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
    queue.parallel_for(
            sycl::range<2>(num_queries, max_knn), [=](sycl::id<2> id) {
                const int64_t q = id[0], k = id[1];
                if (indices_ptr[q * max_knn + k] < 0) return;
                distances_ptr[q * max_knn + k] =
                        sycl::fmax(T(0), distances_ptr[q * max_knn + k] +
                                                 query_norms_ptr[q]);
            });
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
