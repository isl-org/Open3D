// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file FixedRadiusSearchSYCLImpl.h
/// \brief SYCL device kernels: uniform-grid fixed-radius and hybrid neighbor
/// search.
///
/// Included only from \ref KnnSearchOpsSYCL.cpp (not public API). Algorithm
/// matches CUDA `FixedRadiusSearchImpl.cuh`; shared geometry in \ref
/// NeighborSearchCommon.h
/// (`SpatialHash`, `ComputeVoxelIndex`). See `nns/SYCL_DESIGN.md` for overview.
///
/// \section FrsSyclGrid Grid build (\ref BuildSpatialHashTableSYCL)
///
/// 1. Bucket dataset into a uniform spatial-hash grid with **cell size `2 *
/// radius`**
///    (any neighbor within `radius` lies in the query cell or one of seven
///    corner-adjacent cells — **8 bins** visited per query, deduplicated).
/// 2. **Count** points per cell (per batch).
/// 3. **Inclusive scan** of counts → CSR offsets (`hash_table_cell_splits`;
///    oneDPL; CUDA uses CUB `DeviceScan::InclusiveSum`).
/// 4. **Scatter** point indices into cell ranges (`hash_table_index`).
///
/// Runs on **SYCL CPU and GPU**. Host driver uses one in-order queue with
/// minimal sync between batch loops.
///
/// \section FrsSyclQuery Query kernels
///
/// | Mode | Kernels | Passes |
/// |------|---------|--------|
/// | Fixed-radius | \ref CountNeighborsSYCL, \ref WriteNeighborsSYCL | Count →
/// scan on host → allocate → gather | | Hybrid | \ref WriteNeighborsHybridSYCL
/// | Single pass: running top-`max_knn` + in-radius count, then bubble sort |
/// | Optional sort | \ref SortNeighborsByDistanceSYCL (`sort=true`) | Segmented
/// sort per query segment |
///
/// **Metrics:** L1, L2, Linf (same as CUDA). L2 compares **squared** distance
/// to `radius²`; L1/Linf compare metric distance to `radius`.
///
/// **Sort (`sort=true`):** oneDPL `sort_by_key` — `float` uses packed radix
/// key; `double` uses struct key + comparator (needs full 64-bit distance).
/// Ties are
/// **not** secondarily ordered by neighbor index (CUDA parity).
///
/// \section FrsSyclVsCuda Primitives (CUDA → SYCL)
///
/// | Role | CUDA | SYCL (this file) |
/// |------|------|------------------|
/// | Prefix sum | CUB inclusive scan | oneDPL inclusive scan |
/// | Segmented sort | CUB segmented radix sort | oneDPL `sort_by_key` |
/// | Query parallelism | 1 thread / query | 1 work-item / query
/// (`parallel_for`) |

#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "open3d/core/ParallelFor.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace core {
namespace nns {

namespace {

/// Squared L2 distance between two 3D points.
template <class T>
inline T SquaredDistance(const utility::MiniVec<T, 3>& a,
                         const utility::MiniVec<T, 3>& b) {
    utility::MiniVec<T, 3> d = a - b;
    return d.dot(d);
}

/// Distance under \p metric (L2 returns squared distance, matching CUDA).
template <class T>
inline T DistanceForMetric(Metric metric,
                           const utility::MiniVec<T, 3>& p,
                           const utility::MiniVec<T, 3>& q) {
    if (metric == Linf) {
        utility::MiniVec<T, 3> d = (p - q).abs();
        return sycl::fmax(d[0], sycl::fmax(d[1], d[2]));
    }
    if (metric == L1) {
        utility::MiniVec<T, 3> d = (p - q).abs();
        return d[0] + d[1] + d[2];
    }
    return SquaredDistance(p, q);
}

/// True if \p p is a neighbor of \p q under \p metric and \p threshold.
template <class T>
inline bool IsNeighbor(Metric metric,
                       const utility::MiniVec<T, 3>& p,
                       const utility::MiniVec<T, 3>& q,
                       T threshold) {
    return DistanceForMetric(metric, p, q) <= threshold;
}

template <class T>
inline bool IsNeighbor(Metric metric,
                       const utility::MiniVec<T, 3>& p,
                       const utility::MiniVec<T, 3>& q,
                       T threshold,
                       T* out_dist) {
    const T dist = DistanceForMetric(metric, p, q);
    if (out_dist) {
        *out_dist = dist;
    }
    return dist <= threshold;
}

/// Collects the (up to 8, deduplicated) hash bins that may contain a
/// neighbor of \p pos: the bin containing \p pos itself, plus the bin
/// reached by stepping +-radius along each axis (the corner-adjacent bins).
/// Because the grid cell size is 2*radius, these 8 bins are guaranteed to
/// cover every point within \p radius of \p pos. Unused slots are set to -1.
template <class T>
inline void CollectBinsToVisit(const utility::MiniVec<T, 3>& pos,
                               T inv_voxel_size,
                               T radius,
                               uint32_t hash_table_size,
                               int bins_to_visit[8]) {
    auto voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);
    int hash = static_cast<int>(SpatialHash(voxel_index) % hash_table_size);
    bins_to_visit[0] = hash;
    for (int i = 1; i < 8; ++i) bins_to_visit[i] = -1;

    for (int dz = -1; dz <= 1; dz += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dx = -1; dx <= 1; dx += 2) {
                utility::MiniVec<T, 3> p =
                        pos +
                        radius * utility::MiniVec<T, 3>(T(dx), T(dy), T(dz));
                auto vidx = ComputeVoxelIndex(p, inv_voxel_size);
                int h = static_cast<int>(SpatialHash(vidx) % hash_table_size);
                for (int i = 0; i < 8; ++i) {
                    if (bins_to_visit[i] == h) {
                        break;
                    } else if (bins_to_visit[i] == -1) {
                        bins_to_visit[i] = h;
                        break;
                    }
                }
            }
        }
    }
}

template <class T>
struct SortKey {
    int64_t query_id;
    T dist;
};

}  // namespace

/// Builds a uniform spatial-hash grid ("cell list") for a fixed-radius
/// search: count points per cell -> device inclusive scan -> scatter point
/// indices into their cell's slot range. Mirrors BuildSpatialHashTableCUDA.
///
/// Raw-pointer variant: takes a SYCL queue directly plus host-accessible
/// batch arrays, so both the Open3D Tensor API and the PyTorch XPU dispatch
/// can share the same kernel implementation without tensor conversion.
///
/// \p host_points_row_splits and \p host_hash_table_splits are CPU arrays.
/// \p cell_splits_ptr and \p index_ptr are device (USM or XPU) pointers.
///
/// Non-blocking: returns the event of the last enqueued command instead of
/// waiting host-side. Note that Pass 3's `sycl::buffer` scratch (see the
/// comment above it) still forces a host block at its own scope exit, so
/// this function is not fully async yet; the event is still returned for API
/// consistency and because Pass 1/2 no longer add their own redundant waits.
template <class T>
sycl::event BuildSpatialHashTableSYCLRaw(sycl::queue& queue,
                                         const T* points_ptr,
                                         T inv_voxel_size,
                                         int batch_size,
                                         const int64_t* host_points_row_splits,
                                         const uint32_t* host_hash_table_splits,
                                         uint32_t* cell_splits_ptr,
                                         size_t cell_splits_size,
                                         uint32_t* index_ptr) {
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    sycl::event memset_event = queue.memset(
            cell_splits_ptr, 0, cell_splits_size * sizeof(uint32_t));

    // Pass 1: count points per cell (into cell_splits_i[hash + 1]), so the
    // scan in Pass 2 turns this into CSR start offsets. Collect each batch's
    // completion event instead of waiting -- the queue is in-order so these
    // already run after memset_event, but Pass 2's oneDPL scan uses its own
    // internal submission path, so it needs these events passed explicitly
    // (a device_policy does not otherwise inherit pending queue work).
    std::vector<sycl::event> count_events;
    for (int b = 0; b < batch_size; ++b) {
        const int64_t point_begin = host_points_row_splits[b];
        const int64_t point_end = host_points_row_splits[b + 1];
        const int64_t num_points_i = point_end - point_begin;
        if (num_points_i == 0) continue;
        const uint32_t first_cell_idx = host_hash_table_splits[b];
        const uint32_t hash_table_size =
                host_hash_table_splits[b + 1] - first_cell_idx;
        uint32_t* cell_splits_i = cell_splits_ptr + first_cell_idx;

        {
            const size_t wg =
                    core::sy::PreferredWorkGroupSize(queue.get_device());
            const size_t global_size =
                    ((static_cast<size_t>(num_points_i) + wg - 1) / wg) * wg;
            count_events.push_back(queue.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(global_size),
                                      sycl::range<1>(wg)),
                    memset_event,
                    [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                        const int64_t li = it.get_global_id(0);
                        if (li >= num_points_i) return;
                        const int64_t i = point_begin + li;
                        utility::MiniVec<T, 3> pos(points_ptr + 3 * i);
                        auto voxel_index =
                                ComputeVoxelIndex(pos, inv_voxel_size);
                        const size_t hash =
                                SpatialHash(voxel_index) % hash_table_size;
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                                cnt(cell_splits_i[hash + 1]);
                        cnt.fetch_add(1);
                    }));
        }
    }

    // Pass 2: turn per-cell counts into CSR start offsets with a *single*
    // scan over the whole (all-batches-concatenated) array -- mirrors CUDA,
    // which calls cub::DeviceScan::InclusiveSum once over the full
    // count_tmp/hash_table_cell_splits buffer rather than once per batch.
    // This is valid (not just faster) because per-batch segments are laid
    // out back-to-back and each segment's raw count at its own first slot is
    // always 0 (see the "hash + 1" count in Pass 1): the running sum thus
    // carries the *previous* batches' total point counts straight into the
    // next batch's segment, which is exactly the absolute base offset that
    // batch needs into the shared hash_table_index array. A segmented scan
    // would compute the same per-batch-relative values and gain nothing.
    //
    // Uses the async/event variant (not std::inclusive_scan): the device
    // policy wraps the queue but does not inherit its pending Pass-1 work on
    // an out-of-order queue, so without passing count_events explicitly, the
    // scan could race Pass 1's atomics. Open3D's own queues are in-order
    // (SYCLContext.cpp), but this kernel is also reachable from PyTorch's
    // XPU queue via BuildSpatialHashTableOpKernelSYCL.cpp, which is NOT
    // guaranteed in-order -- so the explicit deps are load-bearing there.
    // inclusive_scan_async's dependency pack is variadic (individual
    // sycl::event args, not a container), but count_events' size is a
    // runtime batch_size -- fold it into one barrier event first so exactly
    // one dependency is passed regardless of batch count.
    sycl::event count_barrier =
            count_events.empty()
                    ? sycl::event()
                    : queue.ext_oneapi_submit_barrier(count_events);
    sycl::event scan_event;
    if (cell_splits_size > 0) {
        scan_event = oneapi::dpl::experimental::inclusive_scan_async(
                             policy, cell_splits_ptr,
                             cell_splits_ptr + cell_splits_size,
                             cell_splits_ptr, count_barrier)
                             .event();
    } else {
        scan_event = count_barrier;
    }

    // Pass 3: scatter point indices into their cell's slot range. One reused
    // slot-counter buffer (memset per batch on this in-order queue) avoids
    // per-batch allocation; uses USM so PyTorch and Tensor callers share this
    // path without Open3D Tensor scratch.
    uint32_t max_hash_table_size = 0;
    for (int b = 0; b < batch_size; ++b) {
        max_hash_table_size = std::max<uint32_t>(
                max_hash_table_size,
                host_hash_table_splits[b + 1] - host_hash_table_splits[b]);
    }
    // Kernel-private scratch (never touched by PyTorch/oneDPL/sycl-tla), so
    // it is backed by sycl::buffer rather than malloc_device/free. This also
    // fixes a real bug: the previous USM version called sycl::free() on this
    // pointer immediately after the batch loop but *before* the
    // queue.wait_and_throw() below, while the Pass-3 scatter kernels reading
    // it could still be in flight -- sycl::free() does not wait for
    // in-flight commands, so freeing this memory while a kernel might still
    // be using it is undefined behavior (a use-after-free race). The buffer
    // destructor blocks on its last reader before releasing memory, so no
    // such race is possible.
    std::vector<sycl::event> scatter_events;
    if (max_hash_table_size > 0) {
        sycl::buffer<uint32_t, 1> slot_counts_buf{
                sycl::range<1>(max_hash_table_size)};
        for (int b = 0; b < batch_size; ++b) {
            const int64_t point_begin = host_points_row_splits[b];
            const int64_t point_end = host_points_row_splits[b + 1];
            const int64_t num_points_i = point_end - point_begin;
            if (num_points_i == 0) continue;
            const uint32_t first_cell_idx = host_hash_table_splits[b];
            const uint32_t hash_table_size =
                    host_hash_table_splits[b + 1] - first_cell_idx;
            const uint32_t* cell_splits_i = cell_splits_ptr + first_cell_idx;

            queue.submit([&](sycl::handler& cgh) {
                sycl::accessor slot_counts_acc(slot_counts_buf, cgh,
                                               sycl::range<1>(hash_table_size),
                                               sycl::write_only, sycl::no_init);
                cgh.fill(slot_counts_acc, 0u);
            });

            scatter_events.push_back(queue.submit([&](sycl::handler& cgh) {
                // Explicit dep: cell_splits_ptr (read below) is raw USM, not
                // buffer-tracked, so the SYCL runtime cannot infer this
                // ordering from data dependencies alone -- required for
                // correctness on out-of-order queues (see scan_event above).
                cgh.depends_on(scan_event);
                sycl::accessor slot_counts_acc(slot_counts_buf, cgh,
                                               sycl::range<1>(hash_table_size),
                                               sycl::read_write);
                const size_t wg =
                        core::sy::PreferredWorkGroupSize(queue.get_device());
                const size_t global_size =
                        ((static_cast<size_t>(num_points_i) + wg - 1) / wg) *
                        wg;
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(global_size),
                                          sycl::range<1>(wg)),
                        [=](sycl::nd_item<1>
                                    it) [[intel::kernel_args_restrict]] {
                            const int64_t li = it.get_global_id(0);
                            if (li >= num_points_i) return;
                            const int64_t i = point_begin + li;
                            utility::MiniVec<T, 3> pos(points_ptr + 3 * i);
                            auto voxel_index =
                                    ComputeVoxelIndex(pos, inv_voxel_size);
                            const size_t hash =
                                    SpatialHash(voxel_index) % hash_table_size;
                            sycl::atomic_ref<uint32_t,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device>
                                    cnt(slot_counts_acc[hash]);
                            const uint32_t slot = cnt.fetch_add(1);
                            index_ptr[cell_splits_i[hash] + slot] =
                                    static_cast<uint32_t>(i);
                        });
            }));
        }
        // slot_counts_buf goes out of scope here: its destructor blocks on
        // the last accessor (the scatter kernel above) before releasing the
        // buffer's backing memory, exactly like the free-before-wait fix
        // documented above. This is the one remaining host block in this
        // function (Phase 2.3's buffer conversion traded a would-be
        // use-after-free for a buffer-dtor wait); everything else is async.
    }
    return scatter_events.empty()
                   ? scan_event
                   : queue.ext_oneapi_submit_barrier(scatter_events);
}

/// Builds the uniform-grid spatial hash table. \p points_row_splits and
/// \p hash_table_splits are host (CPU) tensors; \p hash_table_index and
/// \p hash_table_cell_splits are device output tensors already sized by
/// FixedRadiusIndex::SetTensorData. Delegates to BuildSpatialHashTableSYCLRaw.
template <class T>
void BuildSpatialHashTableSYCL(const Tensor& points,
                               double radius,
                               const Tensor& points_row_splits,
                               const Tensor& hash_table_splits,
                               Tensor& hash_table_index,
                               Tensor& hash_table_cell_splits) {
    const Device device = points.GetDevice();
    sycl::queue queue = sy::GetQueue(device);

    const T inv_voxel_size = T(1) / T(2 * radius);
    const int batch_size = static_cast<int>(points_row_splits.GetShape(0)) - 1;

    // points_row_splits / hash_table_splits are CPU tensors; extract raw host
    // arrays so BuildSpatialHashTableSYCLRaw can share its implementation
    // with the PyTorch XPU dispatch (which never has an Open3D Tensor).
    const int64_t* host_points_row_splits =
            points_row_splits.GetDataPtr<int64_t>();
    const uint32_t* host_hash_table_splits =
            hash_table_splits.GetDataPtr<uint32_t>();

    // Tensor-API boundary: wait once here rather than propagating the event,
    // since callers (FixedRadiusIndex::SetTensorData) expect the grid to be
    // fully built on return, matching the Tensor API's synchronous contract.
    BuildSpatialHashTableSYCLRaw<T>(
            queue, points.GetDataPtr<T>(), inv_voxel_size, batch_size,
            host_points_row_splits, host_hash_table_splits,
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            static_cast<size_t>(hash_table_cell_splits.NumElements()),
            hash_table_index.GetDataPtr<uint32_t>())
            .wait_and_throw();
}

/// Counts, for every query, how many dataset points lie within \p radius,
/// using the grid built by \ref BuildSpatialHashTableSYCL. Mirrors
/// CountNeighborsKernel (CUDA).
template <class T>
void CountNeighborsSYCL(sycl::queue& queue,
                        uint32_t* neighbors_count_ptr,
                        const uint32_t* const point_index_table,
                        const uint32_t* const hash_table_cell_splits,
                        uint32_t hash_table_size,
                        const T* const query_points,
                        int64_t num_queries,
                        const T* const points,
                        T inv_voxel_size,
                        T radius,
                        Metric metric,
                        bool ignore_query_point,
                        T threshold) {
    if (num_queries == 0) return;
    const size_t wg = core::sy::PreferredWorkGroupSize(queue.get_device());
    const size_t global_size =
            ((static_cast<size_t>(num_queries) + wg - 1) / wg) * wg;
    queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                const int64_t q = it.get_global_id(0);
                if (q >= num_queries) return;
                utility::MiniVec<T, 3> query_pos(query_points + 3 * q);
                int bins[8];
                CollectBinsToVisit(query_pos, inv_voxel_size, radius,
                                   hash_table_size, bins);
                uint32_t count = 0;
                for (int bi = 0; bi < 8; ++bi) {
                    const int bin = bins[bi];
                    if (bin < 0) break;
                    const uint32_t begin = hash_table_cell_splits[bin];
                    const uint32_t end = hash_table_cell_splits[bin + 1];
                    for (uint32_t j = begin; j < end; ++j) {
                        const uint32_t idx = point_index_table[j];
                        utility::MiniVec<T, 3> p(points + 3 * idx);
                        if (ignore_query_point && (query_pos == p).all()) {
                            continue;
                        }
                        if (IsNeighbor(metric, p, query_pos, threshold))
                            ++count;
                    }
                }
                neighbors_count_ptr[q] = count;
            });
}

/// Writes neighbor indices (and optionally distances) for every query into
/// the offsets given by \p neighbors_row_splits (an exclusive prefix sum
/// over per-query counts). Mirrors WriteNeighborsIndicesAndDistancesKernel
/// (CUDA). Output is unsorted within each query's segment; use
/// \ref SortNeighborsByDistanceSYCL afterward if `sort=true` was requested.
template <class T, class TIndex>
void WriteNeighborsSYCL(sycl::queue& queue,
                        TIndex* indices,
                        T* distances,
                        const int64_t* const neighbors_row_splits,
                        const uint32_t* const point_index_table,
                        const uint32_t* const hash_table_cell_splits,
                        uint32_t hash_table_size,
                        const T* const query_points,
                        int64_t num_queries,
                        const T* const points,
                        T inv_voxel_size,
                        T radius,
                        Metric metric,
                        bool ignore_query_point,
                        T threshold,
                        bool return_distances) {
    if (num_queries == 0) return;
    const size_t wg = core::sy::PreferredWorkGroupSize(queue.get_device());
    const size_t global_size =
            ((static_cast<size_t>(num_queries) + wg - 1) / wg) * wg;
    queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                const int64_t q = it.get_global_id(0);
                if (q >= num_queries) return;
                utility::MiniVec<T, 3> query_pos(query_points + 3 * q);
                int bins[8];
                CollectBinsToVisit(query_pos, inv_voxel_size, radius,
                                   hash_table_size, bins);
                const int64_t offset = neighbors_row_splits[q];
                int64_t count = 0;
                for (int bi = 0; bi < 8; ++bi) {
                    const int bin = bins[bi];
                    if (bin < 0) break;
                    const uint32_t begin = hash_table_cell_splits[bin];
                    const uint32_t end = hash_table_cell_splits[bin + 1];
                    for (uint32_t j = begin; j < end; ++j) {
                        const uint32_t idx = point_index_table[j];
                        utility::MiniVec<T, 3> p(points + 3 * idx);
                        if (ignore_query_point && (query_pos == p).all()) {
                            continue;
                        }
                        T dist;
                        if (IsNeighbor(metric, p, query_pos, threshold,
                                       &dist)) {
                            indices[offset + count] = static_cast<TIndex>(idx);
                            if (return_distances) {
                                distances[offset + count] = dist;
                            }
                            ++count;
                        }
                    }
                }
            });
}

/// Single-pass hybrid search: simultaneously counts all points within
/// \p radius and keeps a running top-\p max_knn (by ascending distance) per
/// query in fixed-size output slots. Mirrors WriteNeighborsHybridKernel
/// (CUDA), including its per-query bubble sort of the (small, bounded by
/// max_knn) result slice -- no device-wide sort is needed here since the
/// output size is already capped. Supports L1, L2, and Linf (via \p metric
/// and \ref IsNeighbor), matching CUDA's NeighborTest<METRIC>. As with fixed-
/// radius search: for L2, \p threshold and the returned/compared distances
/// are SQUARED; for L1/Linf they are the metric distance directly (see \ref
/// FixedRadiusThreshold in KnnSearchOpsSYCL.cpp, which the caller uses to
/// compute \p threshold consistently with this).
template <class T, class TIndex>
void WriteNeighborsHybridSYCL(sycl::queue& queue,
                              TIndex* indices,
                              T* distances,
                              TIndex* counts,
                              const uint32_t* const point_index_table,
                              const uint32_t* const hash_table_cell_splits,
                              uint32_t hash_table_size,
                              const T* const query_points,
                              int64_t num_queries,
                              const T* const points,
                              T inv_voxel_size,
                              T radius,
                              Metric metric,
                              T threshold,
                              int max_knn) {
    if (num_queries == 0) return;
    const size_t wg = core::sy::PreferredWorkGroupSize(queue.get_device());
    const size_t global_size =
            ((static_cast<size_t>(num_queries) + wg - 1) / wg) * wg;
    queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                const int64_t q = it.get_global_id(0);
                if (q >= num_queries) return;
                utility::MiniVec<T, 3> query_pos(query_points + 3 * q);
                int bins[8];
                CollectBinsToVisit(query_pos, inv_voxel_size, radius,
                                   hash_table_size, bins);

                const int64_t offset = int64_t(max_knn) * q;
                int count = 0;
                int max_index = 0;
                T max_value = T(0);

                for (int bi = 0; bi < 8; ++bi) {
                    const int bin = bins[bi];
                    if (bin < 0) break;
                    const uint32_t begin = hash_table_cell_splits[bin];
                    const uint32_t end = hash_table_cell_splits[bin + 1];
                    for (uint32_t j = begin; j < end; ++j) {
                        const uint32_t idx = point_index_table[j];
                        utility::MiniVec<T, 3> p(points + 3 * idx);
                        T dist;
                        if (!IsNeighbor(metric, p, query_pos, threshold,
                                        &dist)) {
                            continue;
                        }

                        if (count < max_knn) {
                            indices[offset + count] = static_cast<TIndex>(idx);
                            distances[offset + count] = dist;
                            if (count == 0 || max_value < dist) {
                                max_index = count;
                                max_value = dist;
                            }
                            ++count;
                        } else if (max_value > dist) {
                            indices[offset + max_index] =
                                    static_cast<TIndex>(idx);
                            distances[offset + max_index] = dist;
                            max_value = dist;
                            for (int k = 0; k < max_knn; ++k) {
                                if (distances[offset + k] > max_value) {
                                    max_index = k;
                                    max_value = distances[offset + k];
                                }
                            }
                        }
                    }
                }

                counts[q] = static_cast<TIndex>(count);

                // Bubble sort: count <= max_knn, which is small in practice
                // (e.g. Open3D estimators default to 30), matching CUDA.
                for (int i = 0; i < count - 1; ++i) {
                    for (int j = 0; j < count - i - 1; ++j) {
                        if (distances[offset + j] > distances[offset + j + 1]) {
                            const T dt = distances[offset + j];
                            const TIndex it = indices[offset + j];
                            distances[offset + j] = distances[offset + j + 1];
                            indices[offset + j] = indices[offset + j + 1];
                            distances[offset + j + 1] = dt;
                            indices[offset + j + 1] = it;
                        }
                    }
                }
            });
}

/// Sorts each query's variable-length neighbor segment by ascending
/// distance, entirely on device (no host round trip). Mirrors
/// cub::DeviceSegmentedRadixSort::SortPairs (CUDA): like CUDA, ties are not
/// secondarily ordered by index.
///
/// float uses a scalar uint64 radix key `(query_id << 32) |
/// bit_cast<uint32>(dist)` so oneDPL's sort_by_key stays on the fast radix
/// path (valid because distances are clamped >= 0, so their float32 bit
/// patterns are monotonic as unsigned integers, and num_queries < 2^32
/// always holds here). double cannot use this trick: a monotonic transform
/// of a double needs all 64 bits, leaving no room to also pack the segment
/// id, so it falls back to a struct key + device comparator (oneDPL merge
/// sort, still fully on device).
///
/// Non-blocking: returns the event of the last enqueued command. `query_id`
/// is filled via `oneapi::dpl::upper_bound` (a single coalesced pass over
/// `row_splits_ptr`, which is tiny and normally L1/L2-resident) instead of
/// one work-item per query looping over its whole variable-length segment
/// (worst-case divergence + fully uncoalesced writes). `sort_by_key` and
/// `upper_bound` are synchronous oneDPL calls (no `_async` variant exists
/// for either), so each blocks the *host* until its device work completes;
/// they still need to run, in order, after the async Pass-1 kernels below,
/// which is why an explicit barrier precedes each such call (item 20 rule,
/// see SYCLUtils.h) rather than relying on submission order on this
/// possibly out-of-order (PyTorch) queue. Only `double`'s `sycl::malloc_device`
/// scratch forces a genuine host wait (Phase 2 rule: raw USM feeding an
/// external free must be provably complete first).
template <class T, class TIndex>
sycl::event SortNeighborsByDistanceSYCL(const Device& device,
                                        TIndex* indices_ptr,
                                        T* distances_ptr,
                                        const int64_t* row_splits_ptr,
                                        int64_t num_queries,
                                        int64_t num_indices) {
    if (num_indices == 0) return sycl::event();
    sycl::queue queue = sy::GetQueue(device);
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    // Per-element segment (query) id, so the sort groups each query's
    // neighbors together (query-major, then distance-ascending). For index
    // i, query_id is the largest q with row_splits_ptr[q] <= i, i.e.
    // upper_bound(row_splits_ptr[1..num_queries], i) - 1 (empty segments are
    // skipped naturally since no i falls in [row_splits[q], row_splits[q])).
    Tensor query_id_t = Tensor::Empty({num_indices}, Int64, device);
    int64_t* query_id_ptr = query_id_t.GetDataPtr<int64_t>();
    // Search row_splits_ptr[1..num_queries] (excludes the leading 0) so the
    // result is already the query id with no -1 needed: upper_bound(i)
    // counts how many of those boundaries are <= i, which is exactly q for
    // i in [row_splits[q], row_splits[q+1]). Verified on
    // row_splits=[0,3,3,5] (query 1 empty): i=0..2 -> 0, i=3..4 -> 2.
    oneapi::dpl::upper_bound(
            policy, row_splits_ptr + 1, row_splits_ptr + 1 + num_queries,
            oneapi::dpl::counting_iterator<int64_t>(0),
            oneapi::dpl::counting_iterator<int64_t>(num_indices), query_id_ptr);

    Tensor values_t =
            Tensor::Empty({num_indices}, Dtype::FromType<TIndex>(), device);
    TIndex* values_ptr = values_t.GetDataPtr<TIndex>();
    sycl::event copy_event =
            queue.memcpy(values_ptr, indices_ptr,
                         static_cast<size_t>(num_indices) * sizeof(TIndex));

    const size_t idx_wg = core::sy::PreferredWorkGroupSize(queue.get_device());
    const size_t idx_global_size =
            ((static_cast<size_t>(num_indices) + idx_wg - 1) / idx_wg) * idx_wg;
    const sycl::nd_range<1> idx_nd_range{sycl::range<1>(idx_global_size),
                                         sycl::range<1>(idx_wg)};

    if constexpr (std::is_same<T, float>::value) {
        Tensor keys_t = Tensor::Empty({num_indices}, UInt64, device);
        uint64_t* keys_ptr = keys_t.GetDataPtr<uint64_t>();
        sycl::event keys_event = queue.parallel_for(
                idx_nd_range,
                [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                    const int64_t i = it.get_global_id(0);
                    if (i >= num_indices) return;
                    const uint32_t dist_bits = sycl::bit_cast<uint32_t>(
                            static_cast<float>(distances_ptr[i]));
                    keys_ptr[i] =
                            (static_cast<uint64_t>(query_id_ptr[i]) << 32) |
                            dist_bits;
                });
        // Item 20 rule: sort_by_key has no async variant, so it does not
        // inherit copy_event/keys_event on an out-of-order queue -- barrier
        // both explicitly (sort_by_key reads keys_ptr AND values_ptr).
        queue.ext_oneapi_submit_barrier({copy_event, keys_event});
        // An explicit comparator is required here: empirically, this
        // oneDPL/device combination does not reliably sort correctly when
        // relying on the default `std::less<uint64_t>` overload of
        // sort_by_key (observed silently-wrong ordering on real hardware,
        // despite oneDPL's default overload being documented as equivalent).
        oneapi::dpl::sort_by_key(policy, keys_ptr, keys_ptr + num_indices,
                                 values_ptr,
                                 [](uint64_t a, uint64_t b) { return a < b; });
        // sort_by_key's documented host-blocking semantics were not
        // sufficient in practice to guarantee its device-side work is
        // complete/visible to the write-back kernel below on this in-order
        // queue; force an explicit host wait to be safe.
        queue.wait_and_throw();

        sycl::event write_event = queue.parallel_for(
                idx_nd_range,
                [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                    const int64_t i = it.get_global_id(0);
                    if (i >= num_indices) return;
                    const uint32_t dist_bits =
                            static_cast<uint32_t>(keys_ptr[i] & 0xffffffffu);
                    distances_ptr[i] =
                            static_cast<T>(sycl::bit_cast<float>(dist_bits));
                    indices_ptr[i] = values_ptr[i];
                });
        // `query_id_t`/`values_t`/`keys_t` are RAII Tensors that free their
        // USM as soon as they go out of scope (i.e. right after this
        // function returns), but `write_event`'s kernel above still reads
        // them asynchronously on the device. Without waiting here, the
        // free can race the kernel and corrupt/UAF the buffers. Block until
        // it's provably safe (mirrors the double path's explicit wait
        // before its manual `sycl::free`).
        write_event.wait_and_throw();
        return write_event;
    } else {
        using KeyT = SortKey<T>;
        KeyT* keys = sycl::malloc_device<KeyT>(num_indices, queue);
        sycl::event keys_event = queue.parallel_for(
                idx_nd_range,
                [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                    const int64_t i = it.get_global_id(0);
                    if (i >= num_indices) return;
                    keys[i] = KeyT{query_id_ptr[i], distances_ptr[i]};
                });
        // sort_by_key reads both keys and values_ptr -> barrier both.
        queue.ext_oneapi_submit_barrier({copy_event, keys_event});
        oneapi::dpl::sort_by_key(policy, keys, keys + num_indices, values_ptr,
                                 [](const KeyT& a, const KeyT& b) {
                                     if (a.query_id != b.query_id)
                                         return a.query_id < b.query_id;
                                     return a.dist < b.dist;
                                 });
        // See float-path comment above: force a host wait to ensure the
        // sort's device-side work is complete before the write-back kernel
        // below reads `keys`.
        queue.wait_and_throw();

        sycl::event write_event = queue.parallel_for(
                idx_nd_range,
                [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {
                    const int64_t i = it.get_global_id(0);
                    if (i >= num_indices) return;
                    distances_ptr[i] = keys[i].dist;
                    indices_ptr[i] = values_ptr[i];
                });
        // `keys` is raw USM freed right below -- must be provably safe, so
        // this is the one genuine host block left in this function (Phase 2
        // rule: cannot free USM from a host_task deferred on write_event
        // without adding a dependency the caller cannot express here).
        write_event.wait_and_throw();
        sycl::free(keys, queue);
        return write_event;
    }
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
