// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// SYCL implementation of Voxelize — ports Voxelize.cuh.
//
// Design differences from the CUDA version (documented deviations):
// - The CUDA version uses cub with a two-pass "query size, then allocate"
//   convention (MemoryAllocation.h) because cub device algorithms require a
//   caller-managed scratch buffer of a specific size. oneDPL algorithms
//   manage their own scratch space internally, so this SYCL port allocates
//   scratch buffers directly with sycl::malloc_device/sycl::free and calls
//   the algorithm once (no size-query pass).
// - cub::DeviceRunLengthEncode::Encode -> oneapi::dpl::reduce_by_key (plan
//   §6.4): reducing a constant-1 "value" sequence grouped by the sorted key
//   sequence yields the unique keys plus their run lengths in one call.
//   reduce_by_key has no non-blocking oneDPL async equivalent, so it remains
//   a genuine synchronization point (see RunLengthEncodeSYCL).
// - cub::DeviceRadixSort::SortPairs -> oneapi::dpl::sort_by_key. Also has no
//   async equivalent; a genuine synchronization point.
// - cub::DeviceScan::InclusiveSum -> oneapi::dpl::experimental::inclusive_scan_async
//   (non-blocking; same pattern as InvertNeighborsListSYCL.h).
//
// Each stage takes an optional `deps` event-vector and returns its
// completion event; callers thread these through instead of blocking waits,
// except at genuine host-synchronization points (ReadScalar, used where a
// value must be read back to make a host-side branch/sizing decision) and
// the two oneDPL algorithms above (which have no async form).
//
// MiniVec (open3d/utility/MiniVec.h) is reused as-is: its FN_SPECIFIERS macro
// expands to plain `inline` when not compiled by nvcc, so it is safe to use
// unmodified inside SYCL device kernels.

#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "open3d/core/ParallelFor.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace ml {
namespace impl {

namespace sycl_voxelize_detail {

using open3d::utility::MiniVec;

/// Reads a single device scalar back to the host. Genuine host-sync point
/// (the caller needs the value to make a branch/sizing decision), not a lazy
/// default -- so it blocks-waits after depending on \p deps.
template <class T>
inline T ReadScalar(sycl::queue& queue,
                    const T* device_ptr,
                    const std::vector<sycl::event>& deps = {}) {
    T value{};
    queue.memcpy(&value, device_ptr, sizeof(T), deps).wait();
    return value;
}

/// Assigns each point its batch id (index into row_splits) by looping over
/// batches on the device. Ports ComputeIndicesBatchesKernel.
inline sycl::event ComputeIndicesBatchesSYCL(
        sycl::queue& queue,
        int64_t* indices_batches,
        const int64_t* row_splits,
        int64_t batch_size) {
    if (batch_size == 0) return sycl::event();
    return core::ParallelFor(
            queue, batch_size,
            [=](int64_t b) {
                for (int64_t i = row_splits[b]; i < row_splits[b + 1]; ++i) {
                    indices_batches[i] = b;
                }
            },
            std::vector<sycl::event>{});
}

/// Computes the hash (linear voxel index, offset by batch) for each point.
/// Points outside [points_range_min, points_range_max] get invalid_hash.
/// Ports ComputeHashKernel. Depends on \p deps and returns its completion
/// event (non-blocking).
template <class T, int NDIM>
inline sycl::event ComputeHashSYCL(sycl::queue& queue,
                                   int64_t* hashes,
                                   int64_t num_points,
                                   const T* const points,
                                   const int64_t* const indices_batches,
                                   const MiniVec<T, NDIM> points_range_min_vec,
                                   const MiniVec<T, NDIM> points_range_max_vec,
                                   const MiniVec<T, NDIM> inv_voxel_size,
                                   const MiniVec<int64_t, NDIM> strides,
                                   int64_t batch_hash,
                                   int64_t invalid_hash,
                                   const std::vector<sycl::event>& deps = {}) {
    if (num_points == 0) return sycl::event();
    typedef MiniVec<T, NDIM> Vec_t;
    return core::ParallelFor(queue, num_points, [=](int64_t i) {
        Vec_t point(points + i * NDIM);
        if ((point >= points_range_min_vec && point <= points_range_max_vec)
                    .all()) {
            auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                                  .template cast<int64_t>();
            int64_t h = coords.dot(strides);
            h += indices_batches[i] * batch_hash;
            hashes[i] = h;
        } else {
            hashes[i] = invalid_hash;
        }
    }, deps);
}

/// Element-wise min(counts[i], limit). Ports LimitCountsKernel. Depends on
/// \p deps and returns its completion event (non-blocking).
inline sycl::event LimitCountsSYCL(sycl::queue& queue,
                                   int64_t* counts,
                                   int64_t num,
                                   int64_t limit,
                                   const std::vector<sycl::event>& deps = {}) {
    if (num == 0) return sycl::event();
    return core::ParallelFor(
            queue, num,
            [=](int64_t i) {
                if (counts[i] > limit) counts[i] = limit;
            },
            deps);
}

/// Run-length-encodes a sorted array of keys via `oneapi::dpl::reduce_by_key`
/// (plan §6.4): reducing a constant-1 "value" sequence, grouped by equal
/// consecutive keys, yields the unique keys plus their run lengths in a
/// single device-parallel call. Replaces
/// cub::DeviceRunLengthEncode::Encode (see file-level comment).
/// reduce_by_key has no non-blocking oneDPL async equivalent, so this
/// function is itself a genuine synchronization point (it blocks before
/// returning); the returned run count is needed immediately by the caller
/// anyway (to size subsequent allocations).
///
/// \param unique_keys_out  Output buffer of size >= num_keys (upper bound on
///        the number of runs).
/// \param unique_counts_out    Output buffer of size >= num_keys.
/// \returns    The number of runs (unique consecutive keys).
inline int64_t RunLengthEncodeSYCL(sycl::queue& queue,
                                   const int64_t* const keys,
                                   int64_t num_keys,
                                   int64_t* unique_keys_out,
                                   int64_t* unique_counts_out,
                                   const std::vector<sycl::event>& deps = {}) {
    if (num_keys == 0) return 0;

    // reduce_by_key sums a "value" sequence per run of equal keys; a
    // constant-1 value sequence turns that sum into the run length, i.e.
    // the RLE count cub::DeviceRunLengthEncode::Encode would produce.
    int64_t* ones = sycl::malloc_device<int64_t>(num_keys, queue);
    // reduce_by_key has no event-dependency parameter, so `deps` and the
    // fill must be explicitly waited on before it (a genuine hazard on an
    // out-of-order queue, not a lazy default).
    queue.fill(ones, int64_t(1), num_keys, deps).wait();

    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);
    auto result =
            oneapi::dpl::reduce_by_key(dpl_policy, keys, keys + num_keys, ones,
                                       unique_keys_out, unique_counts_out);
    sycl::free(ones, queue);

    return static_cast<int64_t>(result.first - unique_keys_out);
}

/// hashes[i] /= batch_hash, i.e., converts a voxel hash to a batch id. Ports
/// ComputeBatchIdKernel.
inline void ComputeBatchIdSYCL(sycl::queue& queue,
                               int64_t* hashes,
                               int64_t num_voxels,
                               int64_t batch_hash) {
    if (num_voxels == 0) return;
    core::ParallelFor(queue, num_voxels,
                      [=](int64_t i) { hashes[i] /= batch_hash; });
}

/// Scatters unique_batches_count into num_voxels_per_batch, indexed by
/// unique_batches (distinct destination indices, so no atomics needed).
/// Ports ComputeVoxelPerBatchKernel.
inline void ComputeVoxelPerBatchSYCL(sycl::queue& queue,
                                     int64_t* num_voxels_per_batch,
                                     const int64_t* unique_batches_count,
                                     const int64_t* unique_batches,
                                     int64_t num_batches) {
    if (num_batches == 0) return;
    core::ParallelFor(queue, num_batches, [=](int64_t i) {
        num_voxels_per_batch[unique_batches[i]] = unique_batches_count[i];
    });
}

/// Computes the starting index and clamped point count for each valid voxel,
/// used when the number of voxels exceeds max_voxels. Ports
/// ComputeStartIdxKernel. Depends on \p deps and returns its completion
/// event (non-blocking).
inline sycl::event ComputeStartIdxSYCL(
        sycl::queue& queue,
        int64_t* start_idx,
        int64_t* points_count,
        const int64_t* num_voxels_prefix_sum,
        const int64_t* unique_hashes_count_prefix_sum,
        const int64_t* out_batch_splits,
        int64_t batch_size,
        int64_t max_points_per_voxel,
        const std::vector<sycl::event>& deps = {}) {
    if (batch_size == 0) return sycl::event();
    return core::ParallelFor(queue, batch_size, [=](int64_t b) {
        int64_t voxel_idx = (b == 0) ? 0 : num_voxels_prefix_sum[b - 1];
        const int64_t begin_out = out_batch_splits[b];
        const int64_t end_out = out_batch_splits[b + 1];
        for (int64_t out_idx = begin_out; out_idx < end_out;
             ++out_idx, ++voxel_idx) {
            if (voxel_idx == 0) {
                start_idx[out_idx] = 0;
                points_count[out_idx] = sycl::min(
                        max_points_per_voxel, unique_hashes_count_prefix_sum[0]);
            } else {
                start_idx[out_idx] =
                        unique_hashes_count_prefix_sum[voxel_idx - 1];
                points_count[out_idx] = sycl::min(
                        max_points_per_voxel,
                        unique_hashes_count_prefix_sum[voxel_idx] -
                                unique_hashes_count_prefix_sum[voxel_idx - 1]);
            }
        }
    }, deps);
}

/// Computes integer voxel coordinates for each valid voxel from the position
/// of its first (sorted) point. Ports ComputeVoxelCoordsKernel.
template <class T, int NDIM>
inline void ComputeVoxelCoordsSYCL(sycl::queue& queue,
                                   int32_t* voxel_coords,
                                   const T* const points,
                                   const int64_t* const point_indices,
                                   const int64_t* const prefix_sum,
                                   const MiniVec<T, NDIM> points_range_min_vec,
                                   const MiniVec<T, NDIM> inv_voxel_size,
                                   int64_t num_voxels) {
    if (num_voxels == 0) return;
    typedef MiniVec<T, NDIM> Vec_t;
    core::ParallelFor(queue, num_voxels, [=](int64_t i) {
        const int64_t point_idx = point_indices[prefix_sum[i]];
        Vec_t point(points + point_idx * NDIM);
        auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                              .template cast<int32_t>();
        for (int d = 0; d < NDIM; ++d) {
            voxel_coords[i * NDIM + d] = coords[d];
        }
    });
}

/// Copies (limited-count) point indices for each valid voxel into the flat
/// output array. Ports CopyPointIndicesKernel.
inline void CopyPointIndicesSYCL(sycl::queue& queue,
                                 int64_t* out,
                                 const int64_t* const point_indices,
                                 const int64_t* const prefix_sum_in,
                                 const int64_t* const prefix_sum_out,
                                 int64_t num_voxels) {
    if (num_voxels == 0) return;
    core::ParallelFor(queue, num_voxels, [=](int64_t i) {
        const int64_t begin_out = (i == 0) ? 0 : prefix_sum_out[i - 1];
        const int64_t end_out = prefix_sum_out[i];
        int64_t in_idx = prefix_sum_in[i];
        for (int64_t out_idx = begin_out; out_idx < end_out;
             ++out_idx, ++in_idx) {
            out[out_idx] = point_indices[in_idx];
        }
    });
}

}  // namespace sycl_voxelize_detail

/// Voxelizes a point cloud (SYCL). See Voxelize.cuh for the full parameter
/// documentation; the signature matches the CUDA version except that this
/// function does the work in one call (no temp-size query pass — see the
/// file-level comment on design differences) and takes a SYCL queue directly
/// (kernels run on the caller-supplied queue).
///
/// \p voxel_size, \p points_range_min, \p points_range_max point to *host*
/// memory (matching the CUDA convention; the PyTorch dispatch layer already
/// keeps these small per-dimension arrays on the CPU).
template <class T, int NDIM, class OUTPUT_ALLOCATOR>
void VoxelizeSYCL(sycl::queue& queue,
                  size_t num_points,
                  const T* const points,
                  const size_t batch_size,
                  const int64_t* const row_splits,
                  const T* const voxel_size,
                  const T* const points_range_min,
                  const T* const points_range_max,
                  const int64_t max_points_per_voxel,
                  const int64_t max_voxels,
                  OUTPUT_ALLOCATOR& output_allocator) {
    using namespace sycl_voxelize_detail;
    using namespace open3d::utility;
    typedef MiniVec<T, NDIM> Vec_t;

    const Vec_t inv_voxel_size = T(1) / Vec_t(voxel_size);
    const Vec_t points_range_min_vec(points_range_min);
    const Vec_t points_range_max_vec(points_range_max);
    MiniVec<int32_t, NDIM> extents =
            ceil((points_range_max_vec - points_range_min_vec) * inv_voxel_size)
                    .template cast<int32_t>();
    MiniVec<int64_t, NDIM> strides;
    for (int i = 0; i < NDIM; ++i) {
        strides[i] = 1;
        for (int j = 0; j < i; ++j) strides[i] *= extents[j];
    }
    const int64_t batch_hash = strides[NDIM - 1] * extents[NDIM - 1];
    const int64_t invalid_hash = batch_hash * int64_t(batch_size);

    // Degenerate case: no input points. Still emit correctly-shaped (empty)
    // outputs and all-zero batch splits.
    if (num_points == 0) {
        int64_t* out_batch_splits = nullptr;
        output_allocator.AllocVoxelBatchSplits(&out_batch_splits,
                                               batch_size + 1);
        if (batch_size)
            queue.fill(out_batch_splits, int64_t(0), batch_size + 1).wait();
        int32_t* out_voxel_coords = nullptr;
        output_allocator.AllocVoxelCoords(&out_voxel_coords, 0, NDIM);
        int64_t* out_voxel_row_splits = nullptr;
        output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits, 1);
        queue.fill(out_voxel_row_splits, int64_t(0), 1).wait();
        int64_t* out_point_indices = nullptr;
        output_allocator.AllocVoxelPointIndices(&out_point_indices, 0);
        return;
    }

    // --- Step 1: hash each point (voxel index + batch offset) ------------
    int64_t* indices_batches = sycl::malloc_device<int64_t>(num_points, queue);
    int64_t* point_indices = sycl::malloc_device<int64_t>(num_points, queue);
    int64_t* hashes = sycl::malloc_device<int64_t>(num_points, queue);

    sycl::event indices_batches_event = ComputeIndicesBatchesSYCL(
            queue, indices_batches, row_splits, int64_t(batch_size));

    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);
    core::ParallelFor(queue, int64_t(num_points),
                      [=](int64_t i) { point_indices[i] = i; });

    // Depends on indices_batches_event: ComputeHashSYCL reads
    // indices_batches, which ComputeIndicesBatchesSYCL wrote asynchronously.
    sycl::event hashes_event = ComputeHashSYCL<T, NDIM>(
            queue, hashes, int64_t(num_points), points, indices_batches,
            points_range_min_vec, points_range_max_vec, inv_voxel_size,
            strides, batch_hash, invalid_hash, {indices_batches_event});
    // indices_batches is freed here, so its last reader (ComputeHashSYCL)
    // must have completed first; sycl::free is not queue-ordered.
    hashes_event.wait();
    sycl::free(indices_batches, queue);

    // --- Step 2: sort points by hash (groups points into voxels) ---------
    // sort_by_key has no async oneDPL equivalent, so this is a genuine
    // synchronization point (blocks internally before returning).
    oneapi::dpl::sort_by_key(dpl_policy, hashes, hashes + num_points,
                             point_indices);

    // --- Step 3: run-length-encode the sorted hashes -> unique voxels ----
    int64_t* unique_hashes = sycl::malloc_device<int64_t>(num_points, queue);
    int64_t* unique_hashes_count =
            sycl::malloc_device<int64_t>(num_points, queue);

    int64_t num_voxels =
            RunLengthEncodeSYCL(queue, hashes, int64_t(num_points),
                                unique_hashes, unique_hashes_count);
    sycl::free(hashes, queue);

    const int64_t last_hash =
            ReadScalar(queue, unique_hashes + (num_voxels - 1));
    if (last_hash == invalid_hash) {
        // Points outside the valid range were hashed to invalid_hash and
        // sort last; drop that trailing "voxel".
        --num_voxels;
    }

    // --- Step 4: prefix sum of (unlimited) per-voxel counts --------------
    int64_t* unique_hashes_count_prefix_sum = sycl::malloc_device<int64_t>(
            num_voxels > 0 ? num_voxels : 1, queue);
    sycl::event scan1_event;
    if (num_voxels > 0) {
        scan1_event = oneapi::dpl::experimental::inclusive_scan_async(
                              dpl_policy, unique_hashes_count,
                              unique_hashes_count + num_voxels,
                              unique_hashes_count_prefix_sum)
                              .event();
    }

    // Clamp per-voxel point counts to max_points_per_voxel (applied after
    // the prefix sum above, matching the CUDA ordering: the prefix sum uses
    // the true point ranges, while the clamped counts become the final
    // per-voxel output sizes). LimitCountsSYCL writes unique_hashes_count in
    // place while scan1_event may still be reading it, so it depends on
    // scan1_event (a genuine hazard, not a lazy default) via the
    // event-accepting overload instead of a blocking wait. Its own
    // completion event is captured below (limit1_event) since
    // unique_hashes_count is read again later (aliased as points_count).
    sycl::event limit1_event;
    if (max_points_per_voxel < int64_t(num_points)) {
        limit1_event = LimitCountsSYCL(queue, unique_hashes_count,
                                       num_voxels, max_points_per_voxel,
                                       {scan1_event});
    }

    // --- Step 5: group voxels by batch -------------------------------
    ComputeBatchIdSYCL(queue, unique_hashes, num_voxels, batch_hash);

    int64_t* unique_batches = sycl::malloc_device<int64_t>(
            batch_size > 0 ? batch_size : 1, queue);
    int64_t* unique_batches_count = sycl::malloc_device<int64_t>(
            batch_size > 0 ? batch_size : 1, queue);
    int64_t num_batches =
            RunLengthEncodeSYCL(queue, unique_hashes, num_voxels,
                                unique_batches, unique_batches_count);
    sycl::free(unique_hashes, queue);

    int64_t* num_voxels_per_batch = sycl::malloc_device<int64_t>(
            batch_size > 0 ? batch_size : 1, queue);
    queue.fill(num_voxels_per_batch, int64_t(0),
               batch_size > 0 ? batch_size : 1)
            .wait();
    ComputeVoxelPerBatchSYCL(queue, num_voxels_per_batch, unique_batches_count,
                             unique_batches, num_batches);
    sycl::free(unique_batches, queue);
    sycl::free(unique_batches_count, queue);

    // Prefix sum of the *unlimited* per-batch voxel counts: gives the index
    // of the first (unlimited) voxel of each batch within the global list.
    // Only used by ComputeStartIdxSYCL when num_voxels > max_voxels.
    int64_t* num_voxels_prefix_sum = sycl::malloc_device<int64_t>(
            batch_size > 0 ? batch_size : 1, queue);
    sycl::event scan2_event;
    if (batch_size > 0) {
        scan2_event = oneapi::dpl::experimental::inclusive_scan_async(
                              dpl_policy, num_voxels_per_batch,
                              num_voxels_per_batch + batch_size,
                              num_voxels_prefix_sum)
                              .event();
    }

    // LimitCountsSYCL writes num_voxels_per_batch in place while scan2_event
    // may still be reading it, so it depends on scan2_event (a genuine
    // hazard, not a lazy default). num_voxels_per_batch_ready tracks
    // whichever of {scan2_event, limit2_event} last touched the buffer, so
    // the Step 6 scan below depends on the right one.
    sycl::event num_voxels_per_batch_ready = scan2_event;
    if (num_voxels >= max_voxels) {
        num_voxels_per_batch_ready = LimitCountsSYCL(
                queue, num_voxels_per_batch, int64_t(batch_size), max_voxels,
                {scan2_event});
    }

    // --- Step 6: batch splits over the (possibly limited) voxel counts ---
    int64_t* out_batch_splits = nullptr;
    output_allocator.AllocVoxelBatchSplits(&out_batch_splits, batch_size + 1);
    queue.fill(out_batch_splits, int64_t(0), 1).wait();
    sycl::event scan3_event;
    if (batch_size > 0) {
        scan3_event = oneapi::dpl::experimental::inclusive_scan_async(
                              dpl_policy, num_voxels_per_batch,
                              num_voxels_per_batch + batch_size,
                              out_batch_splits + 1, num_voxels_per_batch_ready)
                              .event();
    }
    queue.ext_oneapi_submit_barrier({scan3_event}).wait();
    sycl::free(num_voxels_per_batch, queue);

    const int64_t num_valid_voxels =
            ReadScalar(queue, out_batch_splits + batch_size);

    // --- Step 7: per-voxel start index + clamped point count --------------
    int64_t* start_idx = sycl::malloc_device<int64_t>(
            num_valid_voxels > 0 ? num_valid_voxels : 1, queue);
    int64_t* points_count = nullptr;
    bool points_count_is_alias = false;

    sycl::event start_idx_ready_event;
    if (num_voxels <= max_voxels) {
        // All voxels kept: start_idx/points_count come directly from the
        // (unlimited-then-clamped) global arrays computed above.
        queue.fill(start_idx, int64_t(0), 1).wait();
        if (num_voxels > 1) {
            // Depends on scan1_event (unique_hashes_count_prefix_sum).
            start_idx_ready_event =
                    queue.memcpy(start_idx + 1, unique_hashes_count_prefix_sum,
                                (num_voxels - 1) * sizeof(int64_t),
                                scan1_event);
        }
        points_count = unique_hashes_count;
        points_count_is_alias = true;
    } else {
        points_count = sycl::malloc_device<int64_t>(num_valid_voxels, queue);
        // ComputeStartIdxSYCL reads num_voxels_prefix_sum (scan2_event) and
        // unique_hashes_count_prefix_sum (scan1_event); depends on both via
        // the event-accepting overload (a genuine hazard, not a lazy
        // default) instead of a blocking wait.
        start_idx_ready_event = ComputeStartIdxSYCL(
                queue, start_idx, points_count, num_voxels_prefix_sum,
                unique_hashes_count_prefix_sum, out_batch_splits,
                int64_t(batch_size), max_points_per_voxel,
                {scan1_event, scan2_event});
    }
    // num_voxels_prefix_sum/unique_hashes_count_prefix_sum are read by
    // start_idx_ready_event's op (whichever branch above); sycl::free is not
    // queue-ordered, so wait for it before freeing.
    start_idx_ready_event.wait();
    sycl::free(num_voxels_prefix_sum, queue);
    sycl::free(unique_hashes_count_prefix_sum, queue);

    // --- Step 8: row splits over output points per voxel ------------------
    int64_t* out_voxel_row_splits = nullptr;
    output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits,
                                              num_valid_voxels + 1);
    queue.fill(out_voxel_row_splits, int64_t(0), 1).wait();
    sycl::event scan4_event;
    if (num_valid_voxels > 0) {
        // When points_count_is_alias, points_count is unique_hashes_count,
        // which limit1_event may still be writing (a genuine hazard, not a
        // lazy default); depend on it in that case. Otherwise points_count
        // was written by ComputeStartIdxSYCL, already awaited via
        // start_idx_ready_event.wait() above.
        scan4_event = oneapi::dpl::experimental::inclusive_scan_async(
                              dpl_policy, points_count,
                              points_count + num_valid_voxels,
                              out_voxel_row_splits + 1,
                              points_count_is_alias ? limit1_event
                                                    : sycl::event())
                              .event();
    }

    // --- Step 9: voxel coordinates + compacted point indices --------------
    int32_t* out_voxel_coords = nullptr;
    output_allocator.AllocVoxelCoords(&out_voxel_coords, num_valid_voxels,
                                      NDIM);
    // ComputeVoxelCoordsSYCL reads start_idx (already complete: see Step 7's
    // ComputeStartIdxSYCL/queue.memcpy, both awaited via core::ParallelFor's
    // or queue.memcpy's own blocking) -- not points_count/out_voxel_row_splits
    // (scan4_event), so it may run concurrently with the Step 8 scan on an
    // out-of-order queue; no dependency needed.
    ComputeVoxelCoordsSYCL<T, NDIM>(
            queue, out_voxel_coords, points, point_indices, start_idx,
            points_range_min_vec, inv_voxel_size, num_valid_voxels);

    const int64_t num_valid_points =
            num_valid_voxels > 0
                    ? ReadScalar(queue,
                                out_voxel_row_splits + num_valid_voxels,
                                {scan4_event})
                    : 0;
    int64_t* out_point_indices = nullptr;
    output_allocator.AllocVoxelPointIndices(&out_point_indices,
                                            num_valid_points);
    // CopyPointIndicesSYCL reads out_voxel_row_splits (scan4_event, already
    // awaited above via ReadScalar's blocking memcpy) and start_idx (already
    // complete, see above).
    CopyPointIndicesSYCL(queue, out_point_indices, point_indices, start_idx,
                         out_voxel_row_splits + 1, num_valid_voxels);

    sycl::free(start_idx, queue);
    if (!points_count_is_alias) sycl::free(points_count, queue);
    sycl::free(unique_hashes_count, queue);
    sycl::free(point_indices, queue);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
