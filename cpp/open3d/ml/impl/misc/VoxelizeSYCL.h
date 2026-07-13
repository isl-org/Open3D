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
// - cub::DeviceRadixSort::SortPairs -> oneapi::dpl::sort_by_key.
// - cub::DeviceScan::InclusiveSum -> std::inclusive_scan with a oneDPL device
//   policy (same pattern as InvertNeighborsListSYCL.h).
//
// MiniVec (open3d/utility/MiniVec.h) is reused as-is: its FN_SPECIFIERS macro
// expands to plain `inline` when not compiled by nvcc, so it is safe to use
// unmodified inside SYCL device kernels.

#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace ml {
namespace impl {

namespace sycl_voxelize_detail {

using open3d::utility::MiniVec;

/// Reads a single device scalar back to the host (blocking).
template <class T>
inline T ReadScalar(sycl::queue& queue, const T* device_ptr) {
    T value{};
    queue.memcpy(&value, device_ptr, sizeof(T)).wait();
    return value;
}

/// Assigns each point its batch id (index into row_splits) by looping over
/// batches on the device. Ports ComputeIndicesBatchesKernel.
inline void ComputeIndicesBatchesSYCL(sycl::queue& queue,
                                      int64_t* indices_batches,
                                      const int64_t* row_splits,
                                      int64_t batch_size) {
    if (batch_size == 0) return;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(batch_size), [=](sycl::item<1> item) {
            const int64_t b = item.get_id(0);
            for (int64_t i = row_splits[b]; i < row_splits[b + 1]; ++i) {
                indices_batches[i] = b;
            }
        });
    });
    queue.wait_and_throw();
}

/// Computes the hash (linear voxel index, offset by batch) for each point.
/// Points outside [points_range_min, points_range_max] get invalid_hash.
/// Ports ComputeHashKernel.
template <class T, int NDIM>
inline void ComputeHashSYCL(sycl::queue& queue,
                            int64_t* hashes,
                            int64_t num_points,
                            const T* const points,
                            const int64_t* const indices_batches,
                            const MiniVec<T, NDIM> points_range_min_vec,
                            const MiniVec<T, NDIM> points_range_max_vec,
                            const MiniVec<T, NDIM> inv_voxel_size,
                            const MiniVec<int64_t, NDIM> strides,
                            int64_t batch_hash,
                            int64_t invalid_hash) {
    if (num_points == 0) return;
    typedef MiniVec<T, NDIM> Vec_t;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_points), [=](sycl::item<1> item) {
            const int64_t i = item.get_id(0);
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
        });
    });
    queue.wait_and_throw();
}

/// Element-wise min(counts[i], limit). Ports LimitCountsKernel.
inline void LimitCountsSYCL(sycl::queue& queue,
                            int64_t* counts,
                            int64_t num,
                            int64_t limit) {
    if (num == 0) return;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num), [=](sycl::item<1> item) {
            const int64_t i = item.get_id(0);
            if (counts[i] > limit) counts[i] = limit;
        });
    });
    queue.wait_and_throw();
}

/// Run-length-encodes a sorted array of keys via `oneapi::dpl::reduce_by_key`
/// (plan §6.4): reducing a constant-1 "value" sequence, grouped by equal
/// consecutive keys, yields the unique keys plus their run lengths in a
/// single device-parallel call. Replaces
/// cub::DeviceRunLengthEncode::Encode (see file-level comment).
///
/// \param unique_keys_out  Output buffer of size >= num_keys (upper bound on
///        the number of runs).
/// \param unique_counts_out    Output buffer of size >= num_keys.
/// \returns    The number of runs (unique consecutive keys).
inline int64_t RunLengthEncodeSYCL(sycl::queue& queue,
                                   const int64_t* const keys,
                                   int64_t num_keys,
                                   int64_t* unique_keys_out,
                                   int64_t* unique_counts_out) {
    if (num_keys == 0) return 0;

    // reduce_by_key sums a "value" sequence per run of equal keys; a
    // constant-1 value sequence turns that sum into the run length, i.e.
    // the RLE count cub::DeviceRunLengthEncode::Encode would produce.
    int64_t* ones = sycl::malloc_device<int64_t>(num_keys, queue);
    queue.fill(ones, int64_t(1), num_keys).wait();

    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);
    auto result =
            oneapi::dpl::reduce_by_key(dpl_policy, keys, keys + num_keys, ones,
                                       unique_keys_out, unique_counts_out);
    queue.wait_and_throw();
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
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_voxels), [=](sycl::item<1> item) {
            hashes[item.get_id(0)] /= batch_hash;
        });
    });
    queue.wait_and_throw();
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
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_batches), [=](sycl::item<1> item) {
            const int64_t i = item.get_id(0);
            num_voxels_per_batch[unique_batches[i]] = unique_batches_count[i];
        });
    });
    queue.wait_and_throw();
}

/// Computes the starting index and clamped point count for each valid voxel,
/// used when the number of voxels exceeds max_voxels. Ports
/// ComputeStartIdxKernel.
inline void ComputeStartIdxSYCL(sycl::queue& queue,
                                int64_t* start_idx,
                                int64_t* points_count,
                                const int64_t* num_voxels_prefix_sum,
                                const int64_t* unique_hashes_count_prefix_sum,
                                const int64_t* out_batch_splits,
                                int64_t batch_size,
                                int64_t max_points_per_voxel) {
    if (batch_size == 0) return;
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(batch_size), [=](sycl::item<1> item) {
            const int64_t b = item.get_id(0);
            int64_t voxel_idx = (b == 0) ? 0 : num_voxels_prefix_sum[b - 1];
            const int64_t begin_out = out_batch_splits[b];
            const int64_t end_out = out_batch_splits[b + 1];
            for (int64_t out_idx = begin_out; out_idx < end_out;
                 ++out_idx, ++voxel_idx) {
                if (voxel_idx == 0) {
                    start_idx[out_idx] = 0;
                    points_count[out_idx] =
                            sycl::min(max_points_per_voxel,
                                      unique_hashes_count_prefix_sum[0]);
                } else {
                    start_idx[out_idx] =
                            unique_hashes_count_prefix_sum[voxel_idx - 1];
                    points_count[out_idx] = sycl::min(
                            max_points_per_voxel,
                            unique_hashes_count_prefix_sum[voxel_idx] -
                                    unique_hashes_count_prefix_sum[voxel_idx -
                                                                   1]);
                }
            }
        });
    });
    queue.wait_and_throw();
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
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_voxels), [=](sycl::item<1> item) {
            const int64_t i = item.get_id(0);
            const int64_t point_idx = point_indices[prefix_sum[i]];
            Vec_t point(points + point_idx * NDIM);
            auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                                  .template cast<int32_t>();
            for (int d = 0; d < NDIM; ++d) {
                voxel_coords[i * NDIM + d] = coords[d];
            }
        });
    });
    queue.wait_and_throw();
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
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_voxels), [=](sycl::item<1> item) {
            const int64_t i = item.get_id(0);
            const int64_t begin_out = (i == 0) ? 0 : prefix_sum_out[i - 1];
            const int64_t end_out = prefix_sum_out[i];
            int64_t in_idx = prefix_sum_in[i];
            for (int64_t out_idx = begin_out; out_idx < end_out;
                 ++out_idx, ++in_idx) {
                out[out_idx] = point_indices[in_idx];
            }
        });
    });
    queue.wait_and_throw();
}

}  // namespace sycl_voxelize_detail

/// Voxelizes a point cloud (SYCL). See Voxelize.cuh for the full parameter
/// documentation; the signature matches the CUDA version except that this
/// function does the work in one call (no temp-size query pass — see the
/// file-level comment on design differences) and takes a sycl::queue.
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

    ComputeIndicesBatchesSYCL(queue, indices_batches, row_splits,
                              int64_t(batch_size));

    auto dpl_policy = oneapi::dpl::execution::make_device_policy(queue);
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_points), [=](sycl::item<1> item) {
            point_indices[item.get_id(0)] = int64_t(item.get_id(0));
        });
    });
    queue.wait_and_throw();

    ComputeHashSYCL<T, NDIM>(queue, hashes, int64_t(num_points), points,
                             indices_batches, points_range_min_vec,
                             points_range_max_vec, inv_voxel_size, strides,
                             batch_hash, invalid_hash);
    sycl::free(indices_batches, queue);

    // --- Step 2: sort points by hash (groups points into voxels) ---------
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
    if (num_voxels > 0) {
        std::inclusive_scan(dpl_policy, unique_hashes_count,
                            unique_hashes_count + num_voxels,
                            unique_hashes_count_prefix_sum);
        queue.wait_and_throw();
    }

    // Clamp per-voxel point counts to max_points_per_voxel (applied after
    // the prefix sum above, matching the CUDA ordering: the prefix sum uses
    // the true point ranges, while the clamped counts become the final
    // per-voxel output sizes).
    if (max_points_per_voxel < int64_t(num_points)) {
        LimitCountsSYCL(queue, unique_hashes_count, num_voxels,
                        max_points_per_voxel);
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
    if (batch_size > 0) {
        std::inclusive_scan(dpl_policy, num_voxels_per_batch,
                            num_voxels_per_batch + batch_size,
                            num_voxels_prefix_sum);
        queue.wait_and_throw();
    }

    if (num_voxels >= max_voxels) {
        LimitCountsSYCL(queue, num_voxels_per_batch, int64_t(batch_size),
                        max_voxels);
    }

    // --- Step 6: batch splits over the (possibly limited) voxel counts ---
    int64_t* out_batch_splits = nullptr;
    output_allocator.AllocVoxelBatchSplits(&out_batch_splits, batch_size + 1);
    queue.fill(out_batch_splits, int64_t(0), 1).wait();
    if (batch_size > 0) {
        std::inclusive_scan(dpl_policy, num_voxels_per_batch,
                            num_voxels_per_batch + batch_size,
                            out_batch_splits + 1);
        queue.wait_and_throw();
    }
    sycl::free(num_voxels_per_batch, queue);

    const int64_t num_valid_voxels =
            ReadScalar(queue, out_batch_splits + batch_size);

    // --- Step 7: per-voxel start index + clamped point count --------------
    int64_t* start_idx = sycl::malloc_device<int64_t>(
            num_valid_voxels > 0 ? num_valid_voxels : 1, queue);
    int64_t* points_count = nullptr;
    bool points_count_is_alias = false;

    if (num_voxels <= max_voxels) {
        // All voxels kept: start_idx/points_count come directly from the
        // (unlimited-then-clamped) global arrays computed above.
        queue.fill(start_idx, int64_t(0), 1).wait();
        if (num_voxels > 1) {
            queue.memcpy(start_idx + 1, unique_hashes_count_prefix_sum,
                         (num_voxels - 1) * sizeof(int64_t))
                    .wait();
        }
        points_count = unique_hashes_count;
        points_count_is_alias = true;
    } else {
        points_count = sycl::malloc_device<int64_t>(num_valid_voxels, queue);
        ComputeStartIdxSYCL(queue, start_idx, points_count,
                            num_voxels_prefix_sum,
                            unique_hashes_count_prefix_sum, out_batch_splits,
                            int64_t(batch_size), max_points_per_voxel);
    }
    sycl::free(num_voxels_prefix_sum, queue);
    sycl::free(unique_hashes_count_prefix_sum, queue);

    // --- Step 8: row splits over output points per voxel ------------------
    int64_t* out_voxel_row_splits = nullptr;
    output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits,
                                              num_valid_voxels + 1);
    queue.fill(out_voxel_row_splits, int64_t(0), 1).wait();
    if (num_valid_voxels > 0) {
        std::inclusive_scan(dpl_policy, points_count,
                            points_count + num_valid_voxels,
                            out_voxel_row_splits + 1);
        queue.wait_and_throw();
    }

    // --- Step 9: voxel coordinates + compacted point indices --------------
    int32_t* out_voxel_coords = nullptr;
    output_allocator.AllocVoxelCoords(&out_voxel_coords, num_valid_voxels,
                                      NDIM);
    ComputeVoxelCoordsSYCL<T, NDIM>(
            queue, out_voxel_coords, points, point_indices, start_idx,
            points_range_min_vec, inv_voxel_size, num_valid_voxels);

    const int64_t num_valid_points =
            num_valid_voxels > 0
                    ? ReadScalar(queue, out_voxel_row_splits + num_valid_voxels)
                    : 0;
    int64_t* out_point_indices = nullptr;
    output_allocator.AllocVoxelPointIndices(&out_point_indices,
                                            num_valid_points);
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
