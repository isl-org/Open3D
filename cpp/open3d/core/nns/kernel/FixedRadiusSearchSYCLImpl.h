// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file FixedRadiusSearchSYCLImpl.h
/// \brief SYCL device kernels: uniform-grid fixed-radius and hybrid neighbor search.
///
/// Included by \ref KnnSearchOpsSYCL.cpp and PyTorch SYCL op wrappers (not public
/// API). Algorithm matches CUDA `FixedRadiusSearchImpl.cuh`; shared geometry in
/// \ref NeighborSearchCommon.h (`SpatialHash`, `ComputeVoxelIndex`). See
/// `nns/SYCL_DESIGN.md` for overview.
///
/// \section FrsSyclGrid Grid build (\ref BuildSpatialHashTableSYCL)
///
/// 1. Bucket dataset into a uniform spatial-hash grid with **cell size `2 * radius`**
///    (any neighbor within `radius` lies in the query cell or one of seven
///    corner-adjacent cells — **8 bins** visited per query, deduplicated).
/// 2. **Count** points per cell (per batch).
/// 3. **Inclusive scan** of counts → CSR offsets (`hash_table_cell_splits`;
///    oneDPL; CUDA uses CUB `DeviceScan::InclusiveSum`).
/// 4. **Scatter** point indices into cell ranges (`hash_table_index`).
///
/// Runs on **SYCL CPU and GPU**. Host driver uses one in-order queue with minimal
/// sync between batch loops.
///
/// \section FrsSyclQuery Query kernels
///
/// | Mode | Kernels | Passes |
/// |------|---------|--------|
/// | Fixed-radius | \ref CountNeighborsSYCL, \ref WriteNeighborsSYCL | Count → scan on host → allocate → gather |
/// | Hybrid | \ref WriteNeighborsHybridSYCL | Single pass: running top-`max_knn` + in-radius count, then bubble sort |
/// | Optional sort | \ref SortNeighborsByDistanceSYCL (`sort=true`) | Segmented sort per query segment |
///
/// **Metrics:** L1, L2, Linf (same as CUDA). L2 compares **squared** distance to
/// `radius²`; L1/Linf compare metric distance to `radius`.
///
/// **Sort (`sort=true`):** oneDPL `sort_by_key` — `float` uses packed radix key;
/// `double` uses struct key + comparator (needs full 64-bit distance). Ties are
/// **not** secondarily ordered by neighbor index (CUDA parity).
///
/// \section FrsSyclVsCuda Primitives (CUDA → SYCL)
///
/// | Role | CUDA | SYCL (this file) |
/// |------|------|------------------|
/// | Prefix sum | CUB inclusive scan | oneDPL inclusive scan |
/// | Segmented sort | CUB segmented radix sort | oneDPL `sort_by_key` |
/// | Query parallelism | 1 thread / query | 1 work-item / query (`parallel_for`) |

#pragma once

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
#include <type_traits>

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
template <class T>
void BuildSpatialHashTableSYCLRaw(sycl::queue& queue,
                                  const T* points_ptr,
                                  T inv_voxel_size,
                                  int batch_size,
                                  const int64_t* host_points_row_splits,
                                  const uint32_t* host_hash_table_splits,
                                  uint32_t* cell_splits_ptr,
                                  size_t cell_splits_size,
                                  uint32_t* index_ptr) {
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    queue.memset(cell_splits_ptr, 0, cell_splits_size * sizeof(uint32_t))
            .wait_and_throw();

    // Pass 1: count points per cell (into cell_splits_i[hash + 1]), so the
    // scan in Pass 2 turns this into CSR start offsets.
    for (int b = 0; b < batch_size; ++b) {
        const int64_t point_begin = host_points_row_splits[b];
        const int64_t point_end = host_points_row_splits[b + 1];
        const int64_t num_points_i = point_end - point_begin;
        if (num_points_i == 0) continue;
        const uint32_t first_cell_idx = host_hash_table_splits[b];
        const uint32_t hash_table_size =
                host_hash_table_splits[b + 1] - first_cell_idx;
        uint32_t* cell_splits_i = cell_splits_ptr + first_cell_idx;

        queue.parallel_for(
                sycl::range<1>(num_points_i),
                [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                    const int64_t i = point_begin + id[0];
                    utility::MiniVec<T, 3> pos(points_ptr + 3 * i);
                    auto voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);
                    const size_t hash =
                            SpatialHash(voxel_index) % hash_table_size;
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                            cnt(cell_splits_i[hash + 1]);
                    cnt.fetch_add(1);
                });
    }
    queue.wait_and_throw();

    // Pass 2: turn per-cell counts into CSR start offsets with a *single*
    // scan over the whole (all-batches-concatenated) array -- mirrors CUDA,
    // which calls cub::DeviceScan::InclusiveSum once over the full
    // count_tmp/hash_table_cell_splits buffer rather than once per batch.
    if (cell_splits_size > 0) {
        std::inclusive_scan(policy, cell_splits_ptr,
                            cell_splits_ptr + cell_splits_size, cell_splits_ptr);
    }
    queue.wait_and_throw();

    // Pass 3: scatter point indices into their cell's slot range. One reused
    // slot-counter buffer (memset per batch on this in-order queue) avoids
    // per-batch allocation; uses USM so PyTorch and Tensor callers share this
    // path without Open3D Tensor scratch.
    uint32_t max_hash_table_size = 0;
    for (int b = 0; b < batch_size; ++b) {
        max_hash_table_size = std::max<uint32_t>(
                max_hash_table_size, host_hash_table_splits[b + 1] -
                                               host_hash_table_splits[b]);
    }
    uint32_t* slot_counts_ptr = nullptr;
    if (max_hash_table_size > 0) {
        slot_counts_ptr =
                sycl::malloc_device<uint32_t>(max_hash_table_size, queue);
    }
    for (int b = 0; b < batch_size; ++b) {
        const int64_t point_begin = host_points_row_splits[b];
        const int64_t point_end = host_points_row_splits[b + 1];
        const int64_t num_points_i = point_end - point_begin;
        if (num_points_i == 0) continue;
        const uint32_t first_cell_idx = host_hash_table_splits[b];
        const uint32_t hash_table_size =
                host_hash_table_splits[b + 1] - first_cell_idx;
        const uint32_t* cell_splits_i = cell_splits_ptr + first_cell_idx;

        if (slot_counts_ptr) {
            queue.memset(slot_counts_ptr, 0,
                         static_cast<size_t>(hash_table_size) *
                                 sizeof(uint32_t));
        }

        queue.parallel_for(
                sycl::range<1>(num_points_i),
                [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                    const int64_t i = point_begin + id[0];
                    utility::MiniVec<T, 3> pos(points_ptr + 3 * i);
                    auto voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);
                    const size_t hash =
                            SpatialHash(voxel_index) % hash_table_size;
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                            cnt(slot_counts_ptr[hash]);
                    const uint32_t slot = cnt.fetch_add(1);
                    index_ptr[cell_splits_i[hash] + slot] =
                            static_cast<uint32_t>(i);
                });
    }
    if (slot_counts_ptr) {
        sycl::free(slot_counts_ptr, queue);
    }
    queue.wait_and_throw();
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
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    const int batch_size = static_cast<int>(points_row_splits.GetShape(0)) - 1;

    // Read host-side batch arrays (points_row_splits / hash_table_splits are
    // CPU tensors per the FixedRadiusIndex contract).
    std::vector<int64_t> host_pts_row_splits(batch_size + 1);
    std::vector<uint32_t> host_ht_splits(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
        host_pts_row_splits[i] = points_row_splits[i].Item<int64_t>();
        host_ht_splits[i] = hash_table_splits[i].Item<uint32_t>();
    }

    BuildSpatialHashTableSYCLRaw<T>(
            queue, points.GetDataPtr<T>(), T(1) / T(2 * radius), batch_size,
            host_pts_row_splits.data(), host_ht_splits.data(),
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            static_cast<size_t>(hash_table_cell_splits.NumElements()),
            hash_table_index.GetDataPtr<uint32_t>());
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
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
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
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
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
/// output size is already capped.
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
                              T threshold,
                              int max_knn) {
    if (num_queries == 0) return;
    queue.parallel_for(
            sycl::range<1>(num_queries),
            [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                const int64_t q = id[0];
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
                        const T dist = SquaredDistance(p, query_pos);
                        if (dist > threshold) continue;

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
template <class T, class TIndex>
void SortNeighborsByDistanceSYCL(const Device& device,
                                 TIndex* indices_ptr,
                                 T* distances_ptr,
                                 const int64_t* row_splits_ptr,
                                 int64_t num_queries,
                                 int64_t num_indices) {
    if (num_indices == 0) return;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    // Per-element segment (query) id, so the sort groups each query's
    // neighbors together (query-major, then distance-ascending).
    Tensor query_id_t = Tensor::Empty({num_indices}, Int64, device);
    int64_t* query_id_ptr = query_id_t.GetDataPtr<int64_t>();
    queue.parallel_for(sycl::range<1>(num_queries),
                       [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                           const int64_t q = id[0];
                           for (int64_t i = row_splits_ptr[q];
                                i < row_splits_ptr[q + 1]; ++i) {
                               query_id_ptr[i] = q;
                           }
                       });

    Tensor values_t =
            Tensor::Empty({num_indices}, Dtype::FromType<TIndex>(), device);
    TIndex* values_ptr = values_t.GetDataPtr<TIndex>();
    queue.wait_and_throw();
    queue.memcpy(values_ptr, indices_ptr,
                 static_cast<size_t>(num_indices) * sizeof(TIndex));
    queue.wait_and_throw();

    if constexpr (std::is_same<T, float>::value) {
        Tensor keys_t = Tensor::Empty({num_indices}, UInt64, device);
        uint64_t* keys_ptr = keys_t.GetDataPtr<uint64_t>();
        queue.parallel_for(
                sycl::range<1>(num_indices),
                [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                    const int64_t i = id[0];
                    const uint32_t dist_bits = sycl::bit_cast<uint32_t>(
                            static_cast<float>(distances_ptr[i]));
                    keys_ptr[i] =
                            (static_cast<uint64_t>(query_id_ptr[i]) << 32) |
                            dist_bits;
                });
        queue.wait_and_throw();

        oneapi::dpl::sort_by_key(policy, keys_ptr, keys_ptr + num_indices,
                                 values_ptr);

        queue.parallel_for(
                sycl::range<1>(num_indices),
                [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                    const int64_t i = id[0];
                    const uint32_t dist_bits =
                            static_cast<uint32_t>(keys_ptr[i] & 0xffffffffu);
                    distances_ptr[i] =
                            static_cast<T>(sycl::bit_cast<float>(dist_bits));
                    indices_ptr[i] = values_ptr[i];
                });
        queue.wait_and_throw();
    } else {
        using KeyT = SortKey<T>;
        KeyT* keys = sycl::malloc_device<KeyT>(num_indices, queue);
        queue.parallel_for(sycl::range<1>(num_indices),
                           [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                               const int64_t i = id[0];
                               keys[i] =
                                       KeyT{query_id_ptr[i], distances_ptr[i]};
                           });
        queue.wait_and_throw();

        oneapi::dpl::sort_by_key(policy, keys, keys + num_indices, values_ptr,
                                 [](const KeyT& a, const KeyT& b) {
                                     if (a.query_id != b.query_id)
                                         return a.query_id < b.query_id;
                                     return a.dist < b.dist;
                                 });

        queue.parallel_for(sycl::range<1>(num_indices),
                           [=](sycl::id<1> id) [[intel::kernel_args_restrict]] {
                               const int64_t i = id[0];
                               distances_ptr[i] = keys[i].dist;
                               indices_ptr[i] = values_ptr[i];
                           });
        queue.wait_and_throw();
        sycl::free(keys, queue);
    }
}

}  // namespace nns
}  // namespace core
}  // namespace open3d
