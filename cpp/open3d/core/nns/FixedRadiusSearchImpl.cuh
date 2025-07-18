// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <math.h>

#include <cub/cub.cuh>

#include "open3d/core/nns/MemoryAllocation.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace core {
namespace nns {
namespace impl {

namespace {

template <class T>
using Vec3 = utility::MiniVec<T, 3>;

/// Computes the distance of two points and tests if the distance is below a
/// threshold.
///
/// \tparam METRIC    The distance metric. One of L1, L2, Linf.
/// \tparam T    Floating point type for the distances.
///
/// \param p1           A 3D point
/// \param p2           Another 3D point
/// \param dist         Output parameter for the distance.
/// \param threshold    The scalar threshold.
///
/// \return Returns true if the distance is <= threshold.
///
template <int METRIC = L2, class T>
inline __device__ bool NeighborTest(const Vec3<T>& p1,
                                    const Vec3<T>& p2,
                                    T* dist,
                                    T threshold) {
    bool result = false;
    if (METRIC == Linf) {
        Vec3<T> d = (p1 - p2).abs();
        *dist = d[0] > d[1] ? d[0] : d[1];
        *dist = *dist > d[2] ? *dist : d[2];
    } else if (METRIC == L1) {
        Vec3<T> d = (p1 - p2).abs();
        *dist = (d[0] + d[1] + d[2]);
    } else {
        Vec3<T> d = p1 - p2;
        *dist = d.dot(d);
    }
    result = *dist <= threshold;
    return result;
}

/// Kernel for CountHashTableEntries
template <class T>
__global__ void CountHashTableEntriesKernel(uint32_t* count_table,
                                            size_t hash_table_size,
                                            T inv_voxel_size,
                                            const T* const __restrict__ points,
                                            size_t num_points) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) return;

    Vec3<T> pos(&points[idx * 3]);

    Vec3<int> voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);
    size_t hash = SpatialHash(voxel_index) % hash_table_size;
    atomicAdd(&count_table[hash + 1], 1);
}

/// Counts for each hash entry the number of points that map to this entry.
///
/// \param count_table    Pointer to the table for counting.
///        The first element will not be used, i.e. the
///        number of points for the first hash entry is in count_table[1].
///        This array must be initialized before calling this function.
///
/// \param count_table_size    This is the size of the hash table + 1.
///
/// \param inv_voxel_size    Reciproval of the voxel size
///
/// \param points    Array with the 3D point positions.
///
/// \param num_points    The number of points.
///
template <class T>
void CountHashTableEntries(const cudaStream_t& stream,
                           uint32_t* count_table,
                           size_t count_table_size,
                           T inv_voxel_size,
                           const T* points,
                           size_t num_points) {
    const int BLOCKSIZE = 64;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_points, block.x);

    if (grid.x)
        CountHashTableEntriesKernel<T><<<grid, block, 0, stream>>>(
                count_table, count_table_size - 1, inv_voxel_size, points,
                num_points);
}

/// Kernel for ComputePointIndexTable
template <class T>
__global__ void ComputePointIndexTableKernel(
        uint32_t* __restrict__ point_index_table,
        uint32_t* __restrict__ count_tmp,
        const uint32_t* const __restrict__ hash_table_cell_splits,
        size_t hash_table_size,
        T inv_voxel_size,
        const T* const __restrict__ points,
        const size_t points_start_idx,
        const size_t points_end_idx) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x + points_start_idx;
    if (idx >= points_end_idx) return;

    Vec3<T> pos(&points[idx * 3]);

    Vec3<int> voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);
    size_t hash = SpatialHash(voxel_index[0], voxel_index[1], voxel_index[2]) %
                  hash_table_size;

    point_index_table[hash_table_cell_splits[hash] +
                      atomicAdd(&count_tmp[hash], 1)] = idx;
}

/// Writes the index of the points to the hash cells.
///
/// \param point_index_table    The output array storing the point indices for
///        all cells. Start and end of each cell is defined by
///        \p hash_table_prefix_sum
///
/// \param count_tmp    Temporary memory of size
///                     \p hash_table_cell_splits_size.
///
/// \param hash_table_cell_splits    The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_cell_splits_size    The size of the hash table.
///
/// \param inv_voxel_size    Reciproval of the voxel size
///
/// \param points    Array with the 3D point positions.
///
/// \param num_points    The number of points.
///
template <class T>
void ComputePointIndexTable(
        const cudaStream_t& stream,
        uint32_t* __restrict__ point_index_table,
        uint32_t* __restrict__ count_tmp,
        const uint32_t* const __restrict__ hash_table_cell_splits,
        size_t hash_table_cell_splits_size,
        T inv_voxel_size,
        const T* const __restrict__ points,
        size_t points_start_idx,
        size_t points_end_idx) {
    cudaMemsetAsync(count_tmp, 0,
                    sizeof(uint32_t) * hash_table_cell_splits_size, stream);
    size_t num_points = points_end_idx - points_start_idx;

    const int BLOCKSIZE = 64;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_points, block.x);

    if (grid.x)
        ComputePointIndexTableKernel<T><<<grid, block, 0, stream>>>(
                point_index_table, count_tmp, hash_table_cell_splits,
                hash_table_cell_splits_size - 1, inv_voxel_size, points,
                points_start_idx, points_end_idx);
}

/// Kernel for CountNeighbors
template <int METRIC, bool IGNORE_QUERY_POINT, class T>
__global__ void CountNeighborsKernel(
        uint32_t* __restrict__ neighbors_count,
        const uint32_t* const __restrict__ point_index_table,
        const uint32_t* const __restrict__ hash_table_cell_splits,
        size_t hash_table_size,
        const T* const __restrict__ query_points,
        size_t num_queries,
        const T* const __restrict__ points,
        const T inv_voxel_size,
        const T radius,
        const T threshold) {
    int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    int count = 0;  // counts the number of neighbors for this query point

    Vec3<T> query_pos(query_points[query_idx * 3 + 0],
                      query_points[query_idx * 3 + 1],
                      query_points[query_idx * 3 + 2]);
    Vec3<int> voxel_index = ComputeVoxelIndex(query_pos, inv_voxel_size);
    int hash = SpatialHash(voxel_index[0], voxel_index[1], voxel_index[2]) %
               hash_table_size;

    int bins_to_visit[8] = {hash, -1, -1, -1, -1, -1, -1, -1};

    for (int dz = -1; dz <= 1; dz += 2)
        for (int dy = -1; dy <= 1; dy += 2)
            for (int dx = -1; dx <= 1; dx += 2) {
                Vec3<T> p = query_pos + radius * Vec3<T>(T(dx), T(dy), T(dz));
                voxel_index = ComputeVoxelIndex(p, inv_voxel_size);
                hash = SpatialHash(voxel_index[0], voxel_index[1],
                                   voxel_index[2]) %
                       hash_table_size;

                // insert without duplicates
                for (int i = 0; i < 8; ++i) {
                    if (bins_to_visit[i] == hash) {
                        break;
                    } else if (bins_to_visit[i] == -1) {
                        bins_to_visit[i] = hash;
                        break;
                    }
                }
            }

    for (int bin_i = 0; bin_i < 8; ++bin_i) {
        int bin = bins_to_visit[bin_i];
        if (bin == -1) break;

        size_t begin_idx = hash_table_cell_splits[bin];
        size_t end_idx = hash_table_cell_splits[bin + 1];

        for (size_t j = begin_idx; j < end_idx; ++j) {
            uint32_t idx = point_index_table[j];

            Vec3<T> p(&points[idx * 3 + 0]);
            if (IGNORE_QUERY_POINT) {
                if ((query_pos == p).all()) continue;
            }

            T dist;
            if (NeighborTest<METRIC>(p, query_pos, &dist, threshold)) ++count;
        }
    }
    neighbors_count[query_idx] = count;
}

/// Count the number of neighbors for each query point
///
/// \param neighbors_count    Output array for counting the number of neighbors.
///        The size of the array is \p num_queries.
///
/// \param point_index_table    The array storing the point indices for all
///        cells. Start and end of each cell is defined by \p
///        hash_table_cell_splits
///
/// \param hash_table_cell_splits    The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param query_points    Array with the 3D query positions. This may be the
///        same array as \p points.
///
/// \param num_queries    The number of query points.
///
/// \param points    Array with the 3D point positions.
///
/// \param num_points    The number of points.
///
/// \param inv_voxel_size    Reciproval of the voxel size
///
/// \param radius    The search radius.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param ignore_query_point    If true then points with the same position as
///        the query point will be ignored.
///
template <class T>
void CountNeighbors(const cudaStream_t& stream,
                    uint32_t* neighbors_count,
                    const uint32_t* const point_index_table,
                    const uint32_t* const hash_table_cell_splits,
                    size_t hash_table_cell_splits_size,
                    const T* const query_points,
                    size_t num_queries,
                    const T* const points,
                    const T inv_voxel_size,
                    const T radius,
                    const Metric metric,
                    const bool ignore_query_point) {
    const T threshold = (metric == L2 ? radius * radius : radius);

    const int BLOCKSIZE = 64;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_queries, block.x);

    if (grid.x) {
#define FN_PARAMETERS                                                   \
    neighbors_count, point_index_table, hash_table_cell_splits,         \
            hash_table_cell_splits_size - 1, query_points, num_queries, \
            points, inv_voxel_size, radius, threshold

#define CALL_TEMPLATE(METRIC)                                    \
    if (METRIC == metric) {                                      \
        if (ignore_query_point)                                  \
            CountNeighborsKernel<METRIC, true, T>                \
                    <<<grid, block, 0, stream>>>(FN_PARAMETERS); \
        else                                                     \
            CountNeighborsKernel<METRIC, false, T>               \
                    <<<grid, block, 0, stream>>>(FN_PARAMETERS); \
    }

        CALL_TEMPLATE(L1)
        CALL_TEMPLATE(L2)
        CALL_TEMPLATE(Linf)

#undef CALL_TEMPLATE
#undef FN_PARAMETERS
    }
}

/// Kernel for WriteNeighborsIndicesAndDistances
template <class T,
          class TIndex,
          int METRIC,
          bool IGNORE_QUERY_POINT,
          bool RETURN_DISTANCES>
__global__ void WriteNeighborsIndicesAndDistancesKernel(
        TIndex* __restrict__ indices,
        T* __restrict__ distances,
        const int64_t* const __restrict__ neighbors_row_splits,
        const uint32_t* const __restrict__ point_index_table,
        const uint32_t* const __restrict__ hash_table_cell_splits,
        size_t hash_table_size,
        const T* const __restrict__ query_points,
        size_t num_queries,
        const T* const __restrict__ points,
        const T inv_voxel_size,
        const T radius,
        const T threshold) {
    int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    int count = 0;  // counts the number of neighbors for this query point

    size_t indices_offset = neighbors_row_splits[query_idx];

    Vec3<T> query_pos(query_points[query_idx * 3 + 0],
                      query_points[query_idx * 3 + 1],
                      query_points[query_idx * 3 + 2]);
    Vec3<int> voxel_index = ComputeVoxelIndex(query_pos, inv_voxel_size);
    int hash = SpatialHash(voxel_index) % hash_table_size;

    int bins_to_visit[8] = {hash, -1, -1, -1, -1, -1, -1, -1};

    for (int dz = -1; dz <= 1; dz += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dx = -1; dx <= 1; dx += 2) {
                Vec3<T> p = query_pos + radius * Vec3<T>(T(dx), T(dy), T(dz));
                voxel_index = ComputeVoxelIndex(p, inv_voxel_size);
                hash = SpatialHash(voxel_index) % hash_table_size;

                // insert without duplicates
                for (int i = 0; i < 8; ++i) {
                    if (bins_to_visit[i] == hash) {
                        break;
                    } else if (bins_to_visit[i] == -1) {
                        bins_to_visit[i] = hash;
                        break;
                    }
                }
            }
        }
    }

    for (int bin_i = 0; bin_i < 8; ++bin_i) {
        int bin = bins_to_visit[bin_i];
        if (bin == -1) break;

        size_t begin_idx = hash_table_cell_splits[bin];
        size_t end_idx = hash_table_cell_splits[bin + 1];

        for (size_t j = begin_idx; j < end_idx; ++j) {
            uint32_t idx = point_index_table[j];

            Vec3<T> p(&points[idx * 3 + 0]);
            if (IGNORE_QUERY_POINT) {
                if ((query_pos == p).all()) continue;
            }

            T dist;
            if (NeighborTest<METRIC>(p, query_pos, &dist, threshold)) {
                indices[indices_offset + count] = idx;
                if (RETURN_DISTANCES) {
                    distances[indices_offset + count] = dist;
                }
                ++count;
            }
        }
    }
}

/// Write indices and distances of neighbors for each query point
///
/// \param indices    Output array with the neighbors indices.
///
/// \param distances    Output array with the neighbors distances. May be null
///        if return_distances is false.
///
/// \param neighbors_row_splits    This is the prefix sum which describes
///        start and end of the neighbors and distances for each query point.
///
/// \param point_index_table    The array storing the point indices for all
///        cells. Start and end of each cell is defined by \p
///        hash_table_cell_splits
///
/// \param hash_table_cell_splits    The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param query_points    Array with the 3D query positions. This may be the
///        same array as \p points.
///
/// \param num_queries    The number of query points.
///
/// \param points    Array with the 3D point positions.
///
/// \param num_points    The number of points.
///
/// \param inv_voxel_size    Reciproval of the voxel size
///
/// \param radius    The search radius.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param ignore_query_point    If true then points with the same position as
///        the query point will be ignored.
///
/// \param return_distances    If true then this function will return the
///        distances for each neighbor to its query point in the same format
///        as the indices.
///        Note that for the L2 metric the squared distances will be returned!!
template <class T, class TIndex>
void WriteNeighborsIndicesAndDistances(
        const cudaStream_t& stream,
        TIndex* indices,
        T* distances,
        const int64_t* const neighbors_row_splits,
        const uint32_t* const point_index_table,
        const uint32_t* const hash_table_cell_splits,
        size_t hash_table_cell_splits_size,
        const T* const query_points,
        size_t num_queries,
        const T* const points,
        const T inv_voxel_size,
        const T radius,
        const Metric metric,
        const bool ignore_query_point,
        const bool return_distances) {
    const T threshold = (metric == L2 ? radius * radius : radius);

    const int BLOCKSIZE = 64;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_queries, block.x);

    if (grid.x) {
#define FN_PARAMETERS                                                          \
    indices, distances, neighbors_row_splits, point_index_table,               \
            hash_table_cell_splits, hash_table_cell_splits_size, query_points, \
            num_queries, points, inv_voxel_size, radius, threshold

#define CALL_TEMPLATE(METRIC, IGNORE_QUERY_POINT, RETURN_DISTANCES)      \
    if (METRIC == metric && IGNORE_QUERY_POINT == ignore_query_point &&  \
        RETURN_DISTANCES == return_distances) {                          \
        WriteNeighborsIndicesAndDistancesKernel<                         \
                T, TIndex, METRIC, IGNORE_QUERY_POINT, RETURN_DISTANCES> \
                <<<grid, block, 0, stream>>>(FN_PARAMETERS);             \
    }

#define CALL_TEMPLATE2(METRIC)         \
    CALL_TEMPLATE(METRIC, true, true)  \
    CALL_TEMPLATE(METRIC, true, false) \
    CALL_TEMPLATE(METRIC, false, true) \
    CALL_TEMPLATE(METRIC, false, false)

#define CALL_TEMPLATE3 \
    CALL_TEMPLATE2(L1) \
    CALL_TEMPLATE2(L2) \
    CALL_TEMPLATE2(Linf)

        CALL_TEMPLATE3

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef FN_PARAMETERS
    }
}

/// Kernel for WriteNeighborsHybrid
template <class T, class TIndex, int METRIC, bool RETURN_DISTANCES>
__global__ void WriteNeighborsHybridKernel(
        TIndex* __restrict__ indices,
        T* __restrict__ distances,
        TIndex* __restrict__ counts,
        size_t query_offset,
        const uint32_t* const __restrict__ point_index_table,
        const uint32_t* const __restrict__ hash_table_cell_splits,
        size_t hash_table_size,
        const T* const __restrict__ query_points,
        size_t num_queries,
        const T* const __restrict__ points,
        const T inv_voxel_size,
        const T radius,
        const T threshold,
        const int max_knn) {
    int query_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    int count = 0;  // counts the number of neighbors for this query point

    size_t indices_offset = query_offset + max_knn * query_idx;

    Vec3<T> query_pos(query_points[query_idx * 3 + 0],
                      query_points[query_idx * 3 + 1],
                      query_points[query_idx * 3 + 2]);
    Vec3<int> voxel_index = ComputeVoxelIndex(query_pos, inv_voxel_size);
    int hash = SpatialHash(voxel_index) % hash_table_size;

    int bins_to_visit[8] = {hash, -1, -1, -1, -1, -1, -1, -1};

    for (int dz = -1; dz <= 1; dz += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dx = -1; dx <= 1; dx += 2) {
                Vec3<T> p = query_pos + radius * Vec3<T>(T(dx), T(dy), T(dz));
                voxel_index = ComputeVoxelIndex(p, inv_voxel_size);
                hash = SpatialHash(voxel_index) % hash_table_size;

                // insert without duplicates
                for (int i = 0; i < 8; ++i) {
                    if (bins_to_visit[i] == hash) {
                        break;
                    } else if (bins_to_visit[i] == -1) {
                        bins_to_visit[i] = hash;
                        break;
                    }
                }
            }
        }
    }

    int max_index;
    T max_value;

    for (int bin_i = 0; bin_i < 8; ++bin_i) {
        int bin = bins_to_visit[bin_i];
        if (bin == -1) break;

        size_t begin_idx = hash_table_cell_splits[bin];
        size_t end_idx = hash_table_cell_splits[bin + 1];

        for (size_t j = begin_idx; j < end_idx; ++j) {
            uint32_t idx = point_index_table[j];

            Vec3<T> p(&points[idx * 3 + 0]);

            T dist;
            if (NeighborTest<METRIC>(p, query_pos, &dist, threshold)) {
                // If count if less than max_knn, record idx and dist.
                if (count < max_knn) {
                    indices[indices_offset + count] = idx;
                    distances[indices_offset + count] = dist;
                    // Update max_index and max_value.
                    if (count == 0 || max_value < dist) {
                        max_index = count;
                        max_value = dist;
                    }
                    // Increase count
                    ++count;
                } else {
                    // If dist is smaller than current max_value.
                    if (max_value > dist) {
                        // Replace idx and dist at current max_index.
                        indices[indices_offset + max_index] = idx;
                        distances[indices_offset + max_index] = dist;
                        // Update max_value
                        max_value = dist;
                        // Find max_index.
                        for (auto k = 0; k < max_knn; ++k) {
                            if (distances[indices_offset + k] > max_value) {
                                max_index = k;
                                max_value = distances[indices_offset + k];
                            }
                        }
                    }
                }
            }
        }
    }

    counts[query_idx] = count;

    // bubble sort
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (distances[indices_offset + j] >
                distances[indices_offset + j + 1]) {
                T dist_tmp = distances[indices_offset + j];
                TIndex ind_tmp = indices[indices_offset + j];
                distances[indices_offset + j] =
                        distances[indices_offset + j + 1];
                indices[indices_offset + j] = indices[indices_offset + j + 1];
                distances[indices_offset + j + 1] = dist_tmp;
                indices[indices_offset + j + 1] = ind_tmp;
            }
        }
    }
}

/// Write indices and distances for each query point in hybrid search mode.
///
/// \param indices    Output array with the neighbors indices.
///
/// \param distances    Output array with the neighbors distances. May be null
///        if return_distances is false.
///
/// \param counts    Output array with the neighbour counts.
///
/// \param point_index_table    The array storing the point indices for all
///        cells. Start and end of each cell is defined by
///        \p hash_table_cell_splits
///
/// \param hash_table_cell_splits    The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param query_points    Array with the 3D query positions. This may be the
///        same array as \p points.
///
/// \param num_queries    The number of query points.
///
/// \param points    Array with the 3D point positions.
///
/// \param num_points    The number of points.
///
/// \param inv_voxel_size    Reciproval of the voxel size
///
/// \param radius    The search radius.
///
/// \param metric    One of L1, L2, Linf. Defines the distance metric for the
///        search.
///
/// \param ignore_query_point    If true then points with the same position as
///        the query point will be ignored.
///
/// \param return_distances    If true then this function will return the
///        distances for each neighbor to its query point in the same format
///        as the indices.
///        Note that for the L2 metric the squared distances will be returned!!
template <class T, class TIndex>
void WriteNeighborsHybrid(const cudaStream_t& stream,
                          TIndex* indices,
                          T* distances,
                          TIndex* counts,
                          size_t query_offset,
                          const uint32_t* const point_index_table,
                          const uint32_t* const hash_table_cell_splits,
                          size_t hash_table_cell_splits_size,
                          const T* const query_points,
                          size_t num_queries,
                          const T* const points,
                          const T inv_voxel_size,
                          const T radius,
                          const int max_knn,
                          const Metric metric,
                          const bool return_distances) {
    const T threshold = (metric == L2 ? radius * radius : radius);

    const int BLOCKSIZE = 64;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = utility::DivUp(num_queries, block.x);

    if (grid.x) {
#define FN_PARAMETERS                                                  \
    indices, distances, counts, query_offset, point_index_table,       \
            hash_table_cell_splits, hash_table_cell_splits_size - 1,   \
            query_points, num_queries, points, inv_voxel_size, radius, \
            threshold, max_knn

#define CALL_TEMPLATE(METRIC, RETURN_DISTANCES)                         \
    if (METRIC == metric && RETURN_DISTANCES == return_distances) {     \
        WriteNeighborsHybridKernel<T, TIndex, METRIC, RETURN_DISTANCES> \
                <<<grid, block, 0, stream>>>(FN_PARAMETERS);            \
    }

#define CALL_TEMPLATE2(METRIC)  \
    CALL_TEMPLATE(METRIC, true) \
    CALL_TEMPLATE(METRIC, false)

#define CALL_TEMPLATE3 \
    CALL_TEMPLATE2(L1) \
    CALL_TEMPLATE2(L2) \
    CALL_TEMPLATE2(Linf)

        CALL_TEMPLATE3

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef FN_PARAMETERS
    }
}

}  // namespace

template <class T>
void BuildSpatialHashTableCUDA(const cudaStream_t& stream,
                               void* temp,
                               size_t& temp_size,
                               int texture_alignment,
                               const size_t num_points,
                               const T* const points,
                               const T radius,
                               const size_t points_row_splits_size,
                               const int64_t* points_row_splits,
                               const uint32_t* hash_table_splits,
                               const size_t hash_table_cell_splits_size,
                               uint32_t* hash_table_cell_splits,
                               uint32_t* hash_table_index) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case pointer alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    std::pair<uint32_t*, size_t> count_tmp =
            mem_temp.Alloc<uint32_t>(hash_table_cell_splits_size);

    const int batch_size = points_row_splits_size - 1;
    const T voxel_size = 2 * radius;
    const T inv_voxel_size = 1 / voxel_size;

    // count number of points per hash entry
    if (!get_temp_size) {
        cudaMemsetAsync(count_tmp.first, 0, sizeof(uint32_t) * count_tmp.second,
                        stream);

        for (int i = 0; i < batch_size; ++i) {
            const size_t hash_table_size =
                    hash_table_splits[i + 1] - hash_table_splits[i];
            const size_t first_cell_idx = hash_table_splits[i];
            const size_t num_points_i =
                    points_row_splits[i + 1] - points_row_splits[i];
            const T* const points_i = points + 3 * points_row_splits[i];

            CountHashTableEntries(stream, count_tmp.first + first_cell_idx,
                                  hash_table_size + 1, inv_voxel_size, points_i,
                                  num_points_i);
        }
    }

    // compute prefix sum of the hash entry counts and store in
    // hash_table_cell_splits
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);
        cub::DeviceScan::InclusiveSum(inclusive_scan_temp.first,
                                      inclusive_scan_temp.second,
                                      count_tmp.first, hash_table_cell_splits,
                                      count_tmp.second, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

        if (!get_temp_size) {
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    count_tmp.first, hash_table_cell_splits, count_tmp.second,
                    stream);
        }

        mem_temp.Free(inclusive_scan_temp);
    }

    // now compute the global indices which allows us to lookup the point index
    // for the entries in the hash cell
    if (!get_temp_size) {
        for (int i = 0; i < batch_size; ++i) {
            const size_t hash_table_size =
                    hash_table_splits[i + 1] - hash_table_splits[i];
            const size_t first_cell_idx = hash_table_splits[i];
            const size_t points_start_idx = points_row_splits[i];
            const size_t points_end_idx = points_row_splits[i + 1];
            ComputePointIndexTable(stream, hash_table_index, count_tmp.first,
                                   hash_table_cell_splits + first_cell_idx,
                                   hash_table_size + 1, inv_voxel_size, points,
                                   points_start_idx, points_end_idx);
        }
    }

    mem_temp.Free(count_tmp);

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }
}

template <class T, class TIndex>
void SortPairs(void* temp,
               size_t& temp_size,
               int texture_alignment,
               int64_t num_indices,
               int64_t num_segments,
               const int64_t* query_neighbors_row_splits,
               TIndex* indices_unsorted,
               T* distances_unsorted,
               TIndex* indices_sorted,
               T* distances_sorted) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case pointer alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    std::pair<void*, size_t> sort_temp(nullptr, 0);

    cub::DeviceSegmentedRadixSort::SortPairs(
            sort_temp.first, sort_temp.second, distances_unsorted,
            distances_sorted, indices_unsorted, indices_sorted, num_indices,
            num_segments, query_neighbors_row_splits,
            query_neighbors_row_splits + 1);
    sort_temp = mem_temp.Alloc(sort_temp.second);

    if (!get_temp_size) {
        cub::DeviceSegmentedRadixSort::SortPairs(
                sort_temp.first, sort_temp.second, distances_unsorted,
                distances_sorted, indices_unsorted, indices_sorted, num_indices,
                num_segments, query_neighbors_row_splits,
                query_neighbors_row_splits + 1);
    }
    mem_temp.Free(sort_temp);

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }
}

template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void FixedRadiusSearchCUDA(const cudaStream_t& stream,
                           void* temp,
                           size_t& temp_size,
                           int texture_alignment,
                           int64_t* query_neighbors_row_splits,
                           size_t num_points,
                           const T* const points,
                           size_t num_queries,
                           const T* const queries,
                           const T radius,
                           const size_t points_row_splits_size,
                           const int64_t* const points_row_splits,
                           const size_t queries_row_splits_size,
                           const int64_t* const queries_row_splits,
                           const uint32_t* const hash_table_splits,
                           size_t hash_table_cell_splits_size,
                           const uint32_t* const hash_table_cell_splits,
                           const uint32_t* const hash_table_index,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           OUTPUT_ALLOCATOR& output_allocator) {
    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case pointer alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    // return empty output arrays if there are no points
    if ((0 == num_points || 0 == num_queries) && !get_temp_size) {
        cudaMemsetAsync(query_neighbors_row_splits, 0,
                        sizeof(int64_t) * (num_queries + 1), stream);
        TIndex* indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T* distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);

        return;
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    const int batch_size = points_row_splits_size - 1;
    const T voxel_size = 2 * radius;
    const T inv_voxel_size = 1 / voxel_size;

    std::pair<uint32_t*, size_t> query_neighbors_count =
            mem_temp.Alloc<uint32_t>(num_queries);

    // we need this value to compute the size of the index array
    if (!get_temp_size) {
        for (int i = 0; i < batch_size; ++i) {
            const size_t hash_table_size =
                    hash_table_splits[i + 1] - hash_table_splits[i];
            const size_t first_cell_idx = hash_table_splits[i];
            const size_t queries_start_idx = queries_row_splits[i];
            const T* const queries_i = queries + 3 * queries_row_splits[i];
            const size_t num_queries_i =
                    queries_row_splits[i + 1] - queries_row_splits[i];

            CountNeighbors(
                    stream, query_neighbors_count.first + queries_start_idx,
                    hash_table_index, hash_table_cell_splits + first_cell_idx,
                    hash_table_size + 1, queries_i, num_queries_i, points,
                    inv_voxel_size, radius, metric, ignore_query_point);
        }
    }

    int64_t last_prefix_sum_entry = 0;
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);
        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                query_neighbors_count.first, query_neighbors_row_splits + 1,
                num_queries, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

        if (!get_temp_size) {
            // set first element to zero
            cudaMemsetAsync(query_neighbors_row_splits, 0, sizeof(int64_t),
                            stream);
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    query_neighbors_count.first, query_neighbors_row_splits + 1,
                    num_queries, stream);

            // get the last value
            cudaMemcpyAsync(&last_prefix_sum_entry,
                            query_neighbors_row_splits + num_queries,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            // wait for the async copies
            while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
            }
        }
        mem_temp.Free(inclusive_scan_temp);
    }
    mem_temp.Free(query_neighbors_count);

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }

    if (!get_temp_size) {
        // allocate the output array for the neighbor indices
        const size_t num_indices = last_prefix_sum_entry;

        TIndex* indices_ptr;
        T* distances_ptr;

        output_allocator.AllocIndices(&indices_ptr, num_indices);
        output_allocator.AllocDistances(&distances_ptr, num_indices);
        for (int i = 0; i < batch_size; ++i) {
            const size_t hash_table_size =
                    hash_table_splits[i + 1] - hash_table_splits[i];
            const size_t first_cell_idx = hash_table_splits[i];
            const T* const queries_i = queries + 3 * queries_row_splits[i];
            const size_t num_queries_i =
                    queries_row_splits[i + 1] - queries_row_splits[i];

            WriteNeighborsIndicesAndDistances(
                    stream, indices_ptr, distances_ptr,
                    query_neighbors_row_splits + queries_row_splits[i],
                    hash_table_index, hash_table_cell_splits + first_cell_idx,
                    hash_table_size, queries_i, num_queries_i, points,
                    inv_voxel_size, radius, metric, ignore_query_point,
                    return_distances);
        }
    }
}

//// Hybrid Search
template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void HybridSearchCUDA(const cudaStream_t stream,
                      size_t num_points,
                      const T* const points,
                      size_t num_queries,
                      const T* const queries,
                      const T radius,
                      const int max_knn,
                      const size_t points_row_splits_size,
                      const int64_t* const points_row_splits,
                      const size_t queries_row_splits_size,
                      const int64_t* const queries_row_splits,
                      const uint32_t* const hash_table_splits,
                      size_t hash_table_cell_splits_size,
                      const uint32_t* const hash_table_cell_splits,
                      const uint32_t* const hash_table_index,
                      const Metric metric,
                      OUTPUT_ALLOCATOR& output_allocator) {
    // return empty output arrays if there are no points
    if (0 == num_points || 0 == num_queries) {
        TIndex* indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T* distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);

        TIndex* counts_ptr;
        output_allocator.AllocCounts(&counts_ptr, 0);
        return;
    }

    const int batch_size = points_row_splits_size - 1;
    const T voxel_size = 2 * radius;
    const T inv_voxel_size = 1 / voxel_size;

    // Allocate output pointers.
    const size_t num_indices = num_queries * max_knn;

    TIndex* indices_ptr;
    output_allocator.AllocIndices(&indices_ptr, num_indices, -1);

    T* distances_ptr;
    output_allocator.AllocDistances(&distances_ptr, num_indices, 0);

    TIndex* counts_ptr;
    output_allocator.AllocCounts(&counts_ptr, num_queries, 0);

    for (int i = 0; i < batch_size; ++i) {
        const size_t hash_table_size =
                hash_table_splits[i + 1] - hash_table_splits[i];
        const size_t query_offset = max_knn * queries_row_splits[i];
        const size_t first_cell_idx = hash_table_splits[i];
        const T* const queries_i = queries + 3 * queries_row_splits[i];
        const size_t num_queries_i =
                queries_row_splits[i + 1] - queries_row_splits[i];

        WriteNeighborsHybrid(
                stream, indices_ptr, distances_ptr, counts_ptr, query_offset,
                hash_table_index, hash_table_cell_splits + first_cell_idx,
                hash_table_size + 1, queries_i, num_queries_i, points,
                inv_voxel_size, radius, max_knn, metric, true);
    }
}
////

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d
