// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/parallel_for.h>

#include <set>

#include "open3d/core/Atomic.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Eigen.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace core {
namespace nns {
namespace impl {

namespace {

/// Builds a spatial hash table for a fixed radius search of 3D points.
///
/// \param num_points    The number of points.
///
/// \param points    The array of 3D points. This array may be splitted into
///        multiple batch items by defining points_row_splits_size accordingly.
///
/// \param radius    The radius that will be used for searching.
///
/// \param points_row_splits_size    The size of the points_row_splits array.
///        The size of the array is batch_size+1.
///
/// \param points_row_splits    Defines the start and end of the points in
///        each batch item. The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param hash_table_splits    Array defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param hash_table_cell_splits    This is an output array storing the start
///        of each hash table entry. The size of this array defines the size of
///        the hash table.
///        The hash table size is hash_table_cell_splits_size - 1.
///
/// \param hash_table_index    This is an output array storing the values of the
///        hash table, which are the indices to the points. The size of the
///        array must be equal to the number of points.
///
template <class T>
void BuildSpatialHashTableCPU(const size_t num_points,
                              const T* const points,
                              const T radius,
                              const size_t points_row_splits_size,
                              const int64_t* points_row_splits,
                              const uint32_t* hash_table_splits,
                              const size_t hash_table_cell_splits_size,
                              uint32_t* hash_table_cell_splits,
                              uint32_t* hash_table_index) {
    using namespace open3d::utility;
    typedef MiniVec<T, 3> Vec3_t;

    const int batch_size = points_row_splits_size - 1;
    const T voxel_size = 2 * radius;
    const T inv_voxel_size = 1 / voxel_size;

    memset(&hash_table_cell_splits[0], 0,
           sizeof(uint32_t) * hash_table_cell_splits_size);

    // compute number of points that map to each hash
    for (int i = 0; i < batch_size; ++i) {
        const size_t hash_table_size =
                hash_table_splits[i + 1] - hash_table_splits[i];
        const size_t first_cell_idx = hash_table_splits[i];
        tbb::parallel_for(
                tbb::blocked_range<int64_t>(points_row_splits[i],
                                            points_row_splits[i + 1]),
                [&](const tbb::blocked_range<int64_t>& r) {
                    for (int64_t i = r.begin(); i != r.end(); ++i) {
                        Vec3_t pos(points + 3 * i);

                        auto voxel_index =
                                ComputeVoxelIndex(pos, inv_voxel_size);
                        size_t hash =
                                SpatialHash(voxel_index) % hash_table_size;

                        // note the +1 because we want the first
                        // element to be 0
                        core::AtomicFetchAddRelaxed(
                                &hash_table_cell_splits[first_cell_idx + hash +
                                                        1],
                                1);
                    }
                });
    }
    InclusivePrefixSum(&hash_table_cell_splits[0],
                       &hash_table_cell_splits[hash_table_cell_splits_size],
                       &hash_table_cell_splits[0]);

    std::vector<uint32_t> count_tmp(hash_table_cell_splits_size - 1, 0);

    // now compute the indices for hash_table_index
    for (int i = 0; i < batch_size; ++i) {
        const size_t hash_table_size =
                hash_table_splits[i + 1] - hash_table_splits[i];
        const size_t first_cell_idx = hash_table_splits[i];
        tbb::parallel_for(
                tbb::blocked_range<size_t>(points_row_splits[i],
                                           points_row_splits[i + 1]),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        Vec3_t pos(points + 3 * i);

                        auto voxel_index =
                                ComputeVoxelIndex(pos, inv_voxel_size);
                        size_t hash =
                                SpatialHash(voxel_index) % hash_table_size;

                        hash_table_index
                                [hash_table_cell_splits[hash + first_cell_idx] +
                                 core::AtomicFetchAddRelaxed(
                                         &count_tmp[hash + first_cell_idx],
                                         1)] = i;
                    }
                });
    }
}

/// Vectorized distance computation. This function computes the distance to
/// \p p for a fixed number of points.
///
/// \tparam METRIC    The distance metric. One of L1, L2, Linf.
///
/// \tparam TDerived    Eigen Array with shape 3x1.
/// \tparam VECSIZE    The vector size. Equals the number of points for which
///         to compute the distances.
///
/// \param p    A 3D point.
/// \param x    x coordinates of 3D points.
/// \param y    y coordinates of 3D points.
/// \param z    z coordinates of 3D points.
///
/// \return Returns a vector of size \p VECSIZE with the distances to \p p.
///         Note that for the metric L2 the result contains the squared
///         distances to avoid the sqrt.
template <int METRIC, class TDerived, int VECSIZE>
Eigen::Array<typename TDerived::Scalar, VECSIZE, 1> NeighborsDist(
        const Eigen::ArrayBase<TDerived>& p,
        const Eigen::Array<typename TDerived::Scalar, VECSIZE, 3>& points) {
    typedef Eigen::Array<typename TDerived::Scalar, VECSIZE, 1> VecN_t;
    VecN_t dist;

    dist.setZero();
    if (METRIC == Linf) {
        dist = (points.rowwise() - p.transpose()).abs().rowwise().maxCoeff();
    } else if (METRIC == L1) {
        dist = (points.rowwise() - p.transpose()).abs().rowwise().sum();
    } else {
        dist = (points.rowwise() - p.transpose()).square().rowwise().sum();
    }
    return dist;
}

/// Implementation of FixedRadiusSearchCPU with template params for metrics
/// and boolean options.
template <class T,
          class TIndex,
          class OUTPUT_ALLOCATOR,
          int METRIC,
          bool IGNORE_QUERY_POINT,
          bool RETURN_DISTANCES>
void _FixedRadiusSearchCPU(int64_t* query_neighbors_row_splits,
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
                           const size_t hash_table_cell_splits_size,
                           const uint32_t* const hash_table_cell_splits,
                           const uint32_t* const hash_table_index,
                           OUTPUT_ALLOCATOR& output_allocator) {
    using namespace open3d::utility;

// number of elements for vectorization
#define VECSIZE 8
    typedef MiniVec<T, 3> Vec3_t;
    typedef Eigen::Array<T, VECSIZE, 1> Vec_t;
    typedef Eigen::Array<TIndex, VECSIZE, 1> Veci_t;

    typedef Eigen::Array<T, 3, 1> Pos_t;
    typedef Eigen::Array<T, VECSIZE, 3> Poslist_t;
    typedef Eigen::Array<bool, VECSIZE, 1> Result_t;

    const int batch_size = points_row_splits_size - 1;

    // return empty output arrays if there are no points
    if (num_points == 0 || num_queries == 0) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        TIndex* indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T* distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);

        return;
    }

    // use squared radius for L2 to avoid sqrt
    const T threshold = (METRIC == L2 ? radius * radius : radius);

    const T voxel_size = 2 * radius;
    const T inv_voxel_size = 1 / voxel_size;

    // counts the number of indices we have to return. This is the number of all
    // neighbors we find.
    size_t num_indices = 0;

    // count the number of neighbors for all query points and update num_indices
    // and populate query_neighbors_row_splits with the number of neighbors
    // for each query point
    for (int i = 0; i < batch_size; ++i) {
        const size_t hash_table_size =
                hash_table_splits[i + 1] - hash_table_splits[i];
        const size_t first_cell_idx = hash_table_splits[i];
        tbb::parallel_for(
                tbb::blocked_range<size_t>(queries_row_splits[i],
                                           queries_row_splits[i + 1]),
                [&](const tbb::blocked_range<size_t>& r) {
                    size_t num_indices_local = 0;
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        size_t neighbors_count = 0;

                        Vec3_t pos(queries + i * 3);

                        std::set<size_t> bins_to_visit;

                        auto voxel_index =
                                ComputeVoxelIndex(pos, inv_voxel_size);
                        size_t hash =
                                SpatialHash(voxel_index) % hash_table_size;

                        bins_to_visit.insert(first_cell_idx + hash);

                        for (int dz = -1; dz <= 1; dz += 2)
                            for (int dy = -1; dy <= 1; dy += 2)
                                for (int dx = -1; dx <= 1; dx += 2) {
                                    Vec3_t p =
                                            pos + radius * Vec3_t(T(dx), T(dy),
                                                                  T(dz));
                                    voxel_index = ComputeVoxelIndex(
                                            p, inv_voxel_size);
                                    hash = SpatialHash(voxel_index) %
                                           hash_table_size;
                                    bins_to_visit.insert(first_cell_idx + hash);
                                }

                        Poslist_t xyz;
                        int vec_i = 0;

                        for (size_t bin : bins_to_visit) {
                            size_t begin_idx = hash_table_cell_splits[bin];
                            size_t end_idx = hash_table_cell_splits[bin + 1];

                            for (size_t j = begin_idx; j < end_idx; ++j) {
                                uint32_t idx = hash_table_index[j];
                                if (IGNORE_QUERY_POINT) {
                                    if (points[idx * 3 + 0] == pos[0] &&
                                        points[idx * 3 + 1] == pos[1] &&
                                        points[idx * 3 + 2] == pos[2])
                                        continue;
                                }
                                xyz(vec_i, 0) = points[idx * 3 + 0];
                                xyz(vec_i, 1) = points[idx * 3 + 1];
                                xyz(vec_i, 2) = points[idx * 3 + 2];
                                ++vec_i;
                                if (VECSIZE == vec_i) {
                                    Pos_t pos_arr(pos[0], pos[1], pos[2]);
                                    Vec_t dist = NeighborsDist<METRIC, Pos_t,
                                                               VECSIZE>(pos_arr,
                                                                        xyz);
                                    Result_t test_result = dist <= threshold;
                                    neighbors_count += test_result.count();
                                    vec_i = 0;
                                }
                            }
                        }
                        // process the tail
                        if (vec_i) {
                            Pos_t pos_arr(pos[0], pos[1], pos[2]);
                            Vec_t dist = NeighborsDist<METRIC, Pos_t, VECSIZE>(
                                    pos_arr, xyz);
                            Result_t test_result = dist <= threshold;
                            for (int k = 0; k < vec_i; ++k) {
                                neighbors_count += int(test_result(k));
                            }
                            vec_i = 0;
                        }
                        num_indices_local += neighbors_count;
                        // note the +1
                        query_neighbors_row_splits[i + 1] = neighbors_count;
                    }

                    core::AtomicFetchAddRelaxed((uint64_t*)&num_indices,
                                                num_indices_local);
                });
    }

    // Allocate output arrays
    // output for the indices to the neighbors
    TIndex* indices_ptr;
    output_allocator.AllocIndices(&indices_ptr, num_indices);

    // output for the distances
    T* distances_ptr;
    if (RETURN_DISTANCES)
        output_allocator.AllocDistances(&distances_ptr, num_indices);
    else
        output_allocator.AllocDistances(&distances_ptr, 0);

    query_neighbors_row_splits[0] = 0;
    InclusivePrefixSum(query_neighbors_row_splits + 1,
                       query_neighbors_row_splits + num_queries + 1,
                       query_neighbors_row_splits + 1);

    // now populate the indices_ptr and distances_ptr array
    for (int i = 0; i < batch_size; ++i) {
        const size_t hash_table_size =
                hash_table_splits[i + 1] - hash_table_splits[i];
        const size_t first_cell_idx = hash_table_splits[i];
        tbb::parallel_for(
                tbb::blocked_range<size_t>(queries_row_splits[i],
                                           queries_row_splits[i + 1]),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i != r.end(); ++i) {
                        size_t neighbors_count = 0;

                        size_t indices_offset = query_neighbors_row_splits[i];

                        Vec3_t pos(queries[i * 3 + 0], queries[i * 3 + 1],
                                   queries[i * 3 + 2]);

                        std::set<size_t> bins_to_visit;

                        auto voxel_index =
                                ComputeVoxelIndex(pos, inv_voxel_size);
                        size_t hash =
                                SpatialHash(voxel_index) % hash_table_size;

                        bins_to_visit.insert(first_cell_idx + hash);

                        for (int dz = -1; dz <= 1; dz += 2)
                            for (int dy = -1; dy <= 1; dy += 2)
                                for (int dx = -1; dx <= 1; dx += 2) {
                                    Vec3_t p =
                                            pos + radius * Vec3_t(T(dx), T(dy),
                                                                  T(dz));
                                    voxel_index = ComputeVoxelIndex(
                                            p, inv_voxel_size);
                                    hash = SpatialHash(voxel_index) %
                                           hash_table_size;
                                    bins_to_visit.insert(first_cell_idx + hash);
                                }

                        Poslist_t xyz;
                        Veci_t idx_vec;
                        int vec_i = 0;

                        for (size_t bin : bins_to_visit) {
                            size_t begin_idx = hash_table_cell_splits[bin];
                            size_t end_idx = hash_table_cell_splits[bin + 1];

                            for (size_t j = begin_idx; j < end_idx; ++j) {
                                int64_t idx = hash_table_index[j];
                                if (IGNORE_QUERY_POINT) {
                                    if (points[idx * 3 + 0] == pos[0] &&
                                        points[idx * 3 + 1] == pos[1] &&
                                        points[idx * 3 + 2] == pos[2])
                                        continue;
                                }
                                xyz(vec_i, 0) = points[idx * 3 + 0];
                                xyz(vec_i, 1) = points[idx * 3 + 1];
                                xyz(vec_i, 2) = points[idx * 3 + 2];
                                idx_vec(vec_i) = idx;
                                ++vec_i;
                                if (VECSIZE == vec_i) {
                                    Pos_t pos_arr(pos[0], pos[1], pos[2]);
                                    Vec_t dist = NeighborsDist<METRIC, Pos_t,
                                                               VECSIZE>(pos_arr,
                                                                        xyz);
                                    Result_t test_result = dist <= threshold;
                                    for (int k = 0; k < vec_i; ++k) {
                                        if (test_result(k)) {
                                            indices_ptr[indices_offset +
                                                        neighbors_count] =
                                                    idx_vec[k];
                                            if (RETURN_DISTANCES) {
                                                distances_ptr[indices_offset +
                                                              neighbors_count] =
                                                        dist[k];
                                            }
                                        }
                                        neighbors_count += int(test_result(k));
                                    }
                                    vec_i = 0;
                                }
                            }
                        }
                        // process the tail
                        if (vec_i) {
                            Pos_t pos_arr(pos[0], pos[1], pos[2]);
                            Vec_t dist = NeighborsDist<METRIC, Pos_t, VECSIZE>(
                                    pos_arr, xyz);
                            Result_t test_result = dist <= threshold;
                            for (int k = 0; k < vec_i; ++k) {
                                if (test_result(k)) {
                                    indices_ptr[indices_offset +
                                                neighbors_count] = idx_vec[k];
                                    if (RETURN_DISTANCES) {
                                        distances_ptr[indices_offset +
                                                      neighbors_count] =
                                                dist[k];
                                    }
                                }
                                neighbors_count += int(test_result(k));
                            }
                            vec_i = 0;
                        }
                    }
                });
    }
#undef VECSIZE
}

}  // namespace

/// Fixed radius search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param query_neighbors_row_splits    This is the output pointer for the
///        prefix sum. The length of this array is \p num_queries + 1.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the 3D point positions. This must be the array
///        that was used for building the spatial hash table.
///
/// \param num_queries    The number of query points.
///
/// \param queries    Array with the 3D query positions. This may be the same
///                   array as \p points.
///
/// \param radius    The search radius.
///
/// \param points_row_splits_size    The size of the points_row_splits array.
///        The size of the array is batch_size+1.
///
/// \param points_row_splits    Defines the start and end of the points in each
///        batch item. The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param queries_row_splits_size    The size of the queries_row_splits array.
///        The size of the array is batch_size+1.
///
/// \param queries_row_splits    Defines the start and end of the queries in
/// each
///        batch item. The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_queries]
///
/// \param hash_table_splits    Array defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCPU. The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCPU. This is array storing the values of the
///        hash table, which are the indices to the points. The size of the
///        array must be equal to the number of points.
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
///
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement functions
///         AllocIndices(int64_t** ptr, size_t size) and
///         AllocDistances(T** ptr, size_t size). Both functions should
///         allocate memory and return a pointer to that memory in ptr.
///         Argument size specifies the size of the array as the number of
///         elements. Both functions must accept the argument size==0.
///         In this case ptr does not need to be set.
///
template <class T, class TIndex, class OUTPUT_ALLOCATOR>
void FixedRadiusSearchCPU(int64_t* query_neighbors_row_splits,
                          const size_t num_points,
                          const T* const points,
                          const size_t num_queries,
                          const T* const queries,
                          const T radius,
                          const size_t points_row_splits_size,
                          const int64_t* const points_row_splits,
                          const size_t queries_row_splits_size,
                          const int64_t* const queries_row_splits,
                          const uint32_t* const hash_table_splits,
                          const size_t hash_table_cell_splits_size,
                          const uint32_t* const hash_table_cell_splits,
                          const uint32_t* const hash_table_index,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          OUTPUT_ALLOCATOR& output_allocator) {
    // Dispatch all template parameter combinations

#define FN_PARAMETERS                                                       \
    query_neighbors_row_splits, num_points, points, num_queries, queries,   \
            radius, points_row_splits_size, points_row_splits,              \
            queries_row_splits_size, queries_row_splits, hash_table_splits, \
            hash_table_cell_splits_size, hash_table_cell_splits,            \
            hash_table_index, output_allocator

#define CALL_TEMPLATE(METRIC, IGNORE_QUERY_POINT, RETURN_DISTANCES)     \
    if (METRIC == metric && IGNORE_QUERY_POINT == ignore_query_point && \
        RETURN_DISTANCES == return_distances)                           \
        _FixedRadiusSearchCPU<T, TIndex, OUTPUT_ALLOCATOR, METRIC,      \
                              IGNORE_QUERY_POINT, RETURN_DISTANCES>(    \
                FN_PARAMETERS);

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

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d
