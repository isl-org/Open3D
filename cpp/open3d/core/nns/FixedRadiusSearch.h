// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include <cub/cub.cuh>

// #include "open3d/core/nns/MemoryAllocation.h"
// #include "open3d/core/nns/NeighborSearchCommon.h"
// #include "open3d/utility/Helper.h"
// #include "open3d/utility/MiniVec.h"

using namespace open3d::utility;

namespace open3d {
namespace ml {
namespace impl {

/// Builds a spatial hash table for a fixed radius search of 3D points.
///
/// All pointer arguments point to device memory unless stated otherwise.
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr
///
/// \param texture_alignment    The texture alignment in bytes. This is used
///        for allocating segments within the temporary memory.
///
/// \param num_points    The number of points.
///
/// \param points    The array of 3D points.
///
/// \param radius    The radius that will be used for searching.
///
/// \param points_row_splits_size    The size of the points_row_splits array.
///        The size of the array is batch_size+1.
///
/// \param points_row_splits    This pointer points to host memory.
///        Defines the start and end of the points in each batch item.
///        The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param hash_table_splits    This pointer points to host memory.
///        Array defining the start and end the hash table
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
template <class TReal, class TIndex>
void BuildSpatialHashTableCUDA(const cudaStream_t& stream,
                               void* temp,
                               size_t& temp_size,
                               int texture_alignment,
                               const size_t num_points,
                               const TReal* const points,
                               const TReal radius,
                               const size_t points_row_splits_size,
                               const int64_t* points_row_splits,
                               const TIndex* hash_table_splits,
                               const size_t hash_table_cell_splits_size,
                               TIndex* hash_table_cell_splits,
                               TIndex* hash_table_index);
//                                 {
//     const bool get_temp_size = !temp;

//     if (get_temp_size) {
//         temp = (char*)1;  // worst case pointer alignment
//         temp_size = std::numeric_limits<int64_t>::max();
//     }

//     MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

//     std::pair<TIndex*, size_t> count_tmp =
//             mem_temp.Alloc<TIndex>(hash_table_cell_splits_size);

//     const int batch_size = points_row_splits_size - 1;
//     const TReal voxel_size = 2 * radius;
//     const TReal inv_voxel_size = 1 / voxel_size;

//     // count number of points per hash entry
//     if (!get_temp_size) {
//         cudaMemsetAsync(count_tmp.first, 0, sizeof(TIndex) * count_tmp.second,
//                         stream);

//         for (int i = 0; i < batch_size; ++i) {
//             const size_t hash_table_size =
//                     hash_table_splits[i + 1] - hash_table_splits[i];
//             const size_t first_cell_idx = hash_table_splits[i];
//             const size_t num_points_i =
//                     points_row_splits[i + 1] - points_row_splits[i];
//             const TReal* const points_i = points + 3 * points_row_splits[i];

//             CountHashTableEntries(stream, count_tmp.first + first_cell_idx,
//                                   hash_table_size + 1, inv_voxel_size, points_i,
//                                   num_points_i);
//         }
//     }

//     // compute prefix sum of the hash entry counts and store in
//     // hash_table_cell_splits
//     {
//         std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);
//         cub::DeviceScan::InclusiveSum(inclusive_scan_temp.first,
//                                       inclusive_scan_temp.second,
//                                       count_tmp.first, hash_table_cell_splits,
//                                       count_tmp.second, stream);

//         inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

//         if (!get_temp_size) {
//             cub::DeviceScan::InclusiveSum(
//                     inclusive_scan_temp.first, inclusive_scan_temp.second,
//                     count_tmp.first, hash_table_cell_splits, count_tmp.second,
//                     stream);
//         }

//         mem_temp.Free(inclusive_scan_temp);
//     }

//     // now compute the global indices which allows us to lookup the point index
//     // for the entries in the hash cell
//     if (!get_temp_size) {
//         for (int i = 0; i < batch_size; ++i) {
//             const size_t hash_table_size =
//                     hash_table_splits[i + 1] - hash_table_splits[i];
//             const size_t first_cell_idx = hash_table_splits[i];
//             const size_t points_start_idx = points_row_splits[i];
//             const size_t points_end_idx = points_row_splits[i + 1];
//             ComputePointIndexTable(stream, hash_table_index, count_tmp.first,
//                                    hash_table_cell_splits + first_cell_idx,
//                                    hash_table_size + 1, inv_voxel_size, points,
//                                    points_start_idx, points_end_idx);
//         }
//     }

//     mem_temp.Free(count_tmp);

//     if (get_temp_size) {
//         // return the memory peak as the required temporary memory size.
//         temp_size = mem_temp.MaxUsed();
//         return;
//     }
// }

/// Fixed radius search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// All pointer arguments point to device memory unless stated otherwise.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr
///
/// \param texture_alignment    The texture alignment in bytes. This is used
///        for allocating segments within the temporary memory.
///
/// \param query_neighbors_row_splits    This is the output pointer for the
///        prefix sum. The length of this array is \p num_queries + 1.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the 3D point positions. This may be the same
///        array as \p queries.
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
/// \param points_row_splits    This pointer points to host memory.
///        Defines the start and end of the points in each batch item.
///        The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_points]
///
/// \param queries_row_splits_size    The size of the queries_row_splits array.
///        The size of the array is batch_size+1.
///
/// \param queries_row_splits    This pointer points to host memory.
///        Defines the start and end of the queries in each batch item.
///        The size of the array is batch_size+1. If there is
///        only 1 batch item then this array is [0, num_queries]
///
/// \param hash_table_splits    This pointer points to host memory.
///        Array defining the start and end the hash table
///        for each batch item. This is [0, number of cells] if there is only
///        1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
///
/// \param hash_table_cell_splits_size    This is the length of the
///        hash_table_cell_splits array.
///
/// \param hash_table_cell_splits    This is an output of the function
///        BuildSpatialHashTableCUDA. The row splits array describing the start
///        and end of each cell.
///
/// \param hash_table_index    This is an output of the function
///        BuildSpatialHashTableCUDA. This is array storing the values of the
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
///         AllocIndices(int32_t** ptr, size_t size) and
///         AllocDistances(T** ptr, size_t size). Both functions should
///         allocate memory and return a pointer to that memory in ptr.
///         Argument size specifies the size of the array as the number of
///         elements. Both functions must accept the argument size==0.
///         In this case ptr does not need to be set.
///
template <class T, class OUTPUT_ALLOCATOR>
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
                           OUTPUT_ALLOCATOR& output_allocator);
//                             {
//     const bool get_temp_size = !temp;

//     if (get_temp_size) {
//         temp = (char*)1;  // worst case pointer alignment
//         temp_size = std::numeric_limits<int64_t>::max();
//     }

//     // return empty output arrays if there are no points
//     if ((0 == num_points || 0 == num_queries) && !get_temp_size) {
//         cudaMemsetAsync(query_neighbors_row_splits, 0,
//                         sizeof(int64_t) * (num_queries + 1), stream);
//         int32_t* indices_ptr;
//         output_allocator.AllocIndices(&indices_ptr, 0);

//         T* distances_ptr;
//         output_allocator.AllocDistances(&distances_ptr, 0);

//         return;
//     }

//     MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

//     const int batch_size = points_row_splits_size - 1;
//     const T voxel_size = 2 * radius;
//     const T inv_voxel_size = 1 / voxel_size;

//     std::pair<uint32_t*, size_t> query_neighbors_count =
//             mem_temp.Alloc<uint32_t>(num_queries);

//     // we need this value to compute the size of the index array
//     if (!get_temp_size) {
//         for (int i = 0; i < batch_size; ++i) {
//             const size_t hash_table_size =
//                     hash_table_splits[i + 1] - hash_table_splits[i];
//             const size_t first_cell_idx = hash_table_splits[i];
//             const size_t queries_start_idx = queries_row_splits[i];
//             const T* const queries_i = queries + 3 * queries_row_splits[i];
//             const size_t num_queries_i =
//                     queries_row_splits[i + 1] - queries_row_splits[i];

//             CountNeighbors(
//                     stream, query_neighbors_count.first + queries_start_idx,
//                     hash_table_index, hash_table_cell_splits + first_cell_idx,
//                     hash_table_size + 1, queries_i, num_queries_i, points,
//                     num_points, inv_voxel_size, radius, metric,
//                     ignore_query_point);
//         }
//     }

//     // we need this value to compute the size of the index array
//     int64_t last_prefix_sum_entry = 0;
//     {
//         std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);
//         cub::DeviceScan::InclusiveSum(
//                 inclusive_scan_temp.first, inclusive_scan_temp.second,
//                 query_neighbors_count.first, query_neighbors_row_splits + 1,
//                 num_queries, stream);

//         inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

//         if (!get_temp_size) {
//             // set first element to zero
//             cudaMemsetAsync(query_neighbors_row_splits, 0, sizeof(int64_t),
//                             stream);
//             cub::DeviceScan::InclusiveSum(
//                     inclusive_scan_temp.first, inclusive_scan_temp.second,
//                     query_neighbors_count.first, query_neighbors_row_splits + 1,
//                     num_queries, stream);

//             // get the last value
//             cudaMemcpyAsync(&last_prefix_sum_entry,
//                             query_neighbors_row_splits + num_queries,
//                             sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
//             // wait for the async copies
//             while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
//             }
//         }
//         mem_temp.Free(inclusive_scan_temp);
//     }

//     mem_temp.Free(query_neighbors_count);

//     if (get_temp_size) {
//         // return the memory peak as the required temporary memory size.
//         temp_size = mem_temp.MaxUsed();
//         return;
//     }

//     // allocate the output array for the neighbor indices
//     const size_t num_indices = last_prefix_sum_entry;
//     int32_t* indices_ptr;
//     output_allocator.AllocIndices(&indices_ptr, num_indices);

//     T* distances_ptr;
//     if (return_distances)
//         output_allocator.AllocDistances(&distances_ptr, num_indices);
//     else
//         output_allocator.AllocDistances(&distances_ptr, 0);

//     if (!get_temp_size) {
//         for (int i = 0; i < batch_size; ++i) {
//             const size_t hash_table_size =
//                     hash_table_splits[i + 1] - hash_table_splits[i];
//             const size_t first_cell_idx = hash_table_splits[i];
//             const T* const queries_i = queries + 3 * queries_row_splits[i];
//             const size_t num_queries_i =
//                     queries_row_splits[i + 1] - queries_row_splits[i];

//             WriteNeighborsIndicesAndDistances(
//                     stream, indices_ptr, distances_ptr,
//                     query_neighbors_row_splits + queries_row_splits[i],
//                     hash_table_index, hash_table_cell_splits + first_cell_idx,
//                     hash_table_size + 1, queries_i, num_queries_i, points,
//                     num_points, inv_voxel_size, radius, metric,
//                     ignore_query_point, return_distances);
//         }
//     }
// }

}  // namespace impl
}  // namespace ml
}  // namespace open3d
