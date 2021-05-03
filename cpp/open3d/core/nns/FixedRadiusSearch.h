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

#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/NeighborSearchCommon.h"

namespace open3d {
namespace core {
namespace nns {

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
template <class T>
void BuildSpatialHashTableCUDA(void* temp,
                               size_t& temp_size,
                               const size_t num_points,
                               const T* const points,
                               const T radius,
                               const size_t points_row_splits_size,
                               const int64_t* points_row_splits,
                               const int64_t* hash_table_splits,
                               const size_t hash_table_cell_splits_size,
                               int64_t* hash_table_cell_splits,
                               int64_t* hash_table_index,
                               int64_t* sorted_point_indices,
                               T* sorted_points);

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
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement functions
///         AllocIndices(int32_t** ptr, size_t size) and
///         AllocDistances(T** ptr, size_t size). Both functions should
///         allocate memory and return a pointer to that memory in ptr.
///         Argument size specifies the size of the array as the number of
///         elements. Both functions must accept the argument size==0.
///         In this case ptr does not need to be set.
template <class T>
void FixedRadiusSearchCUDA(void* temp,
                           size_t& temp_size,
                           int64_t* query_neighbors_row_splits,
                           size_t num_points,
                           const T* const points,
                           const int64_t* const point_indices,
                           size_t num_queries,
                           const T* const queries,
                           const T radius,
                           const size_t points_row_splits_size,
                           const int64_t* const points_row_splits,
                           const size_t queries_row_splits_size,
                           const int64_t* const queries_row_splits,
                           const int64_t* const hash_table_splits,
                           size_t hash_table_cell_splits_size,
                           const int64_t* const hash_table_cell_splits,
                           const int64_t* const hash_table_index,
                           NeighborSearchAllocator<T>& output_allocator);

/// Hybrid search. This function computes a list of neighbor indices and
/// distances for each query point. The lists are stored linearly per each query
/// point. If the neighbors within radius is not enough, neighbor indices and
/// distances are padded with -1.
///
/// All pointer arguments point to device memory unless stated otherwise.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
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
/// \param max_knn    The maximum number of neighbors to search for each query.
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
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement functions
///         AllocIndices(int32_t** ptr, size_t size) and
///         AllocDistances(T** ptr, size_t size). Both functions should
///         allocate memory and return a pointer to that memory in ptr.
///         Argument size specifies the size of the array as the number of
///         elements. Both functions must accept the argument size==0.
///         In this case ptr does not need to be set.
template <class T>
void HybridSearchCUDA(size_t num_points,
                      const T* const points,
                      const int64_t* const point_indices,
                      size_t num_queries,
                      const T* const queries,
                      const T radius,
                      const int max_knn,
                      const size_t points_row_splits_size,
                      const int64_t* const points_row_splits,
                      const size_t queries_row_splits_size,
                      const int64_t* const queries_row_splits,
                      const int64_t* const hash_table_splits,
                      size_t hash_table_cell_splits_size,
                      const int64_t* const hash_table_cell_splits,
                      const int64_t* const hash_table_index,
                      NeighborSearchAllocator<T>& output_allocator);

/// This function sorts a list of neighbor indices and distances in
/// descending order of distance. It is based-on moderngpu's merge sort.
///
/// All pointer arguments point to device memory unless stated otherwise.
///
/// \tparam T    Floating-point data type for the point positions.
///
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr
///
/// \param num_indices    The number of indices to sort.
///
/// \param num_segments    The number of segments to sort.
///
/// \param query_neighbors_row_splits    This is the output pointer for the
///        prefix sum. The length of this array is \p num_queries + 1.
///
/// \param indices    Pointer to indices.
///
/// \param distances    Pointer to distances.
///
template <class T>
void SortPairs(int64_t num_indices,
               int64_t num_segments,
               const int64_t* query_neighbors_row_splits,
               int64_t* indices,
               T* distances);
}  // namespace nns
}  // namespace core
}  // namespace open3d
