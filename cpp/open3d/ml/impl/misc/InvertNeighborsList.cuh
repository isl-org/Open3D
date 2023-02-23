// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cub/cub.cuh>

#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/utility/Helper.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {

/// Kernel for CountNeighborsCUDA
template <class T>
__global__ void CountNeighborsCUDAKernel(uint32_t* __restrict__ count,
                                         size_t count_size,
                                         const T* const __restrict__ indices,
                                         size_t indices_size) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= indices_size) return;

    T idx = indices[i];
    atomicAdd(&count[idx], 1);
}

/// Counts the number of neighbors per index value.
///
/// \param count    Output array for counting.
///
/// \param count_size    The size of count. This is the number of queries
///        with respect to the inverted neighbors list.
///
/// \param indices    Array of indices. This is the nested list of neighbors.
///
/// \param indices_size    Size of indices.
///
template <class T>
void CountNeighborsCUDA(const cudaStream_t& stream,
                        uint32_t* __restrict__ count,
                        size_t count_size,
                        const T* const __restrict__ indices,
                        size_t indices_size) {
    using namespace open3d::utility;

    cudaMemsetAsync(count, 0, sizeof(uint32_t) * count_size, stream);

    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = DivUp(indices_size, block.x);

    if (grid.x) {
        CountNeighborsCUDAKernel<T><<<grid, block, 0, stream>>>(
                count, count_size, indices, indices_size);
    }
}

/// Kernel for FillNeighborsIndexAndAttributesCUDA.
template <class TIndex, class TAttr, bool FILL_ATTRIBUTES>
__global__ void FillNeighborsIndexAndAttributesCUDAKernel(
        uint32_t* __restrict__ count,
        size_t count_size,
        TIndex* __restrict__ out_neighbors_index,
        TAttr* __restrict__ out_neighbors_attributes,
        const TIndex* const __restrict__ inp_neighbors_index,
        const TAttr* const __restrict__ inp_neighbors_attributes,
        int num_attributes_per_neighbor,
        size_t indices_size,
        const int64_t* const __restrict__ inp_neighbors_row_splits,
        size_t inp_num_queries,
        const int64_t* const __restrict__ out_neighbors_row_splits,
        size_t out_num_queries) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= inp_num_queries) return;

    TIndex query_idx = i;

    size_t begin_idx = inp_neighbors_row_splits[i];
    size_t end_idx = inp_neighbors_row_splits[i + 1];

    for (size_t j = begin_idx; j < end_idx; ++j) {
        TIndex neighbor_idx = inp_neighbors_index[j];

        size_t list_offset = out_neighbors_row_splits[neighbor_idx];
        size_t item_offset = atomicAdd(&count[neighbor_idx], 1);
        out_neighbors_index[list_offset + item_offset] = query_idx;

        if (FILL_ATTRIBUTES) {
            TAttr* attr_ptr =
                    out_neighbors_attributes +
                    num_attributes_per_neighbor * (list_offset + item_offset);
            for (int attr_i = 0; attr_i < num_attributes_per_neighbor;
                 ++attr_i) {
                attr_ptr[attr_i] =
                        inp_neighbors_attributes[num_attributes_per_neighbor *
                                                         j +
                                                 attr_i];
            }
        }
    }
}

/// Fills in the indices and attributes for the inverted neighbors list
///
/// \param count    Temporary array for index computation
///
/// \param count_size    The size of count. This is the number of queries
///        with respect to the inverted neighbors list.
///
/// See InvertNeighborsListCUDA for the meaning of the remaining parameters.
///
template <class TIndex, class TAttr>
void FillNeighborsIndexAndAttributesCUDA(
        const cudaStream_t& stream,
        uint32_t* __restrict__ count,
        size_t count_size,
        TIndex* __restrict__ out_neighbors_index,
        TAttr* __restrict__ out_neighbors_attributes,
        const TIndex* const __restrict__ inp_neighbors_index,
        const TAttr* const __restrict__ inp_neighbors_attributes,
        const int num_attributes_per_neighbor,
        size_t index_size,
        const int64_t* const __restrict__ inp_neighbors_row_splits,
        size_t inp_num_queries,
        const int64_t* const __restrict__ out_neighbors_row_splits,
        size_t out_num_queries) {
    using namespace open3d::utility;

    cudaMemsetAsync(count, 0, sizeof(uint32_t) * count_size, stream);

    const int BLOCKSIZE = 128;
    dim3 block(BLOCKSIZE, 1, 1);
    dim3 grid(0, 1, 1);
    grid.x = DivUp(inp_num_queries, block.x);

    if (grid.x) {
        if (inp_neighbors_attributes) {
            FillNeighborsIndexAndAttributesCUDAKernel<TIndex, TAttr, true>
                    <<<grid, block, 0, stream>>>(
                            count, count_size, out_neighbors_index,
                            out_neighbors_attributes, inp_neighbors_index,
                            inp_neighbors_attributes,
                            num_attributes_per_neighbor, index_size,
                            inp_neighbors_row_splits, inp_num_queries,
                            out_neighbors_row_splits, out_num_queries);
        } else {
            FillNeighborsIndexAndAttributesCUDAKernel<TIndex, TAttr, false>
                    <<<grid, block, 0, stream>>>(
                            count, count_size, out_neighbors_index,
                            out_neighbors_attributes, inp_neighbors_index,
                            inp_neighbors_attributes,
                            num_attributes_per_neighbor, index_size,
                            inp_neighbors_row_splits, inp_num_queries,
                            out_neighbors_row_splits, out_num_queries);
        }
    }
}

}  // namespace

/// Inverts a neighbors list, which is a tuple of the form
/// (neighbors_index, neighbors_row_splits, neighbors_attributes).
/// neighbors_index is a nested list of indices to the neighbors. Each entry
/// defines an edge between two indices (points).
/// The neighbors_row_splits defines the start and end of each sublist.
/// neighbors_attributes is an optional array of attributes for each entry in
/// neighbors_index.
///
/// Example: The neighbors for point cloud A (3 points) in point cloud B
/// (2 points) is defined by:
/// - neighbors_index [0 1 0 0]
/// - neighbors_row_splits [0 2 3 4]
/// - optional neighbors_attributes [0.1 0.2 0.3 0.4] (1 scalar attribute)
///
/// The inverted neighbors list is then the neighbors for point cloud B in A
/// - neighbors_index [0 1 2 0]
/// - neighbors_row_splits [0 3 4]
/// - optional neighbors_attributes [0.1 0.3 0.4 0.2]
///
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
/// \param inp_neighbors_index    The nested list of neighbor indices.
///
/// \param inp_neighbors_attributes    The array of attributes for each entry
///        in \p inp_neighbors_index. This is optional and can be set to null.
///
/// \param num_attributes_per_neighbor    The number of scalar attributes for
///        each entry in \p inp_neighbors_index.
///
/// \param inp_neighbors_row_splits    Defines the start and end of the
///        sublists in \p inp_neighbors_index. This is an exclusive prefix sum
///        with 0 as the first element and the length of
///        \p inp_neighbors_index as last element.
///        The size is \p inp_num_queries + 1
///
/// \param inp_num_queries    The number of queries.
///
/// \param out_neighbors_index    The inverted neighbors_index list with the
///        same size as \p inp_neighbors_index .
///
/// \param out_neighbors_attributes    The inverted array of attributes with
///        the same size as \p inp_neighbors_attributes .
///
/// \param index_size    This is the size of \p inp_neighbors_index and
///        \p out_neighbors_index, both have the same size.
///
/// \param out_neighbors_row_splits   The prefix sum which defines the start
///        and end of the sublists in \p out_neighbors_index.
///
/// \param out_num_queries    The number of queries with respect to the
///        inverted neighbors list.
///
template <class TIndex, class TAttr>
void InvertNeighborsListCUDA(const cudaStream_t& stream,
                             void* temp,
                             size_t& temp_size,
                             int texture_alignment,
                             const TIndex* const inp_neighbors_index,
                             const TAttr* const inp_neighbors_attributes,
                             const int num_attributes_per_neighbor,
                             const int64_t* const inp_neighbors_row_splits,
                             const size_t inp_num_queries,
                             TIndex* out_neighbors_index,
                             TAttr* out_neighbors_attributes,
                             const size_t index_size,
                             int64_t* out_neighbors_row_splits,
                             const size_t out_num_queries) {
    using namespace open3d::utility;

    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case pointer alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    // Object for allocating memory within the temporary memory
    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

    // Reserve memory for counting the neighbors
    std::pair<uint32_t*, size_t> tmp_neighbors_count =
            mem_temp.Alloc<uint32_t>(out_num_queries);

    if (!get_temp_size) {
        CountNeighborsCUDA(stream, tmp_neighbors_count.first,
                           tmp_neighbors_count.second, inp_neighbors_index,
                           index_size);
    }

    // compute prefix sum
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);
        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                tmp_neighbors_count.first, out_neighbors_row_splits + 1,
                out_num_queries, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

        if (!get_temp_size) {
            // set first element to zero
            cudaMemsetAsync(out_neighbors_row_splits, 0, sizeof(int64_t),
                            stream);
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    tmp_neighbors_count.first, out_neighbors_row_splits + 1,
                    out_num_queries, stream);
        }

        mem_temp.Free(inclusive_scan_temp);
    }
    mem_temp.Free(tmp_neighbors_count);

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }

    if (!get_temp_size) {
        FillNeighborsIndexAndAttributesCUDA(
                stream, tmp_neighbors_count.first, tmp_neighbors_count.second,
                out_neighbors_index, out_neighbors_attributes,
                inp_neighbors_index, inp_neighbors_attributes,
                num_attributes_per_neighbor, index_size,
                inp_neighbors_row_splits, inp_num_queries,
                out_neighbors_row_splits, out_num_queries);
    }
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
