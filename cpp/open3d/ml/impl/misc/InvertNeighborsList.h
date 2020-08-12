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

#include <tbb/parallel_for.h>

#include "open3d/core/Atomic.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace ml {
namespace impl {

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
void InvertNeighborsListCPU(const TIndex* const inp_neighbors_index,
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

    std::vector<uint32_t> tmp_neighbors_count(out_num_queries + 1, 0);

    // count how often an idx appears in inp_neighbors_index
    tbb::parallel_for(tbb::blocked_range<size_t>(0, index_size),
                      [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              TIndex idx = inp_neighbors_index[i];
                              core::AtomicFetchAddRelaxed(
                                      &tmp_neighbors_count[idx + 1], 1);
                          }
                      });

    InclusivePrefixSum(&tmp_neighbors_count[0],
                       &tmp_neighbors_count[tmp_neighbors_count.size()],
                       out_neighbors_row_splits);

    memset(tmp_neighbors_count.data(), 0,
           sizeof(uint32_t) * tmp_neighbors_count.size());

    // fill the new index vector
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, inp_num_queries),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    TIndex query_idx = i;

                    size_t begin_idx = inp_neighbors_row_splits[i];
                    size_t end_idx = inp_neighbors_row_splits[i + 1];
                    for (size_t j = begin_idx; j < end_idx; ++j) {
                        TIndex neighbor_idx = inp_neighbors_index[j];

                        size_t list_offset =
                                out_neighbors_row_splits[neighbor_idx];
                        size_t item_offset = core::AtomicFetchAddRelaxed(
                                &tmp_neighbors_count[neighbor_idx], 1);
                        out_neighbors_index[list_offset + item_offset] =
                                query_idx;

                        if (inp_neighbors_attributes) {
                            TAttr* attr_ptr =
                                    out_neighbors_attributes +
                                    num_attributes_per_neighbor *
                                            (list_offset + item_offset);
                            for (int attr_i = 0;
                                 attr_i < num_attributes_per_neighbor;
                                 ++attr_i) {
                                attr_ptr[attr_i] = inp_neighbors_attributes
                                        [num_attributes_per_neighbor * j +
                                         attr_i];
                            }
                        }
                    }
                }
            });
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
