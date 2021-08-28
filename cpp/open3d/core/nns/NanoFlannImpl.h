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

#include "open3d/core/Atomic.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace core {
namespace nns {

typedef int32_t index_t;

namespace impl {

template <class T>
std::unique_ptr<NanoFlannIndexHolderBase> BuildKdTree(size_t num_points,
                                                      const T *const points,
                                                      size_t dimension) {
    return std::make_unique<NanoFlannIndexHolder<L2, T, index_t>>(
            num_points, dimension, points);
}

template <class T, class OUTPUT_ALLOCATOR>
void KnnSearchCPU(NanoFlannIndexHolderBase *holder,
                  size_t num_queries,
                  const T *const queries,
                  const size_t dimension,
                  int knn,
                  bool ignore_query_point,
                  bool return_distances,
                  OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    index_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, knn * num_queries);
    output_allocator.AllocDistances(&distances_ptr, knn * num_queries);

    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_queries),
                      [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              holder_->index_->knnSearch(
                                      &queries[i * dimension], knn,
                                      &indices_ptr[knn * i],
                                      &distances_ptr[knn * i]);
                          }
                      });
}

template <class T, class OUTPUT_ALLOCATOR>
void RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                     int64_t *query_neighbors_row_splits,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T *const radii,
                     bool ignore_query_point,
                     bool return_distances,
                     nanoflann::SearchParams *params,
                     OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    std::vector<std::vector<index_t>> indices_vec(num_queries);
    std::vector<std::vector<T>> distances_vec(num_queries);

    size_t num_indices = 0;
    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    T radius = radii[i];
                    T radius_squared = radius * radius;
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, *params);
                    ret_matches.resize(num_results);

                    std::vector<index_t> indices_vec_i;
                    std::vector<T> distances_vec_i;
                    for (auto &it : ret_matches) {
                        indices_vec_i.push_back(it.first);
                        distances_vec_i.push_back(it.second);
                    }
                    indices_vec[i] = indices_vec_i;
                    distances_vec[i] = distances_vec_i;
                    query_neighbors_row_splits[i + 1] =
                            static_cast<int64_t>(num_results);

                    AtomicFetchAddRelaxed((uint64_t *)&num_indices,
                                          num_results);
                }
            });

    index_t *indices_ptr;
    T *distances_ptr;

    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);

    query_neighbors_row_splits[0] = 0;
    utility::InclusivePrefixSum(query_neighbors_row_splits + 1,
                                query_neighbors_row_splits + num_queries + 1,
                                query_neighbors_row_splits + 1);

    for (size_t i = 0; i < indices_vec.size(); ++i) {
        int64_t start_idx = query_neighbors_row_splits[i];
        for (size_t j = 0; j < indices_vec[i].size(); ++j) {
            indices_ptr[start_idx + j] = indices_vec[i][j];
            distances_ptr[start_idx + j] = distances_vec[i][j];
        }
    }
}

template <class T, class OUTPUT_ALLOCATOR>
void HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T radius,
                     const int max_knn,
                     bool ignore_query_point,
                     bool return_distances,
                     nanoflann::SearchParams *params,
                     OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0) {
        index_t *indices_ptr, *counts_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);
        output_allocator.AllocCounts(&counts_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    T radius_squared = radius * radius;
    index_t *indices_ptr, *counts_ptr;
    T *distances_ptr;

    size_t num_indices = static_cast<size_t>(max_knn) * num_queries;
    output_allocator.AllocIndices(&indices_ptr, num_indices);
    output_allocator.AllocDistances(&distances_ptr, num_indices);
    output_allocator.AllocCounts(&counts_ptr, num_queries);

    auto holder_ = static_cast<NanoFlannIndexHolder<L2, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, *params);
                    ret_matches.resize(num_results);

                    index_t count_i = static_cast<index_t>(num_results);
                    count_i = count_i < max_knn ? count_i : max_knn;
                    counts_ptr[i] = count_i;

                    int neighbor_idx = 0;
                    for (auto it = ret_matches.begin();
                         it < ret_matches.end() && neighbor_idx < max_knn;
                         it++, neighbor_idx++) {
                        indices_ptr[i * max_knn + neighbor_idx] = it->first;
                        distances_ptr[i * max_knn + neighbor_idx] = it->second;
                    }

                    while (neighbor_idx < max_knn) {
                        indices_ptr[i * max_knn + neighbor_idx] = -1;
                        distances_ptr[i * max_knn + neighbor_idx] = 0;
                        neighbor_idx += 1;
                    }
                }
            });
}

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d