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

#include <mutex>
#include <nanoflann.hpp>

#include "open3d/core/Atomic.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace core {
namespace nns {

typedef int32_t index_t;

/// NanoFlann Index Holder.
template <int METRIC, class TReal, class TIndex>
struct NanoFlannIndexHolder : NanoFlannIndexHolderBase {
    /// This class is the Adaptor for connecting Open3D Tensor and NanoFlann.
    struct DataAdaptor {
        DataAdaptor(size_t dataset_size,
                    int dimension,
                    const TReal *const data_ptr)
            : dataset_size_(dataset_size),
              dimension_(dimension),
              data_ptr_(data_ptr) {}

        inline size_t kdtree_get_point_count() const { return dataset_size_; }

        inline TReal kdtree_get_pt(const size_t idx, const size_t dim) const {
            return data_ptr_[idx * dimension_ + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const {
            return false;
        }

        size_t dataset_size_ = 0;
        int dimension_ = 0;
        const TReal *const data_ptr_;
    };

    /// Adaptor Selector.
    template <int M, typename fake = void>
    struct SelectNanoflannAdaptor {};

    template <typename fake>
    struct SelectNanoflannAdaptor<L2, fake> {
        typedef nanoflann::L2_Adaptor<TReal, DataAdaptor, TReal> adaptor_t;
    };

    template <typename fake>
    struct SelectNanoflannAdaptor<L1, fake> {
        typedef nanoflann::L1_Adaptor<TReal, DataAdaptor, TReal> adaptor_t;
    };

    /// typedef for KDtree.
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            typename SelectNanoflannAdaptor<METRIC>::adaptor_t,
            DataAdaptor,
            -1,
            TIndex>
            KDTree_t;

    NanoFlannIndexHolder(size_t dataset_size,
                         int dimension,
                         const TReal *data_ptr) {
        adaptor_.reset(new DataAdaptor(dataset_size, dimension, data_ptr));
        index_.reset(new KDTree_t(dimension, *adaptor_.get()));
        index_->buildIndex();
    }

    std::unique_ptr<KDTree_t> index_;
    std::unique_ptr<DataAdaptor> adaptor_;
};
namespace impl {

namespace {
template <class T, int METRIC>
void _BuildKdTree(size_t num_points,
                  const T *const points,
                  size_t dimension,
                  NanoFlannIndexHolderBase **holder) {
    *holder = new NanoFlannIndexHolder<METRIC, T, index_t>(num_points,
                                                           dimension, points);
}

template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _KnnSearchCPU(NanoFlannIndexHolderBase *holder,
                   int64_t *query_neighbors_row_splits,
                   size_t num_points,
                   const T *const points,
                   size_t num_queries,
                   const T *const queries,
                   const size_t dimension,
                   int knn,
                   bool ignore_query_point,
                   bool return_distances,
                   OUTPUT_ALLOCATOR &output_allocator) {
    // cast NanoFlannIndexHolder
    auto holder_ =
            static_cast<NanoFlannIndexHolder<METRIC, T, index_t> *>(holder);

    // return empty indices array if there are no points
    if (num_queries == 0 || num_points == 0) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    struct Pair {
        int32_t i, j;
    };
    std::vector<Pair> pairs;

    auto points_equal = [](const T *const p1, const T *const p2,
                           size_t dimension) {
        std::vector<T> p1_vec(p1, p1 + dimension);
        std::vector<T> p2_vec(p2, p2 + dimension);
        return p1_vec == p2_vec;
    };

    auto distance_fn = [](const T *const p1, const T *const p2,
                          size_t dimension) {
        double ret = 0.0;
        for (size_t i = 0; i < dimension; i++) {
            if (METRIC == L2) {
                double dist = p1[i] - p2[i];
                dist = dist * dist;
                ret += dist;
            } else {
                double dist = std::abs(p1[i] - p2[i]);
                ret += dist;
            }
        }
        return ret;
    };

    std::mutex pairs_mutex;
    std::vector<uint32_t> neighbors_count(num_queries, 0);

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<Pair> pairs_private;

                std::vector<index_t> result_indices(knn);
                std::vector<T> result_distances(knn);
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    size_t num_valid = holder_->index_->knnSearch(
                            &queries[i * dimension], knn, result_indices.data(),
                            result_distances.data());

                    int num_neighbors = 0;
                    for (size_t valid_i = 0; valid_i < num_valid; ++valid_i) {
                        index_t idx = result_indices[valid_i];
                        if (ignore_query_point &&
                            points_equal(&queries[i * dimension],
                                         &points[idx * dimension], dimension)) {
                            continue;
                        }
                        pairs_private.push_back(Pair{int32_t(i), int32_t(idx)});
                        ++num_neighbors;
                    }
                    neighbors_count[i] = num_neighbors;
                }
                {
                    std::lock_guard<std::mutex> lock(pairs_mutex);
                    pairs.insert(pairs.end(), pairs_private.begin(),
                                 pairs_private.end());
                }
            });

    query_neighbors_row_splits[0] = 0;
    utility::InclusivePrefixSum(&neighbors_count[0],
                                &neighbors_count[neighbors_count.size()],
                                query_neighbors_row_splits + 1);

    index_t *indices_ptr;
    output_allocator.AllocIndices(&indices_ptr, pairs.size());
    T *distances_ptr;
    if (return_distances)
        output_allocator.AllocDistances(&distances_ptr, pairs.size());
    else
        output_allocator.AllocDistances(&distances_ptr, 0);

    std::fill(neighbors_count.begin(), neighbors_count.end(), 0);

    // fill output index and distance arrays
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, pairs.size()),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    Pair pair = pairs[i];

                    int64_t idx = query_neighbors_row_splits[pair.i] +
                                  core::AtomicFetchAddRelaxed(
                                          &neighbors_count[pair.i], 1);
                    indices_ptr[idx] = pair.j;

                    if (return_distances) {
                        T dist = distance_fn(&points[pair.j * dimension],
                                             &queries[pair.i * dimension],
                                             dimension);
                        distances_ptr[idx] = dist;
                    }
                }
            });
}

template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                      int64_t *query_neighbors_row_splits,
                      size_t num_queries,
                      const T *const queries,
                      const size_t dimension,
                      const T *const radii,
                      bool ignore_query_point,
                      bool return_distances,
                      bool sort,
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

    nanoflann::SearchParams params;
    params.sorted = sort;

    size_t num_indices = 0;
    auto holder_ =
            static_cast<NanoFlannIndexHolder<METRIC, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    T radius = radii[i];
                    T radius_squared = radius * radius;
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, params);
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

template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                      size_t num_queries,
                      const T *const queries,
                      const size_t dimension,
                      const T radius,
                      const int max_knn,
                      bool ignore_query_point,
                      bool return_distances,
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

    nanoflann::SearchParams params;
    params.sorted = true;

    auto holder_ =
            static_cast<NanoFlannIndexHolder<METRIC, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> ret_matches;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    size_t num_results = holder_->index_->radiusSearch(
                            &queries[i * dimension], radius_squared,
                            ret_matches, params);
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

}  // namespace

template <class T>
std::unique_ptr<NanoFlannIndexHolderBase> BuildKdTree(size_t num_points,
                                                      const T *const points,
                                                      size_t dimension,
                                                      const Metric metric) {
    NanoFlannIndexHolderBase *holder;
#define FN_PARAMETERS num_points, points, dimension, &holder

#define CALL_TEMPLATE(METRIC)                   \
    if (METRIC == metric) {                     \
        _BuildKdTree<T, METRIC>(FN_PARAMETERS); \
    }

#define CALL_TEMPLATE2 \
    CALL_TEMPLATE(L1)  \
    CALL_TEMPLATE(L2)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
    return std::unique_ptr<NanoFlannIndexHolderBase>(holder);
}

template <class T, class OUTPUT_ALLOCATOR>
void KnnSearchCPU(NanoFlannIndexHolderBase *holder,
                  int64_t *query_neighbors_row_splits,
                  size_t num_points,
                  const T *const points,
                  size_t num_queries,
                  const T *const queries,
                  const size_t dimension,
                  int knn,
                  const Metric metric,
                  bool ignore_query_point,
                  bool return_distances,
                  OUTPUT_ALLOCATOR &output_allocator) {
#define FN_PARAMETERS                                                      \
    holder, query_neighbors_row_splits, num_points, points, num_queries,   \
            queries, dimension, knn, ignore_query_point, return_distances, \
            output_allocator

#define CALL_TEMPLATE(METRIC)                                      \
    if (METRIC == metric) {                                        \
        _KnnSearchCPU<T, OUTPUT_ALLOCATOR, METRIC>(FN_PARAMETERS); \
    }

#define CALL_TEMPLATE2 \
    CALL_TEMPLATE(L1)  \
    CALL_TEMPLATE(L2)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
}

template <class T, class OUTPUT_ALLOCATOR>
void RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                     int64_t *query_neighbors_row_splits,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T *const radii,
                     const Metric metric,
                     bool ignore_query_point,
                     bool return_distances,
                     bool sort,
                     OUTPUT_ALLOCATOR &output_allocator) {
#define FN_PARAMETERS                                                    \
    holder, query_neighbors_row_splits, num_queries, queries, dimension, \
            radii, ignore_query_point, return_distances, sort,           \
            output_allocator

#define CALL_TEMPLATE(METRIC)                                         \
    if (METRIC == metric) {                                           \
        _RadiusSearchCPU<T, OUTPUT_ALLOCATOR, METRIC>(FN_PARAMETERS); \
    }

#define CALL_TEMPLATE2 \
    CALL_TEMPLATE(L1)  \
    CALL_TEMPLATE(L2)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
}

template <class T, class OUTPUT_ALLOCATOR>
void HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T radius,
                     const int max_knn,
                     const Metric metric,
                     bool ignore_query_point,
                     bool return_distances,
                     OUTPUT_ALLOCATOR &output_allocator) {
#define FN_PARAMETERS                                         \
    holder, num_queries, queries, dimension, radius, max_knn, \
            ignore_query_point, return_distances, output_allocator

#define CALL_TEMPLATE(METRIC)                                         \
    if (METRIC == metric) {                                           \
        _HybridSearchCPU<T, OUTPUT_ALLOCATOR, METRIC>(FN_PARAMETERS); \
    }

#define CALL_TEMPLATE2 \
    CALL_TEMPLATE(L1)  \
    CALL_TEMPLATE(L2)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
}

}  // namespace impl
}  // namespace nns
}  // namespace core
}  // namespace open3d
