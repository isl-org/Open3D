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

#include <algorithm>
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
    // return empty indices array if there are no points
    if (num_queries == 0 || num_points == 0 || holder == nullptr) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    auto points_equal = [](const T *const p1, const T *const p2,
                           size_t dimension) {
        std::vector<T> p1_vec(p1, p1 + dimension);
        std::vector<T> p2_vec(p2, p2 + dimension);
        return p1_vec == p2_vec;
    };

    std::vector<std::vector<index_t>> neighbors_indices(num_queries);
    std::vector<std::vector<T>> neighbors_distances(num_queries);
    std::vector<uint32_t> neighbors_count(num_queries, 0);

    // cast NanoFlannIndexHolder
    auto holder_ =
            static_cast<NanoFlannIndexHolder<METRIC, T, index_t> *>(holder);

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
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
                        neighbors_indices[i].push_back(idx);
                        if (return_distances) {
                            T dist = result_distances[valid_i];
                            neighbors_distances[i].push_back(dist);
                        }
                        ++num_neighbors;
                    }
                    neighbors_count[i] = num_neighbors;
                }
            });

    query_neighbors_row_splits[0] = 0;
    utility::InclusivePrefixSum(neighbors_count.data(),
                                neighbors_count.data() + neighbors_count.size(),
                                query_neighbors_row_splits + 1);

    int64_t num_indices = query_neighbors_row_splits[num_queries];

    index_t *indices_ptr;
    output_allocator.AllocIndices(&indices_ptr, num_indices);
    T *distances_ptr;
    if (return_distances)
        output_allocator.AllocDistances(&distances_ptr, num_indices);
    else
        output_allocator.AllocDistances(&distances_ptr, 0);

    std::fill(neighbors_count.begin(), neighbors_count.end(), 0);

    // fill output index and distance arrays
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_queries),
                      [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              int64_t start_idx = query_neighbors_row_splits[i];
                              std::copy(neighbors_indices[i].begin(),
                                        neighbors_indices[i].end(),
                                        &indices_ptr[start_idx]);

                              if (return_distances) {
                                  std::copy(neighbors_distances[i].begin(),
                                            neighbors_distances[i].end(),
                                            &distances_ptr[start_idx]);
                              }
                          }
                      });
}

template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                      int64_t *query_neighbors_row_splits,
                      size_t num_points,
                      const T *const points,
                      size_t num_queries,
                      const T *const queries,
                      const size_t dimension,
                      const T *const radii,
                      bool ignore_query_point,
                      bool return_distances,
                      bool normalize_distances,
                      bool sort,
                      OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0 || num_points == 0 || holder == nullptr) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        index_t *indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T *distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);
        return;
    }

    auto points_equal = [](const T *const p1, const T *const p2,
                           size_t dimension) {
        std::vector<T> p1_vec(p1, p1 + dimension);
        std::vector<T> p2_vec(p2, p2 + dimension);
        return p1_vec == p2_vec;
    };

    std::vector<std::vector<index_t>> neighbors_indices(num_queries);
    std::vector<std::vector<T>> neighbors_distances(num_queries);
    std::vector<uint32_t> neighbors_count(num_queries, 0);

    nanoflann::SearchParams params;
    params.sorted = sort;

    auto holder_ =
            static_cast<NanoFlannIndexHolder<METRIC, T, index_t> *>(holder);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                std::vector<std::pair<index_t, T>> search_result;
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    T radius = radii[i];
                    if (METRIC == L2) {
                        radius = radius * radius;
                    }

                    holder_->index_->radiusSearch(&queries[i * dimension],
                                                  radius, search_result,
                                                  params);

                    int num_neighbors = 0;
                    for (const auto &idx_dist : search_result) {
                        if (ignore_query_point &&
                            points_equal(&queries[i * dimension],
                                         &points[idx_dist.first * dimension],
                                         dimension)) {
                            continue;
                        }
                        neighbors_indices[i].push_back(idx_dist.first);
                        if (return_distances) {
                            neighbors_distances[i].push_back(idx_dist.second);
                        }
                        ++num_neighbors;
                    }
                    neighbors_count[i] = num_neighbors;
                }
            });

    query_neighbors_row_splits[0] = 0;
    utility::InclusivePrefixSum(neighbors_count.data(),
                                neighbors_count.data() + neighbors_count.size(),
                                query_neighbors_row_splits + 1);

    int64_t num_indices = query_neighbors_row_splits[num_queries];

    index_t *indices_ptr;
    output_allocator.AllocIndices(&indices_ptr, num_indices);
    T *distances_ptr;
    if (return_distances)
        output_allocator.AllocDistances(&distances_ptr, num_indices);
    else
        output_allocator.AllocDistances(&distances_ptr, 0);

    std::fill(neighbors_count.begin(), neighbors_count.end(), 0);

    // fill output index and distance arrays
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    int64_t start_idx = query_neighbors_row_splits[i];
                    std::copy(neighbors_indices[i].begin(),
                              neighbors_indices[i].end(),
                              &indices_ptr[start_idx]);
                    if (return_distances) {
                        std::transform(neighbors_distances[i].begin(),
                                       neighbors_distances[i].end(),
                                       &distances_ptr[start_idx], [&](T dist) {
                                           T d = dist;
                                           if (normalize_distances) {
                                               if (METRIC == L2) {
                                                   d /= (radii[i] * radii[i]);
                                               } else {
                                                   d /= radii[i];
                                               }
                                           }
                                           return d;
                                       });
                    }
                }
            });
}

template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                      size_t num_points,
                      const T *const points,
                      size_t num_queries,
                      const T *const queries,
                      const size_t dimension,
                      const T radius,
                      const int max_knn,
                      bool ignore_query_point,
                      bool return_distances,
                      OUTPUT_ALLOCATOR &output_allocator) {
    if (num_queries == 0 || num_points == 0 || holder == nullptr) {
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

/// Build KD Tree. This function build a KDTree for given dataset points.
///
/// \tparam T   Floating-point data type for the point positions.
///
///
/// \param num_points   The number of points.
///
/// \param points   Array with the point positions.
///
/// \param dimension    The dimension of points.
///
/// \param metric   Onf of L1, L2. Defines the distance metric for the
/// search
///
template <class T>
std::unique_ptr<NanoFlannIndexHolderBase> BuildKdTree(size_t num_points,
                                                      const T *const points,
                                                      size_t dimension,
                                                      const Metric metric) {
    NanoFlannIndexHolderBase *holder = nullptr;
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

/// KNN search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of each list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param holder   The pointer that point to NanFlannIndexHolder that is built
///        with BuildKdTree function.
///
/// \param query_neighbors_row_splits    This is the output pointer for the
///        prefix sum. The length of this array is \p num_queries + 1.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the point positions. This may be the same
///        array as \p queries.
///
/// \param num_queries    The number of query points.
///
/// \param queries    Array with the query positions. This may be the same
///                   array as \p points.
///
/// \param dimension    The dimension of \p points and \p queries.
///
/// \param knn    The number of neighbors to search.
///
/// \param metric    One of L1, L2. Defines the distance metric for the
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

/// Radius search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of each list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param holder   The pointer that point to NanFlannIndexHolder that is built
///        with BuildKdTree function.
///
/// \param query_neighbors_row_splits    This is the output pointer for the
///        prefix sum. The length of this array is \p num_queries + 1.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the point positions. This may be the same
///        array as \p queries.
///
/// \param num_queries    The number of query points.
///
/// \param queries    Array with the query positions. This may be the same
///                   array as \p points.
///
/// \param dimension    The dimension of \p points and \p queries.
///
/// \param radii    A vector of search radii with length \p num_queries.
///
/// \param metric    One of L1, L2. Defines the distance metric for the
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
/// \param normalize_distances    If true then the returned distances are
///        normalized in the range [0,1]. Note that for L2 the normalized
///        distance is squared.
///
/// \param sort     If true then sort the resulting indices and distances in
///        ascending order of distances.
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
void RadiusSearchCPU(NanoFlannIndexHolderBase *holder,
                     int64_t *query_neighbors_row_splits,
                     size_t num_points,
                     const T *const points,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T *const radii,
                     const Metric metric,
                     bool ignore_query_point,
                     bool return_distances,
                     bool normalize_distances,
                     bool sort,
                     OUTPUT_ALLOCATOR &output_allocator) {
#define FN_PARAMETERS                                                        \
    holder, query_neighbors_row_splits, num_points, points, num_queries,     \
            queries, dimension, radii, ignore_query_point, return_distances, \
            normalize_distances, sort, output_allocator

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

/// Hybrid search. This function computes a list of neighbor indices
/// for each query point. The lists are stored linearly and an exclusive prefix
/// sum defines the start and end of each list in the array.
/// In addition the function optionally can return the distances for each
/// neighbor in the same format as the indices to the neighbors.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param holder   The pointer that point to NanFlannIndexHolder that is built
///        with BuildKdTree function.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the point positions. This may be the same
///        array as \p queries.
///
/// \param num_queries    The number of query points.
///
/// \param queries    Array with the query positions. This may be the same
///                   array as \p points.
///
/// \param dimension    The dimension of \p points and \p queries.
///
/// \param radius    The radius value that defines the neighbors region.
///
/// \param max_knn    The maximum number of neighbors to search.
///
/// \param metric    One of L1, L2. Defines the distance metric for the
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
void HybridSearchCPU(NanoFlannIndexHolderBase *holder,
                     size_t num_points,
                     const T *const points,
                     size_t num_queries,
                     const T *const queries,
                     const size_t dimension,
                     const T radius,
                     const int max_knn,
                     const Metric metric,
                     bool ignore_query_point,
                     bool return_distances,
                     OUTPUT_ALLOCATOR &output_allocator) {
#define FN_PARAMETERS                                                    \
    holder, num_points, points, num_queries, queries, dimension, radius, \
            max_knn, ignore_query_point, return_distances, output_allocator

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
