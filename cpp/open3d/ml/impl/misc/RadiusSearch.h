// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include "open3d/core/Atomic.h"
#include "open3d/ml/impl/misc/NeighborSearchCommon.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {

/// Implementation of RadiusSearchCPU with template params for metrics
/// and boolean options
template <class T, class OUTPUT_ALLOCATOR, int METRIC>
void _RadiusSearchCPU(int64_t* query_neighbors_row_splits,
                      size_t num_points,
                      const T* const points,
                      size_t num_queries,
                      const T* const queries,
                      const T* const radii,
                      bool ignore_query_point,
                      bool return_distances,
                      bool normalize_distances,
                      OUTPUT_ALLOCATOR& output_allocator) {
    using namespace open3d::utility;

    // return empty indices array if there are no points
    if (num_points == 0 || num_queries == 0) {
        std::fill(query_neighbors_row_splits,
                  query_neighbors_row_splits + num_queries + 1, 0);
        int32_t* indices_ptr;
        output_allocator.AllocIndices(&indices_ptr, 0);

        T* distances_ptr;
        output_allocator.AllocDistances(&distances_ptr, 0);

        return;
    }

    Adaptor<T> adaptor(num_points, points);

    typedef nanoflann::KDTreeSingleIndexAdaptor<
            typename SelectNanoflannAdaptor<METRIC, T>::Adaptor_t, Adaptor<T>,
            3>
            KDTree_t;

    KDTree_t index(3, adaptor);
    index.buildIndex();

    // temporary storage for the result
    struct Pair {
        int32_t i, j;
    };
    std::vector<Pair> pairs;

    // do not sort search results
    nanoflann::SearchParams search_params(32, 0, false);

    auto points_equal = [](const T* const p1, const T* const p2) {
        return p1[0] == p2[0] && p1[1] == p2[1] && p1[2] == p2[2];
    };

    auto distance_fn = [](const T* const p1, const T* const p2) {
        T dx = p1[0] - p2[0];
        T dy = p1[1] - p2[1];
        T dz = p1[2] - p2[2];
        if (METRIC == L2) {
            return dx * dx + dy * dy + dz * dz;
        } else  // L1
        {
            return std::abs(dx) + std::abs(dy) + std::abs(dz);
        }
    };

    std::mutex pairs_mutex;  // protects write access to pairs
    std::vector<uint32_t> neighbors_count(num_queries, 0);

    // compute nearest neighbors and store in pairs
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_queries),
            [&](const tbb::blocked_range<size_t>& r) {
                std::vector<Pair> pairs_private;
                std::vector<std::pair<size_t, T>> search_result;

                for (size_t i = r.begin(); i != r.end(); ++i) {
                    T radius;
                    if (METRIC == L2)
                        radius = radii[i] * radii[i];
                    else  // L1
                        radius = radii[i];
                    index.radiusSearch(&queries[i * 3], radius, search_result,
                                       search_params);

                    int num_neighbors = 0;
                    for (const auto& idx_dist : search_result) {
                        if (ignore_query_point &&
                            points_equal(&queries[i * 3],
                                         &points[idx_dist.first * 3])) {
                            continue;
                        }
                        pairs_private.push_back(
                                Pair{int32_t(i), int32_t(idx_dist.first)});
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
    InclusivePrefixSum(&neighbors_count[0],
                       &neighbors_count[neighbors_count.size()],
                       query_neighbors_row_splits + 1);

    int32_t* neighbors_indices_ptr;
    output_allocator.AllocIndices(&neighbors_indices_ptr, pairs.size());
    T* distances_ptr;
    if (return_distances)
        output_allocator.AllocDistances(&distances_ptr, pairs.size());
    else
        output_allocator.AllocDistances(&distances_ptr, 0);

    std::fill(neighbors_count.begin(), neighbors_count.end(), 0);

    // fill output index and distance arrays
    tbb::parallel_for(tbb::blocked_range<size_t>(0, pairs.size()),
                      [&](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                              Pair pair = pairs[i];

                              int64_t idx =
                                      query_neighbors_row_splits[pair.i] +
                                      core::AtomicFetchAddRelaxed(
                                              &neighbors_count[pair.i], 1);
                              neighbors_indices_ptr[idx] = pair.j;

                              if (return_distances) {
                                  T dist = distance_fn(&points[pair.j * 3],
                                                       &queries[pair.i * 3]);
                                  if (normalize_distances) {
                                      if (METRIC == L2)
                                          dist /= radii[pair.i] * radii[pair.i];
                                      else  // L1
                                          dist /= radii[pair.i];
                                  }
                                  distances_ptr[idx] = dist;
                              }
                          }
                      });
}

}  // namespace

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
/// \param radii    A vector of search radii with length \p num_queries.
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
/// \param normalize_distances    If true then the returned distances are
///        normalized in the range [0,1]. Note that for L2 the normalized
///        distance is squared.
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
void RadiusSearchCPU(int64_t* query_neighbors_row_splits,
                     size_t num_points,
                     const T* const points,
                     size_t num_queries,
                     const T* const queries,
                     const T* radii,
                     const Metric metric,
                     const bool ignore_query_point,
                     const bool return_distances,
                     const bool normalize_distances,
                     OUTPUT_ALLOCATOR& output_allocator) {
#define FN_PARAMETERS                                                         \
    query_neighbors_row_splits, num_points, points, num_queries, queries,     \
            radii, ignore_query_point, return_distances, normalize_distances, \
            output_allocator

#define CALL_TEMPLATE(METRIC) \
    if (METRIC == metric)     \
        _RadiusSearchCPU<T, OUTPUT_ALLOCATOR, METRIC>(FN_PARAMETERS);

#define CALL_TEMPLATE2 \
    CALL_TEMPLATE(L1)  \
    CALL_TEMPLATE(L2)

    CALL_TEMPLATE2

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2

#undef FN_PARAMETERS
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
