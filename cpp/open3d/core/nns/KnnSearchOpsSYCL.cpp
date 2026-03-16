// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <limits>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/AddMM.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

// SYCL nearest-neighbor search uses tiled distance evaluation with AddMM,
// batched threshold/count helpers, and query-local top-k selection/merge.
namespace {

// Chooses a query/point tile shape that limits temporary distance storage.
inline void ChooseTileSizeSYCL(int64_t num_queries,
                               int64_t num_points,
                               int64_t element_size,
                               int64_t& tile_queries,
                               int64_t& tile_points) {
    constexpr int64_t kTargetTileBytes = 8 * 1024 * 1024;
    tile_queries = std::min<int64_t>(num_queries, 128);
    tile_queries = std::max<int64_t>(tile_queries, 1);
    tile_points = std::max<int64_t>(
            kTargetTileBytes / (tile_queries * element_size), int64_t(256));
    tile_points = std::min<int64_t>(tile_points, num_points);
    tile_points = std::max<int64_t>(tile_points, 1);
}

// Computes squared L2 distances from one query to all points with a SYCL
// kernel. This path is still used by the small single-row helpers below.
template <typename T>
void ComputeSquaredDistancesSYCL(const Device& device,
                                 const T* points_ptr,
                                 int64_t num_points,
                                 int64_t dimension,
                                 const T* query_ptr,
                                 T* distances_ptr) {
    if (num_points == 0) {
        return;
    }

    constexpr int64_t kWorkGroupSize = 128;
    const int64_t global_size =
            utility::DivUp(num_points, kWorkGroupSize) * kWorkGroupSize;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> local_query(sycl::range<1>(dimension), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global_size),
                                  sycl::range<1>(kWorkGroupSize)),
                [=](sycl::nd_item<1> item) {
                    const int64_t lid = item.get_local_linear_id();
                    for (int64_t d = lid; d < dimension;
                         d += item.get_local_range(0)) {
                        local_query[d] = query_ptr[d];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const int64_t point_idx = item.get_global_linear_id();
                    if (point_idx >= num_points) {
                        return;
                    }

                    const T* point = points_ptr + point_idx * dimension;
                    T dist = 0;
                    for (int64_t d = 0; d < dimension; ++d) {
                        const T diff = point[d] - local_query[d];
                        dist += diff * diff;
                    }
                    distances_ptr[point_idx] = dist;
                });
    });
}

// Counts how many entries per query are within a threshold using one launch per
// tile, avoiding per-query device algorithm dispatch.
template <typename T>
void CountWithinThresholdQueriesSYCL(const Device& device,
                                     const T* distances_ptr,
                                     int64_t distance_query_stride,
                                     int64_t num_queries,
                                     int64_t num_points,
                                     T threshold,
                                     int64_t* counts_ptr) {
    if (num_queries == 0 || num_points == 0) {
        return;
    }

    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t query_idx = id[0];
        int64_t local_count = 0;
        const T* query_distances =
                distances_ptr + query_idx * distance_query_stride;
        for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
            if (query_distances[point_idx] <= threshold) {
                ++local_count;
            }
        }
        counts_ptr[query_idx] += local_count;
    });
}

// Appends per-query threshold matches into the final neighbor buffers using
// caller-provided query offsets.
template <typename T, typename TIndex>
void GatherWithinThresholdQueriesSYCL(const Device& device,
                                      const T* distances_ptr,
                                      int64_t distance_query_stride,
                                      int64_t num_queries,
                                      int64_t num_points,
                                      T threshold,
                                      TIndex index_offset,
                                      int64_t* offsets_ptr,
                                      TIndex* out_indices_ptr,
                                      T* out_distances_ptr) {
    if (num_queries == 0 || num_points == 0) {
        return;
    }

    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t query_idx = id[0];
        const T* query_distances =
                distances_ptr + query_idx * distance_query_stride;
        int64_t offset = offsets_ptr[query_idx];
        for (int64_t point_idx = 0; point_idx < num_points; ++point_idx) {
            const T dist = query_distances[point_idx];
            if (dist <= threshold) {
                out_indices_ptr[offset] = index_offset + point_idx;
                if (out_distances_ptr != nullptr) {
                    out_distances_ptr[offset] = dist;
                }
                ++offset;
            }
        }
        offsets_ptr[query_idx] = offset;
    });
}

// Selects the smallest k distances per query with OneDPL partial_sort while
// reusing scratch index buffers allocated by the caller.
template <typename T, typename TIndex>
void SelectTopKQueriesSYCL(const Device& device,
                           const T* distances_ptr,
                           int64_t distance_query_stride,
                           int64_t num_queries,
                           int64_t num_points,
                           int64_t knn,
                           TIndex index_offset,
                           TIndex* scratch_indices_ptr,
                           int64_t scratch_query_stride,
                           TIndex* out_indices_ptr,
                           T* out_distances_ptr,
                           int64_t out_query_stride,
                           bool use_threshold = false,
                           T threshold = T(0)) {
    if (num_queries == 0 || num_points == 0 || knn <= 0) {
        return;
    }

    const T inf = std::numeric_limits<T>::max();
    const int64_t actual_knn = std::min(knn, num_points);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    queue.parallel_for(
            sycl::range<2>(num_queries, num_points), [=](sycl::id<2> id) {
                scratch_indices_ptr[id[0] * scratch_query_stride + id[1]] =
                        static_cast<TIndex>(id[1]);
            });

    for (int64_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        TIndex* query_indices =
                scratch_indices_ptr + query_idx * scratch_query_stride;
        const T* query_distances =
                distances_ptr + query_idx * distance_query_stride;
        std::partial_sort(policy, query_indices, query_indices + actual_knn,
                          query_indices + num_points,
                          [query_distances](TIndex lhs, TIndex rhs) {
                              const T lhs_dist = query_distances[lhs];
                              const T rhs_dist = query_distances[rhs];
                              if (lhs_dist < rhs_dist) {
                                  return true;
                              }
                              if (rhs_dist < lhs_dist) {
                                  return false;
                              }
                              return lhs < rhs;
                          });
    }

    queue.parallel_for(sycl::range<2>(num_queries, knn), [=](sycl::id<2> id) {
        const int64_t query_idx = id[0];
        const int64_t k = id[1];
        TIndex* query_out_indices =
                out_indices_ptr + query_idx * out_query_stride;
        T* query_out_distances =
                out_distances_ptr + query_idx * out_query_stride;

        if (k >= actual_knn) {
            query_out_indices[k] = TIndex(-1);
            query_out_distances[k] = inf;
            return;
        }

        const TIndex local_idx =
                scratch_indices_ptr[query_idx * scratch_query_stride + k];
        const T dist =
                distances_ptr[query_idx * distance_query_stride + local_idx];
        if (use_threshold && dist > threshold) {
            query_out_indices[k] = TIndex(-1);
            query_out_distances[k] = inf;
            return;
        }
        query_out_indices[k] = index_offset + local_idx;
        query_out_distances[k] = dist;
    });
}

// Merges two sorted top-k query buffers into a new top-k query buffer with
// OneDPL partial_sort over 2k candidates.
template <typename T, typename TIndex>
void MergeTopKQueriesSYCL(const Device& device,
                          const T* current_distances_ptr,
                          const TIndex* current_indices_ptr,
                          int64_t current_query_stride,
                          const T* candidate_distances_ptr,
                          const TIndex* candidate_indices_ptr,
                          int64_t candidate_query_stride,
                          int64_t num_queries,
                          int64_t knn,
                          TIndex* scratch_indices_ptr,
                          int64_t scratch_query_stride,
                          TIndex* out_indices_ptr,
                          T* out_distances_ptr,
                          int64_t out_query_stride) {
    if (num_queries == 0 || knn <= 0) {
        return;
    }

    const T inf = std::numeric_limits<T>::max();
    const int64_t combined_cols = 2 * knn;
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    auto policy = oneapi::dpl::execution::make_device_policy(queue);

    queue.parallel_for(
            sycl::range<2>(num_queries, combined_cols), [=](sycl::id<2> id) {
                scratch_indices_ptr[id[0] * scratch_query_stride + id[1]] =
                        static_cast<TIndex>(id[1]);
            });

    for (int64_t query_idx = 0; query_idx < num_queries; ++query_idx) {
        TIndex* query_indices =
                scratch_indices_ptr + query_idx * scratch_query_stride;
        const T* query_current_dist =
                current_distances_ptr + query_idx * current_query_stride;
        const TIndex* query_current_idx =
                current_indices_ptr + query_idx * current_query_stride;
        const T* query_candidate_dist =
                candidate_distances_ptr + query_idx * candidate_query_stride;
        const TIndex* query_candidate_idx =
                candidate_indices_ptr + query_idx * candidate_query_stride;

        std::partial_sort(
                policy, query_indices, query_indices + knn,
                query_indices + combined_cols,
                [query_current_dist, query_current_idx, query_candidate_dist,
                 query_candidate_idx, knn](TIndex lhs, TIndex rhs) {
                    const bool lhs_is_current = lhs < knn;
                    const bool rhs_is_current = rhs < knn;
                    const TIndex lhs_idx =
                            lhs_is_current ? query_current_idx[lhs]
                                           : query_candidate_idx[lhs - knn];
                    const TIndex rhs_idx =
                            rhs_is_current ? query_current_idx[rhs]
                                           : query_candidate_idx[rhs - knn];
                    const T lhs_dist =
                            lhs_is_current ? query_current_dist[lhs]
                                           : query_candidate_dist[lhs - knn];
                    const T rhs_dist =
                            rhs_is_current ? query_current_dist[rhs]
                                           : query_candidate_dist[rhs - knn];
                    const bool lhs_valid = lhs_idx >= 0;
                    const bool rhs_valid = rhs_idx >= 0;
                    if (lhs_valid != rhs_valid) {
                        return lhs_valid;
                    }
                    if (lhs_dist < rhs_dist) {
                        return true;
                    }
                    if (rhs_dist < lhs_dist) {
                        return false;
                    }
                    return lhs_idx < rhs_idx;
                });
    }

    queue.parallel_for(sycl::range<2>(num_queries, knn), [=](sycl::id<2> id) {
        const int64_t query_idx = id[0];
        const int64_t k = id[1];
        TIndex* query_out_indices =
                out_indices_ptr + query_idx * out_query_stride;
        T* query_out_distances =
                out_distances_ptr + query_idx * out_query_stride;
        const TIndex source =
                scratch_indices_ptr[query_idx * scratch_query_stride + k];
        const bool is_current = source < knn;
        const int64_t offset = is_current ? source : source - knn;
        const TIndex idx =
                is_current
                        ? current_indices_ptr[query_idx * current_query_stride +
                                              offset]
                        : candidate_indices_ptr[query_idx *
                                                        candidate_query_stride +
                                                offset];
        const T dist =
                is_current
                        ? current_distances_ptr[query_idx *
                                                        current_query_stride +
                                                offset]
                        : candidate_distances_ptr
                                  [query_idx * candidate_query_stride + offset];
        if (idx < 0) {
            query_out_indices[k] = TIndex(-1);
            query_out_distances[k] = inf;
        } else {
            query_out_indices[k] = idx;
            query_out_distances[k] = dist;
        }
    });
}

// Finalizes hybrid-search counts and clears any unused tail elements in the
// fixed-width output tensors.
template <typename T, typename TIndex>
void FinalizeHybridResultsSYCL(const Device& device,
                               const int64_t* counts_ptr,
                               int64_t num_queries,
                               int64_t max_knn,
                               TIndex* neighbors_index_ptr,
                               T* neighbors_distance_ptr,
                               TIndex* neighbors_count_ptr) {
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<1>(num_queries), [=](sycl::id<1> id) {
        const int64_t query_idx = id[0];
        const int64_t clipped_count =
                std::min<int64_t>(counts_ptr[query_idx], max_knn);
        neighbors_count_ptr[query_idx] = static_cast<TIndex>(clipped_count);
        for (int64_t k = clipped_count; k < max_knn; ++k) {
            neighbors_index_ptr[query_idx * max_knn + k] = TIndex(-1);
            neighbors_distance_ptr[query_idx * max_knn + k] = T(0);
        }
    });
}

}  // namespace

// Batched KNN search with tiled AddMM distance evaluation and query-local top-k
// selection/merge on each tile.
template <class T, class TIndex>
void KnnSearchSYCL(const Tensor& points,
                   const Tensor& points_row_splits,
                   const Tensor& queries,
                   const Tensor& queries_row_splits,
                   int knn,
                   Tensor& neighbors_index,
                   Tensor& neighbors_row_splits,
                   Tensor& neighbors_distance) {
    const Device device = points.GetDevice();
    const Dtype dtype = points.GetDtype();
    const Dtype index_dtype = Dtype::FromType<TIndex>();
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    const int batch_size = points_row_splits.GetShape(0) - 1;
    std::vector<NeighborSearchAllocator<T, TIndex>> batch_output_allocators(
            batch_size, NeighborSearchAllocator<T, TIndex>(device));

    int64_t* neighbors_row_splits_ptr =
            neighbors_row_splits.GetDataPtr<int64_t>();
    int64_t last_neighbors_count = 0;
    int64_t batch_knn = 0;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();

        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        batch_knn = std::min<int64_t>(knn, num_points_i);

        neighbors_row_splits_ptr[query_begin] = last_neighbors_count;
        for (int64_t q = 0; q < num_queries_i; ++q) {
            neighbors_row_splits_ptr[query_begin + q + 1] =
                    last_neighbors_count + (q + 1) * batch_knn;
        }
        last_neighbors_count += num_queries_i * batch_knn;

        TIndex* indices_ptr;
        T* distances_ptr;
        batch_output_allocators[batch_idx].AllocIndices(
                &indices_ptr, num_queries_i * batch_knn, TIndex(-1));
        batch_output_allocators[batch_idx].AllocDistances(
                &distances_ptr, num_queries_i * batch_knn, T(0));

        Tensor out_indices =
                batch_output_allocators[batch_idx].NeighborsIndex().View(
                        {num_queries_i, batch_knn});
        Tensor out_distances =
                batch_output_allocators[batch_idx].NeighborsDistance().View(
                        {num_queries_i, batch_knn});

        // Precompute norms for batched distance evaluation.
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        int64_t tile_queries = 0;
        int64_t tile_points = 0;
        ChooseTileSizeSYCL(num_queries_i, num_points_i, sizeof(T), tile_queries,
                           tile_points);

        Tensor temp_distances =
                Tensor::Empty({tile_queries, tile_points}, dtype, device);
        Tensor tile_sort_indices =
                Tensor::Empty({tile_queries, tile_points}, index_dtype, device);
        Tensor tile_top_indices =
                Tensor::Empty({tile_queries, batch_knn}, index_dtype, device);
        Tensor tile_top_distances =
                Tensor::Empty({tile_queries, batch_knn}, dtype, device);
        Tensor best_indices =
                Tensor::Empty({tile_queries, batch_knn}, index_dtype, device);
        Tensor best_distances =
                Tensor::Empty({tile_queries, batch_knn}, dtype, device);
        Tensor merged_indices =
                Tensor::Empty({tile_queries, batch_knn}, index_dtype, device);
        Tensor merged_distances =
                Tensor::Empty({tile_queries, batch_knn}, dtype, device);
        Tensor merge_sort_indices = Tensor::Empty({tile_queries, 2 * batch_knn},
                                                  index_dtype, device);

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);

            Tensor best_indices_view =
                    best_indices.Slice(0, 0, num_queries_iter);
            Tensor best_distances_view =
                    best_distances.Slice(0, 0, num_queries_iter);
            best_indices_view.Fill(TIndex(-1));
            best_distances_view.Fill(std::numeric_limits<T>::max());

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_distances_view =
                        temp_distances.Slice(0, 0, num_queries_iter)
                                .Slice(1, 0, num_points_iter);

                AddMM(queries_tile, points_tile.T(), temp_distances_view, -2.0,
                      0.0);
                temp_distances_view.Add_(
                        point_norms_tile.View({1, num_points_iter}));
                temp_distances_view.Add_(
                        query_norms_tile.View({num_queries_iter, 1}));

                Tensor tile_top_indices_view =
                        tile_top_indices.Slice(0, 0, num_queries_iter);
                Tensor tile_top_distances_view =
                        tile_top_distances.Slice(0, 0, num_queries_iter);
                Tensor merged_indices_view =
                        merged_indices.Slice(0, 0, num_queries_iter);
                Tensor merged_distances_view =
                        merged_distances.Slice(0, 0, num_queries_iter);

                SelectTopKQueriesSYCL<T, TIndex>(
                        device, temp_distances_view.GetDataPtr<T>(),
                        temp_distances_view.GetStride(0), num_queries_iter,
                        num_points_iter, batch_knn, TIndex(p),
                        tile_sort_indices.GetDataPtr<TIndex>(),
                        tile_sort_indices.GetStride(0),
                        tile_top_indices_view.GetDataPtr<TIndex>(),
                        tile_top_distances_view.GetDataPtr<T>(), batch_knn);
                MergeTopKQueriesSYCL<T, TIndex>(
                        device, best_distances_view.GetDataPtr<T>(),
                        best_indices_view.GetDataPtr<TIndex>(), batch_knn,
                        tile_top_distances_view.GetDataPtr<T>(),
                        tile_top_indices_view.GetDataPtr<TIndex>(), batch_knn,
                        num_queries_iter, batch_knn,
                        merge_sort_indices.GetDataPtr<TIndex>(),
                        merge_sort_indices.GetStride(0),
                        merged_indices_view.GetDataPtr<TIndex>(),
                        merged_distances_view.GetDataPtr<T>(), batch_knn);

                best_indices_view.AsRvalue() = merged_indices_view;
                best_distances_view.AsRvalue() = merged_distances_view;
            }

            out_indices.Slice(0, q, q + num_queries_iter).AsRvalue() =
                    best_indices_view;
            out_distances.Slice(0, q, q + num_queries_iter).AsRvalue() =
                    best_distances_view;
        }

        queue.wait_and_throw();
    }

    if (batch_size == 1) {
        neighbors_index = batch_output_allocators[0].NeighborsIndex().View(
                {queries.GetShape(0), batch_knn});
        neighbors_distance =
                batch_output_allocators[0].NeighborsDistance().View(
                        {queries.GetShape(0), batch_knn});
        return;
    }

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    int64_t neighbors_size = 0;
    for (const auto& allocator : batch_output_allocators) {
        neighbors_size += allocator.NeighborsIndex().GetShape(0);
    }

    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, neighbors_size);
    output_allocator.AllocDistances(&neighbors_distance_ptr, neighbors_size);

    int64_t offset = 0;
    for (const auto& allocator : batch_output_allocators) {
        const int64_t batch_size_i = allocator.NeighborsIndex().GetShape(0);
        if (batch_size_i == 0) {
            continue;
        }
        MemoryManager::Memcpy(neighbors_index_ptr + offset, device,
                              allocator.IndicesPtr(), device,
                              sizeof(TIndex) * batch_size_i);
        MemoryManager::Memcpy(neighbors_distance_ptr + offset, device,
                              allocator.DistancesPtr(), device,
                              sizeof(T) * batch_size_i);
        offset += batch_size_i;
    }
    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

// Fixed-radius search using two tiled passes: one to count matches and one to
// gather indices and optional distances.
template <class T, class TIndex>
void FixedRadiusSearchSYCL(const Tensor& points,
                           const Tensor& queries,
                           double radius,
                           const Tensor& points_row_splits,
                           const Tensor& queries_row_splits,
                           const Tensor&,
                           const Tensor&,
                           const Tensor&,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           const bool,
                           Tensor& neighbors_index,
                           Tensor& neighbors_row_splits,
                           Tensor& neighbors_distance) {
    if (metric != Metric::L2) {
        utility::LogError("SYCL fixed radius search only supports L2 metric.");
    }
    if (ignore_query_point) {
        utility::LogError(
                "SYCL fixed radius search does not support "
                "ignore_query_point.");
    }

    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const Dtype dtype = points.GetDtype();
    const T threshold = static_cast<T>(radius * radius);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    Tensor counts = Tensor::Zeros({num_queries}, Int64, device);

    int64_t tile_queries = 0;
    int64_t tile_points = 0;
    ChooseTileSizeSYCL(num_queries, points.GetShape(0), sizeof(T), tile_queries,
                       tile_points);
    Tensor temp_distances =
            Tensor::Empty({tile_queries, tile_points}, dtype, device);

    for (int batch_idx = 0; batch_idx < points_row_splits.GetShape(0) - 1;
         ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            int64_t* counts_ptr =
                    counts.GetDataPtr<int64_t>() + query_begin + q;

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_distances_view =
                        temp_distances.Slice(0, 0, num_queries_iter)
                                .Slice(1, 0, num_points_iter);

                AddMM(queries_tile, points_tile.T(), temp_distances_view, -2.0,
                      0.0);
                temp_distances_view.Add_(
                        point_norms_tile.View({1, num_points_iter}));
                temp_distances_view.Add_(
                        query_norms_tile.View({num_queries_iter, 1}));
                CountWithinThresholdQueriesSYCL(
                        device, temp_distances_view.GetDataPtr<T>(),
                        temp_distances_view.GetStride(0), num_queries_iter,
                        num_points_iter, threshold, counts_ptr);
            }
        }
    }

    queue.wait_and_throw();

    Tensor counts_cpu = counts.To(Device("CPU:0"));
    const int64_t* counts_cpu_ptr = counts_cpu.GetDataPtr<int64_t>();
    std::vector<int64_t> row_splits(num_queries + 1, 0);
    for (int64_t q = 0; q < num_queries; ++q) {
        row_splits[q + 1] = row_splits[q] + counts_cpu_ptr[q];
    }
    neighbors_row_splits =
            Tensor(row_splits, {num_queries + 1}, Int64).To(device);

    std::vector<int64_t> row_offsets(row_splits.begin(), row_splits.end() - 1);
    Tensor write_offsets = Tensor(row_offsets, {num_queries}, Int64).To(device);

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, row_splits.back());
    if (return_distances) {
        output_allocator.AllocDistances(&neighbors_distance_ptr,
                                        row_splits.back());
    } else {
        output_allocator.AllocDistances(&neighbors_distance_ptr, 0);
    }

    for (int batch_idx = 0; batch_idx < points_row_splits.GetShape(0) - 1;
         ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            int64_t* offsets_ptr =
                    write_offsets.GetDataPtr<int64_t>() + query_begin + q;

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_distances_view =
                        temp_distances.Slice(0, 0, num_queries_iter)
                                .Slice(1, 0, num_points_iter);

                AddMM(queries_tile, points_tile.T(), temp_distances_view, -2.0,
                      0.0);
                temp_distances_view.Add_(
                        point_norms_tile.View({1, num_points_iter}));
                temp_distances_view.Add_(
                        query_norms_tile.View({num_queries_iter, 1}));
                GatherWithinThresholdQueriesSYCL<T, TIndex>(
                        device, temp_distances_view.GetDataPtr<T>(),
                        temp_distances_view.GetStride(0), num_queries_iter,
                        num_points_iter, threshold, TIndex(point_begin + p),
                        offsets_ptr, neighbors_index_ptr,
                        return_distances ? neighbors_distance_ptr : nullptr);
            }
        }
    }

    queue.wait_and_throw();

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

// Hybrid search counts all neighbors within radius while keeping only the best
// max_knn results per query in fixed-size output tensors.
template <class T, class TIndex>
void HybridSearchSYCL(const Tensor& points,
                      const Tensor& queries,
                      double radius,
                      int max_knn,
                      const Tensor& points_row_splits,
                      const Tensor& queries_row_splits,
                      const Tensor&,
                      const Tensor&,
                      const Tensor&,
                      const Metric metric,
                      Tensor& neighbors_index,
                      Tensor& neighbors_count,
                      Tensor& neighbors_distance) {
    if (metric != Metric::L2) {
        utility::LogError("SYCL hybrid search only supports L2 metric.");
    }

    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const Dtype dtype = points.GetDtype();
    const T threshold = static_cast<T>(radius * radius);
    sycl::queue queue = sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr, num_queries * max_knn,
                                  TIndex(-1));
    output_allocator.AllocDistances(&neighbors_distance_ptr,
                                    num_queries * max_knn,
                                    std::numeric_limits<T>::max());

    Tensor counts = Tensor::Zeros({num_queries}, Int64, device);
    Tensor out_indices =
            output_allocator.NeighborsIndex().View({num_queries, max_knn});
    Tensor out_distances =
            output_allocator.NeighborsDistance().View({num_queries, max_knn});

    int64_t tile_queries = 0;
    int64_t tile_points = 0;
    ChooseTileSizeSYCL(num_queries, points.GetShape(0), sizeof(T), tile_queries,
                       tile_points);
    Tensor temp_distances =
            Tensor::Empty({tile_queries, tile_points}, dtype, device);
    Tensor tile_sort_indices = Tensor::Empty({tile_queries, tile_points},
                                             Dtype::FromType<TIndex>(), device);
    Tensor tile_top_indices = Tensor::Empty({tile_queries, max_knn},
                                            Dtype::FromType<TIndex>(), device);
    Tensor tile_top_distances =
            Tensor::Empty({tile_queries, max_knn}, dtype, device);
    Tensor merged_indices = Tensor::Empty({tile_queries, max_knn},
                                          Dtype::FromType<TIndex>(), device);
    Tensor merged_distances =
            Tensor::Empty({tile_queries, max_knn}, dtype, device);
    Tensor merge_sort_indices = Tensor::Empty(
            {tile_queries, 2 * max_knn}, Dtype::FromType<TIndex>(), device);

    for (int batch_idx = 0; batch_idx < points_row_splits.GetShape(0) - 1;
         ++batch_idx) {
        const int64_t point_begin =
                points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end =
                points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);
        const Tensor queries_i = queries.Slice(0, query_begin, query_end);
        const int64_t num_points_i = points_i.GetShape(0);
        const int64_t num_queries_i = queries_i.GetShape(0);
        Tensor point_norms = points_i.Mul(points_i).Sum({1});
        Tensor query_norms = queries_i.Mul(queries_i).Sum({1});

        for (int64_t q = 0; q < num_queries_i; q += tile_queries) {
            const int64_t num_queries_iter =
                    std::min(tile_queries, num_queries_i - q);
            Tensor queries_tile = queries_i.Slice(0, q, q + num_queries_iter);
            Tensor query_norms_tile =
                    query_norms.Slice(0, q, q + num_queries_iter);
            Tensor best_indices_view = out_indices.Slice(
                    0, query_begin + q, query_begin + q + num_queries_iter);
            Tensor best_distances_view = out_distances.Slice(
                    0, query_begin + q, query_begin + q + num_queries_iter);
            int64_t* counts_ptr =
                    counts.GetDataPtr<int64_t>() + query_begin + q;

            for (int64_t p = 0; p < num_points_i; p += tile_points) {
                const int64_t num_points_iter =
                        std::min(tile_points, num_points_i - p);
                Tensor points_tile = points_i.Slice(0, p, p + num_points_iter);
                Tensor point_norms_tile =
                        point_norms.Slice(0, p, p + num_points_iter);
                Tensor temp_distances_view =
                        temp_distances.Slice(0, 0, num_queries_iter)
                                .Slice(1, 0, num_points_iter);

                AddMM(queries_tile, points_tile.T(), temp_distances_view, -2.0,
                      0.0);
                temp_distances_view.Add_(
                        point_norms_tile.View({1, num_points_iter}));
                temp_distances_view.Add_(
                        query_norms_tile.View({num_queries_iter, 1}));

                CountWithinThresholdQueriesSYCL(
                        device, temp_distances_view.GetDataPtr<T>(),
                        temp_distances_view.GetStride(0), num_queries_iter,
                        num_points_iter, threshold, counts_ptr);

                Tensor tile_top_indices_view =
                        tile_top_indices.Slice(0, 0, num_queries_iter);
                Tensor tile_top_distances_view =
                        tile_top_distances.Slice(0, 0, num_queries_iter);
                Tensor merged_indices_view =
                        merged_indices.Slice(0, 0, num_queries_iter);
                Tensor merged_distances_view =
                        merged_distances.Slice(0, 0, num_queries_iter);

                SelectTopKQueriesSYCL<T, TIndex>(
                        device, temp_distances_view.GetDataPtr<T>(),
                        temp_distances_view.GetStride(0), num_queries_iter,
                        num_points_iter, max_knn, TIndex(point_begin + p),
                        tile_sort_indices.GetDataPtr<TIndex>(),
                        tile_sort_indices.GetStride(0),
                        tile_top_indices_view.GetDataPtr<TIndex>(),
                        tile_top_distances_view.GetDataPtr<T>(), max_knn, true,
                        threshold);
                MergeTopKQueriesSYCL<T, TIndex>(
                        device, best_distances_view.GetDataPtr<T>(),
                        best_indices_view.GetDataPtr<TIndex>(), max_knn,
                        tile_top_distances_view.GetDataPtr<T>(),
                        tile_top_indices_view.GetDataPtr<TIndex>(), max_knn,
                        num_queries_iter, max_knn,
                        merge_sort_indices.GetDataPtr<TIndex>(),
                        merge_sort_indices.GetStride(0),
                        merged_indices_view.GetDataPtr<TIndex>(),
                        merged_distances_view.GetDataPtr<T>(), max_knn);

                best_indices_view.AsRvalue() = merged_indices_view;
                best_distances_view.AsRvalue() = merged_distances_view;
            }
        }
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    neighbors_count =
            Tensor::Empty({num_queries}, Dtype::FromType<TIndex>(), device);
    FinalizeHybridResultsSYCL<T, TIndex>(device, counts.GetDataPtr<int64_t>(),
                                         num_queries, max_knn,
                                         neighbors_index.GetDataPtr<TIndex>(),
                                         neighbors_distance.GetDataPtr<T>(),
                                         neighbors_count.GetDataPtr<TIndex>());
    queue.wait_and_throw();
}

#define INSTANTIATE(T, TIndex)                                                 \
    template void KnnSearchSYCL<T, TIndex>(                                    \
            const Tensor& points, const Tensor& points_row_splits,             \
            const Tensor& queries, const Tensor& queries_row_splits, int knn,  \
            Tensor& neighbors_index, Tensor& neighbors_row_splits,             \
            Tensor& neighbors_distance);                                       \
    template void FixedRadiusSearchSYCL<T, TIndex>(                            \
            const Tensor& points, const Tensor& queries, double radius,        \
            const Tensor& points_row_splits, const Tensor& queries_row_splits, \
            const Tensor&, const Tensor&, const Tensor&, const Metric metric,  \
            const bool ignore_query_point, const bool return_distances,        \
            const bool sort, Tensor& neighbors_index,                          \
            Tensor& neighbors_row_splits, Tensor& neighbors_distance);         \
    template void HybridSearchSYCL<T, TIndex>(                                 \
            const Tensor& points, const Tensor& queries, double radius,        \
            int max_knn, const Tensor& points_row_splits,                      \
            const Tensor& queries_row_splits, const Tensor&, const Tensor&,    \
            const Tensor&, const Metric metric, Tensor& neighbors_index,       \
            Tensor& neighbors_count, Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)

}  // namespace nns
}  // namespace core
}  // namespace open3d
