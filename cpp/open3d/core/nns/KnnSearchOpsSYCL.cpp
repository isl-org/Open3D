// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>
#include <vector>

#include "open3d/core/MemoryManager.h"
#include "open3d/core/SYCLContext.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/KnnIndex.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

namespace {

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
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    queue.submit([&](sycl::handler& cgh) {
             sycl::local_accessor<T, 1> local_query(
                     sycl::range<1>(dimension), cgh);
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
         }).wait_and_throw();
}

template <typename T, typename TIndex>
void SortIndicesByDistanceSYCL(const Device& device,
                               const T* distances_ptr,
                               TIndex* indices_ptr,
                               int64_t num_points) {
    if (num_points <= 1) {
        return;
    }
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    auto policy = oneapi::dpl::execution::make_device_policy(queue);
    std::sort(policy, indices_ptr, indices_ptr + num_points,
              [distances_ptr](TIndex lhs, TIndex rhs) {
                  const T lhs_dist = distances_ptr[lhs];
                  const T rhs_dist = distances_ptr[rhs];
                  if (lhs_dist < rhs_dist) {
                      return true;
                  }
                  if (rhs_dist < lhs_dist) {
                      return false;
                  }
                  return lhs < rhs;
              });
    queue.wait_and_throw();
}

template <typename T, typename TIndex>
int64_t CountWithinRadiusSYCL(const Device& device,
                              const T* distances_ptr,
                              const TIndex* sorted_indices_ptr,
                              int64_t num_points,
                              T threshold) {
    if (num_points == 0) {
        return 0;
    }

    constexpr int64_t kWorkGroupSize = 128;
    const int64_t global_size =
            utility::DivUp(num_points, kWorkGroupSize) * kWorkGroupSize;
    Tensor count = Tensor::Zeros({1}, Int64, device);
    int64_t* count_ptr = count.GetDataPtr<int64_t>();
    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);

    queue.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(
                     sycl::nd_range<1>(sycl::range<1>(global_size),
                                       sycl::range<1>(kWorkGroupSize)),
                     [=](sycl::nd_item<1> item) {
                         const int64_t idx = item.get_global_linear_id();
                         int local_count = 0;
                         if (idx < num_points) {
                             local_count =
                                     distances_ptr[sorted_indices_ptr[idx]] <=
                                             threshold
                                     ? 1
                                     : 0;
                         }
                         auto group = item.get_group();
                         const int group_count = sycl::reduce_over_group(
                                 group, local_count, sycl::plus<int>());
                         if (item.get_local_linear_id() == 0 &&
                             group_count > 0) {
                             sycl::atomic_ref<int64_t,
                                              sycl::memory_order::relaxed,
                                              sycl::memory_scope::device>
                                     count_ref(*count_ptr);
                             count_ref += group_count;
                         }
                     });
         }).wait_and_throw();

    return count.To(Device("CPU:0")).Item<int64_t>();
}

template <typename T, typename TIndex>
void GatherResultsSYCL(const Device& device,
                       const T* distances_ptr,
                       const TIndex* sorted_indices_ptr,
                       int64_t num_results,
                       TIndex index_offset,
                       TIndex* out_indices_ptr,
                       T* out_distances_ptr) {
    if (num_results == 0) {
        return;
    }

    sycl::queue queue =
            sy::SYCLContext::GetInstance().GetDefaultQueue(device);
    queue.parallel_for(sycl::range<1>(num_results), [=](sycl::id<1> id) {
             const int64_t i = id[0];
             const TIndex point_idx = sorted_indices_ptr[i];
             out_indices_ptr[i] = point_idx + index_offset;
             if (out_distances_ptr != nullptr) {
                 out_distances_ptr[i] = distances_ptr[point_idx];
             }
         }).wait_and_throw();
}

template <typename T, typename TIndex>
void ProcessQuerySYCL(const Tensor& points,
                      const Tensor& query,
                      int64_t num_results,
                      TIndex index_offset,
                      TIndex* out_indices_ptr,
                      T* out_distances_ptr) {
    const Device device = points.GetDevice();
    const int64_t num_points = points.GetShape(0);

    Tensor distances = Tensor::Empty({num_points}, points.GetDtype(), device);
    Tensor sorted_indices = Tensor::Arange(
            0, num_points, 1, Dtype::FromType<TIndex>(), device);

    ComputeSquaredDistancesSYCL<T>(device, points.GetDataPtr<T>(), num_points,
                                   points.GetShape(1), query.GetDataPtr<T>(),
                                   distances.GetDataPtr<T>());
    SortIndicesByDistanceSYCL<T, TIndex>(device, distances.GetDataPtr<T>(),
                                         sorted_indices.GetDataPtr<TIndex>(),
                                         num_points);
    GatherResultsSYCL<T, TIndex>(device, distances.GetDataPtr<T>(),
                                 sorted_indices.GetDataPtr<TIndex>(),
                                 num_results, index_offset, out_indices_ptr,
                                 out_distances_ptr);
}

template <typename T, typename TIndex>
int64_t CountQueryResultsWithinRadiusSYCL(const Tensor& points,
                                          const Tensor& query,
                                          T threshold) {
    const Device device = points.GetDevice();
    const int64_t num_points = points.GetShape(0);

    Tensor distances = Tensor::Empty({num_points}, points.GetDtype(), device);
    Tensor sorted_indices = Tensor::Arange(
            0, num_points, 1, Dtype::FromType<TIndex>(), device);

    ComputeSquaredDistancesSYCL<T>(device, points.GetDataPtr<T>(), num_points,
                                   points.GetShape(1), query.GetDataPtr<T>(),
                                   distances.GetDataPtr<T>());
    SortIndicesByDistanceSYCL<T, TIndex>(device, distances.GetDataPtr<T>(),
                                         sorted_indices.GetDataPtr<TIndex>(),
                                         num_points);
    return CountWithinRadiusSYCL<T, TIndex>(
            device, distances.GetDataPtr<T>(), sorted_indices.GetDataPtr<TIndex>(),
            num_points, threshold);
}

template <typename T, typename TIndex>
void GatherQueryResultsWithinRadiusSYCL(const Tensor& points,
                                        const Tensor& query,
                                        int64_t num_results,
                                        TIndex index_offset,
                                        TIndex* out_indices_ptr,
                                        T* out_distances_ptr) {
    ProcessQuerySYCL<T, TIndex>(points, query, num_results, index_offset,
                                out_indices_ptr, out_distances_ptr);
}

}  // namespace

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
    const int batch_size = points_row_splits.GetShape(0) - 1;
    std::vector<NeighborSearchAllocator<T, TIndex>> batch_output_allocators(
            batch_size, NeighborSearchAllocator<T, TIndex>(device));

    int64_t* neighbors_row_splits_ptr =
            neighbors_row_splits.GetDataPtr<int64_t>();
    int64_t last_neighbors_count = 0;
    int64_t batch_knn = 0;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int64_t point_begin = points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end = points_row_splits[batch_idx + 1].Item<int64_t>();
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

        for (int64_t q = 0; q < num_queries_i; ++q) {
            const Tensor query = queries_i.Slice(0, q, q + 1).Flatten();
            ProcessQuerySYCL<T, TIndex>(
                    points_i, query, batch_knn, TIndex(point_begin),
                    indices_ptr + q * batch_knn, distances_ptr + q * batch_knn);
        }
    }

    if (batch_size == 1) {
        neighbors_index =
                batch_output_allocators[0].NeighborsIndex().View({queries.GetShape(0),
                                                                  batch_knn});
        neighbors_distance = batch_output_allocators[0].NeighborsDistance().View(
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
                "SYCL fixed radius search does not support ignore_query_point.");
    }

    const Device device = points.GetDevice();
    const int64_t num_queries = queries.GetShape(0);
    const T threshold = static_cast<T>(radius * radius);
    std::vector<int64_t> counts(num_queries, 0);

    for (int batch_idx = 0; batch_idx < points_row_splits.GetShape(0) - 1;
         ++batch_idx) {
        const int64_t point_begin = points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end = points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);

        for (int64_t q = query_begin; q < query_end; ++q) {
            const Tensor query = queries.Slice(0, q, q + 1).Flatten();
            counts[q] = CountQueryResultsWithinRadiusSYCL<T, TIndex>(
                    points_i, query, threshold);
        }
    }

    std::vector<int64_t> row_splits(num_queries + 1, 0);
    for (int64_t q = 0; q < num_queries; ++q) {
        row_splits[q + 1] = row_splits[q] + counts[q];
    }
    neighbors_row_splits =
            Tensor(row_splits, {num_queries + 1}, Int64).To(device);

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
        const int64_t point_begin = points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end = points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);

        for (int64_t q = query_begin; q < query_end; ++q) {
            if (counts[q] == 0) {
                continue;
            }
            const Tensor query = queries.Slice(0, q, q + 1).Flatten();
            GatherQueryResultsWithinRadiusSYCL<T, TIndex>(
                    points_i, query, counts[q], TIndex(point_begin),
                    neighbors_index_ptr + row_splits[q],
                    return_distances ? neighbors_distance_ptr + row_splits[q]
                                     : nullptr);
        }
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

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
    const T threshold = static_cast<T>(radius * radius);

    NeighborSearchAllocator<T, TIndex> output_allocator(device);
    TIndex* neighbors_index_ptr;
    T* neighbors_distance_ptr;
    output_allocator.AllocIndices(&neighbors_index_ptr,
                                  num_queries * max_knn, TIndex(-1));
    output_allocator.AllocDistances(&neighbors_distance_ptr,
                                    num_queries * max_knn, T(0));

    std::vector<TIndex> counts(num_queries, 0);
    for (int batch_idx = 0; batch_idx < points_row_splits.GetShape(0) - 1;
         ++batch_idx) {
        const int64_t point_begin = points_row_splits[batch_idx].Item<int64_t>();
        const int64_t point_end = points_row_splits[batch_idx + 1].Item<int64_t>();
        const int64_t query_begin =
                queries_row_splits[batch_idx].Item<int64_t>();
        const int64_t query_end =
                queries_row_splits[batch_idx + 1].Item<int64_t>();
        const Tensor points_i = points.Slice(0, point_begin, point_end);

        for (int64_t q = query_begin; q < query_end; ++q) {
            const Tensor query = queries.Slice(0, q, q + 1).Flatten();
            const int64_t num_neighbors =
                    CountQueryResultsWithinRadiusSYCL<T, TIndex>(
                            points_i, query, threshold);
            counts[q] = static_cast<TIndex>(std::min<int64_t>(num_neighbors,
                                                              max_knn));
            if (counts[q] == 0) {
                continue;
            }
            GatherQueryResultsWithinRadiusSYCL<T, TIndex>(
                    points_i, query, counts[q], TIndex(point_begin),
                    neighbors_index_ptr + q * max_knn,
                    neighbors_distance_ptr + q * max_knn);
        }
    }

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
    neighbors_count =
            Tensor(counts, {num_queries}, Dtype::FromType<TIndex>()).To(device);
}

#define INSTANTIATE(T, TIndex)                                                \
    template void KnnSearchSYCL<T, TIndex>(                                   \
            const Tensor& points, const Tensor& points_row_splits,            \
            const Tensor& queries, const Tensor& queries_row_splits, int knn, \
            Tensor& neighbors_index, Tensor& neighbors_row_splits,            \
            Tensor& neighbors_distance);                                      \
    template void FixedRadiusSearchSYCL<T, TIndex>(                           \
            const Tensor& points, const Tensor& queries, double radius,       \
            const Tensor& points_row_splits, const Tensor& queries_row_splits, \
            const Tensor&, const Tensor&, const Tensor&, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,       \
            const bool sort, Tensor& neighbors_index,                         \
            Tensor& neighbors_row_splits, Tensor& neighbors_distance);        \
    template void HybridSearchSYCL<T, TIndex>(                                \
            const Tensor& points, const Tensor& queries, double radius,       \
            int max_knn, const Tensor& points_row_splits,                     \
            const Tensor& queries_row_splits, const Tensor&, const Tensor&,   \
            const Tensor&, const Metric metric, Tensor& neighbors_index,      \
            Tensor& neighbors_count, Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)

}  // namespace nns
}  // namespace core
}  // namespace open3d
