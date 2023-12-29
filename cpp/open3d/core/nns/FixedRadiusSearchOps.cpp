// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/Tensor.h"
#include "open3d/core/nns/FixedRadiusIndex.h"
#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/core/nns/NeighborSearchAllocator.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace nns {

template <class T>
void BuildSpatialHashTableCPU(const Tensor& points,
                              double radius,
                              const Tensor& points_row_splits,
                              const Tensor& hash_table_splits,
                              Tensor& hash_table_index,
                              Tensor& hash_table_cell_splits) {
    impl::BuildSpatialHashTableCPU(
            points.GetShape()[0], points.GetDataPtr<T>(), T(radius),
            points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>());
}

template <class T, class TIndex>
void FixedRadiusSearchCPU(const Tensor& points,
                          const Tensor& queries,
                          double radius,
                          const Tensor& points_row_splits,
                          const Tensor& queries_row_splits,
                          const Tensor& hash_table_splits,
                          const Tensor& hash_table_index,
                          const Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          const bool sort,
                          Tensor& neighbors_index,
                          Tensor& neighbors_row_splits,
                          Tensor& neighbors_distance) {
    Device device = points.GetDevice();
    NeighborSearchAllocator<T, TIndex> output_allocator(device);

    impl::FixedRadiusSearchCPU<T, TIndex>(
            neighbors_row_splits.GetDataPtr<int64_t>(), points.GetShape()[0],
            points.GetDataPtr<T>(), queries.GetShape()[0],
            queries.GetDataPtr<T>(), T(radius), points_row_splits.GetShape()[0],
            points_row_splits.GetDataPtr<int64_t>(),
            queries_row_splits.GetShape()[0],
            queries_row_splits.GetDataPtr<int64_t>(),
            hash_table_splits.GetDataPtr<uint32_t>(),
            hash_table_cell_splits.GetShape()[0],
            hash_table_cell_splits.GetDataPtr<uint32_t>(),
            hash_table_index.GetDataPtr<uint32_t>(), metric, ignore_query_point,
            return_distances, output_allocator);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

template <class T, class TIndex>
void HybridSearchCPU(const Tensor& points,
                     const Tensor& queries,
                     double radius,
                     int max_knn,
                     const Tensor& points_row_splits,
                     const Tensor& queries_row_splits,
                     const Tensor& hash_table_splits,
                     const Tensor& hash_table_index,
                     const Tensor& hash_table_cell_splits,
                     const Metric metric,
                     Tensor& neighbors_index,
                     Tensor& neighbors_count,
                     Tensor& neighbors_distance) {
    utility::LogError("Not implemented.");
}

#define INSTANTIATE_BUILD(T)                                                  \
    template void BuildSpatialHashTableCPU<T>(                                \
            const Tensor& points, double radius,                              \
            const Tensor& points_row_splits, const Tensor& hash_table_splits, \
            Tensor& hash_table_index, Tensor& hash_table_cell_splits);

#define INSTANTIATE_RADIUS(T, TIndex)                                          \
    template void FixedRadiusSearchCPU<T, TIndex>(                             \
            const Tensor& points, const Tensor& queries, double radius,        \
            const Tensor& points_row_splits, const Tensor& queries_row_splits, \
            const Tensor& hash_table_splits, const Tensor& hash_table_index,   \
            const Tensor& hash_table_cell_splits, const Metric metric,         \
            const bool ignore_query_point, const bool return_distances,        \
            const bool sort, Tensor& neighbors_index,                          \
            Tensor& neighbors_row_splits, Tensor& neighbors_distance);

#define INSTANTIATE_HYBRID(T, TIndex)                                          \
    template void HybridSearchCPU<T, TIndex>(                                  \
            const Tensor& points, const Tensor& queries, double radius,        \
            int max_knn, const Tensor& points_row_splits,                      \
            const Tensor& queries_row_splits, const Tensor& hash_table_splits, \
            const Tensor& hash_table_index,                                    \
            const Tensor& hash_table_cell_splits, const Metric metric,         \
            Tensor& neighbors_index, Tensor& neighbors_count,                  \
            Tensor& neighbors_distance);

INSTANTIATE_BUILD(float)
INSTANTIATE_BUILD(double)

INSTANTIATE_RADIUS(float, int32_t)
INSTANTIATE_RADIUS(float, int64_t)
INSTANTIATE_RADIUS(double, int32_t)
INSTANTIATE_RADIUS(double, int64_t)

INSTANTIATE_HYBRID(float, int32_t)
INSTANTIATE_HYBRID(float, int64_t)
INSTANTIATE_HYBRID(double, int32_t)
INSTANTIATE_HYBRID(double, int64_t)
}  // namespace nns
}  // namespace core
}  // namespace open3d
