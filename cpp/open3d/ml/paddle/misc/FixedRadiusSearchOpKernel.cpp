// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/paddle/PaddleHelper.h"
#include "open3d/ml/paddle/misc/NeighborSearchAllocator.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchCPU(const paddle::Tensor& points,
                          const paddle::Tensor& queries,
                          double radius,
                          const paddle::Tensor& points_row_splits,
                          const paddle::Tensor& queries_row_splits,
                          const paddle::Tensor& hash_table_splits,
                          const paddle::Tensor& hash_table_index,
                          const paddle::Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          paddle::Tensor& neighbors_index,
                          paddle::Tensor& neighbors_row_splits,
                          paddle::Tensor& neighbors_distance) {
    NeighborSearchAllocator<T, TIndex> output_allocator(points.place());

    impl::FixedRadiusSearchCPU<T, TIndex>(
            neighbors_row_splits.data<int64_t>(), points.shape()[0],
            points.data<T>(), queries.shape()[0], queries.data<T>(), T(radius),
            points_row_splits.shape()[0], points_row_splits.data<int64_t>(),
            queries_row_splits.shape()[0], queries_row_splits.data<int64_t>(),
            reinterpret_cast<uint32_t*>(
                    const_cast<int32_t*>(hash_table_splits.data<int32_t>())),
            hash_table_cell_splits.shape()[0],
            reinterpret_cast<uint32_t*>(const_cast<int32_t*>(
                    hash_table_cell_splits.data<int32_t>())),
            reinterpret_cast<uint32_t*>(
                    const_cast<int32_t*>(hash_table_index.data<int32_t>())),
            metric, ignore_query_point, return_distances, output_allocator);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                 \
    template void FixedRadiusSearchCPU<T, TIndex>(                             \
            const paddle::Tensor& points, const paddle::Tensor& queries,       \
            double radius, const paddle::Tensor& points_row_splits,            \
            const paddle::Tensor& queries_row_splits,                          \
            const paddle::Tensor& hash_table_splits,                           \
            const paddle::Tensor& hash_table_index,                            \
            const paddle::Tensor& hash_table_cell_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,        \
            paddle::Tensor& neighbors_index,                                   \
            paddle::Tensor& neighbors_row_splits,                              \
            paddle::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)
