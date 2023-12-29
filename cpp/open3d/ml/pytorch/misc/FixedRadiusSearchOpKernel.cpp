// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/core/nns/NeighborSearchCommon.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "open3d/ml/pytorch/misc/NeighborSearchAllocator.h"
#include "torch/script.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchCPU(const torch::Tensor& points,
                          const torch::Tensor& queries,
                          double radius,
                          const torch::Tensor& points_row_splits,
                          const torch::Tensor& queries_row_splits,
                          const torch::Tensor& hash_table_splits,
                          const torch::Tensor& hash_table_index,
                          const torch::Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          torch::Tensor& neighbors_index,
                          torch::Tensor& neighbors_row_splits,
                          torch::Tensor& neighbors_distance) {
    NeighborSearchAllocator<T, TIndex> output_allocator(
            points.device().type(), points.device().index());

    impl::FixedRadiusSearchCPU<T, TIndex>(
            neighbors_row_splits.data_ptr<int64_t>(), points.size(0),
            points.data_ptr<T>(), queries.size(0), queries.data_ptr<T>(),
            T(radius), points_row_splits.size(0),
            points_row_splits.data_ptr<int64_t>(), queries_row_splits.size(0),
            queries_row_splits.data_ptr<int64_t>(),
            (uint32_t*)hash_table_splits.data_ptr<int32_t>(),
            hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>(), metric,
            ignore_query_point, return_distances, output_allocator);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                \
    template void FixedRadiusSearchCPU<T, TIndex>(                            \
            const torch::Tensor& points, const torch::Tensor& queries,        \
            double radius, const torch::Tensor& points_row_splits,            \
            const torch::Tensor& queries_row_splits,                          \
            const torch::Tensor& hash_table_splits,                           \
            const torch::Tensor& hash_table_index,                            \
            const torch::Tensor& hash_table_cell_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,       \
            torch::Tensor& neighbors_index,                                   \
            torch::Tensor& neighbors_row_splits,                              \
            torch::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)
