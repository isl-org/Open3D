// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/ml/paddle/PaddleHelper.h"

template <class T>
void BuildSpatialHashTableCPU(const paddle::Tensor& points,
                              double radius,
                              const paddle::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              paddle::Tensor& hash_table_index,
                              paddle::Tensor& hash_table_cell_splits) {
    open3d::core::nns::impl::BuildSpatialHashTableCPU(
            points.shape()[0], points.data<T>(), T(radius),
            points_row_splits.shape()[0], points_row_splits.data<int64_t>(),
            hash_table_splits.data(), hash_table_cell_splits.shape()[0],
            reinterpret_cast<uint32_t*>(const_cast<int32_t*>(
                    hash_table_cell_splits.data<int32_t>())),
            reinterpret_cast<uint32_t*>(
                    const_cast<int32_t*>(hash_table_index.data<int32_t>())));
}
#define INSTANTIATE(T)                                            \
    template void BuildSpatialHashTableCPU<T>(                    \
            const paddle::Tensor&, double, const paddle::Tensor&, \
            const std::vector<uint32_t>&, paddle::Tensor&, paddle::Tensor&);

INSTANTIATE(float)
INSTANTIATE(double)
