// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/core/nns/FixedRadiusSearchImpl.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

template <class T>
void BuildSpatialHashTableCPU(const torch::Tensor& points,
                              double radius,
                              const torch::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              torch::Tensor& hash_table_index,
                              torch::Tensor& hash_table_cell_splits) {
    open3d::core::nns::impl::BuildSpatialHashTableCPU(
            points.size(0), points.data_ptr<T>(), T(radius),
            points_row_splits.size(0), points_row_splits.data_ptr<int64_t>(),
            hash_table_splits.data(), hash_table_cell_splits.size(0),
            (uint32_t*)hash_table_cell_splits.data_ptr<int32_t>(),
            (uint32_t*)hash_table_index.data_ptr<int32_t>());
}
#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableCPU<T>(                  \
            const torch::Tensor&, double, const torch::Tensor&, \
            const std::vector<uint32_t>&, torch::Tensor&, torch::Tensor&);

INSTANTIATE(float)
INSTANTIATE(double)
