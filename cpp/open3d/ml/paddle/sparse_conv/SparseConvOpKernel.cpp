// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/impl/sparse_conv/SparseConv.h"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvCPU(const paddle::Tensor& filters,
                   const paddle::Tensor& inp_features,
                   const paddle::Tensor& inp_importance,
                   const paddle::Tensor& neighbors_index,
                   const paddle::Tensor& neighbors_kernel_index,
                   const paddle::Tensor& neighbors_importance,
                   const paddle::Tensor& neighbors_row_splits,
                   const bool normalize,
                   const int64_t max_temp_mem_MB,
                   paddle::Tensor& out_features) {
    std::vector<int> filter_dims;
    for (auto d : filters.shape()) {
        filter_dims.push_back(static_cast<int>(d));
    }
    SparseConvComputeFeaturesCPU<TFeat, TOut, TIndex, TKernelIndex>(
            out_features.data<TOut>(), filter_dims, filters.data<TFeat>(),
            neighbors_row_splits.shape()[0] - 1, inp_features.shape()[0],
            inp_features.data<TFeat>(),
            inp_importance.shape()[0] ? inp_importance.data<TFeat>() : nullptr,
            neighbors_index.shape()[0], neighbors_index.data<TIndex>(),
            neighbors_kernel_index.data<TKernelIndex>(),
            neighbors_importance.shape()[0] ? neighbors_importance.data<TFeat>()
                                            : nullptr,
            neighbors_row_splits.data<int64_t>(), normalize);
}
#define INSTANTIATE(TFeat, TOut, TIndex, TKernelIndex)                         \
    template void SparseConvCPU<TFeat, TOut, TIndex, TKernelIndex>(            \
            const paddle::Tensor& filters, const paddle::Tensor& inp_features, \
            const paddle::Tensor& inp_importance,                              \
            const paddle::Tensor& neighbors_index,                             \
            const paddle::Tensor& neighbors_kernel_index,                      \
            const paddle::Tensor& neighbors_importance,                        \
            const paddle::Tensor& neighbors_row_splits, const bool normalize,  \
            const int64_t max_temp_mem_MB, paddle::Tensor& out_features);

INSTANTIATE(float, float, int32_t, uint8_t)
