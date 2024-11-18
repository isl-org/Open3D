// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/impl/sparse_conv/SparseConvTransposeBackpropFilter.h"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvTransposeBackpropFilterCPU(
        const paddle::Tensor& filters,
        const paddle::Tensor& out_importance,
        const paddle::Tensor& inp_features,
        const paddle::Tensor& inp_neighbors_importance_sum,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& neighbors_index,
        const paddle::Tensor& neighbors_kernel_index,
        const paddle::Tensor& neighbors_importance,
        const paddle::Tensor& neighbors_row_splits,
        const paddle::Tensor& out_features_gradient,
        const bool normalize,
        const int64_t max_temp_mem_MB,
        paddle::Tensor& filter_backprop) {
    std::vector<int> filter_dims;
    for (auto d : filters.shape()) {
        filter_dims.push_back(static_cast<int>(d));
    }
    SparseConvTransposeBackpropFilterCPU<TFeat, TOut, TIndex, TKernelIndex>(
            filter_backprop.data<TOut>(), filter_dims,
            neighbors_row_splits.shape()[0] - 1,
            out_importance.shape()[0] ? out_importance.data<TFeat>() : nullptr,
            inp_features.shape()[0], inp_features.data<TFeat>(),
            inp_neighbors_importance_sum.shape()[0]
                    ? inp_neighbors_importance_sum.data<TFeat>()
                    : nullptr,
            inp_neighbors_row_splits.data<int64_t>(),
            neighbors_index.data<TIndex>(),
            neighbors_kernel_index.data<TKernelIndex>(),
            neighbors_importance.shape()[0] ? neighbors_importance.data<TFeat>()
                                            : nullptr,
            neighbors_row_splits.data<int64_t>(),
            out_features_gradient.data<TFeat>(), normalize);
}
#define INSTANTIATE(TFeat, TOut, TIndex, TKernelIndex)                         \
    template void                                                              \
    SparseConvTransposeBackpropFilterCPU<TFeat, TOut, TIndex, TKernelIndex>(   \
            const paddle::Tensor& filters,                                     \
            const paddle::Tensor& out_importance,                              \
            const paddle::Tensor& inp_features,                                \
            const paddle::Tensor& inp_neighbors_importance_sum,                \
            const paddle::Tensor& inp_neighbors_row_splits,                    \
            const paddle::Tensor& neighbors_index,                             \
            const paddle::Tensor& neighbors_kernel_index,                      \
            const paddle::Tensor& neighbors_importance,                        \
            const paddle::Tensor& neighbors_row_splits,                        \
            const paddle::Tensor& out_features_gradient, const bool normalize, \
            const int64_t max_temp_mem_MB, paddle::Tensor& filter_backprop);

INSTANTIATE(float, float, int32_t, uint8_t)
