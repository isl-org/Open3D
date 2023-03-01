// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <torch/script.h>

#include <vector>

#include "open3d/ml/impl/sparse_conv/SparseConvBackpropFilter.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvBackpropFilterCPU(const torch::Tensor& filters,
                                 const torch::Tensor& inp_features,
                                 const torch::Tensor& inp_importance,
                                 const torch::Tensor& neighbors_index,
                                 const torch::Tensor& neighbors_kernel_index,
                                 const torch::Tensor& neighbors_importance,
                                 const torch::Tensor& neighbors_row_splits,
                                 const torch::Tensor& out_features_gradient,
                                 const bool normalize,
                                 const int64_t max_temp_mem_MB,
                                 torch::Tensor& filter_backprop) {
    std::vector<int> filter_dims;
    for (auto d : filters.sizes()) {
        filter_dims.push_back(d);
    }
    SparseConvBackpropFilterCPU<TFeat, TOut, TIndex, TKernelIndex>(
            filter_backprop.data_ptr<TOut>(), filter_dims,
            neighbors_row_splits.size(0) - 1, inp_features.size(0),
            inp_features.data_ptr<TFeat>(),
            inp_importance.size(0) ? inp_importance.data_ptr<TFeat>() : nullptr,
            (TIndex*)neighbors_index.data_ptr<TIndex>(),
            (TKernelIndex*)neighbors_kernel_index.data_ptr<TKernelIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(),
            out_features_gradient.data_ptr<TFeat>(), normalize);
}
#define INSTANTIATE(TFeat, TOut, TIndex, TKernelIndex)                        \
    template void                                                             \
    SparseConvBackpropFilterCPU<TFeat, TOut, TIndex, TKernelIndex>(           \
            const torch::Tensor& filters, const torch::Tensor& inp_features,  \
            const torch::Tensor& inp_importance,                              \
            const torch::Tensor& neighbors_index,                             \
            const torch::Tensor& neighbors_kernel_index,                      \
            const torch::Tensor& neighbors_importance,                        \
            const torch::Tensor& neighbors_row_splits,                        \
            const torch::Tensor& out_features_gradient, const bool normalize, \
            const int64_t max_temp_mem_MB, torch::Tensor& filter_backprop);

INSTANTIATE(float, float, int32_t, uint8_t)
