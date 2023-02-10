// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

#include <vector>

#include "open3d/ml/impl/sparse_conv/SparseConvTransposeBackpropFilter.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvTransposeBackpropFilterCUDA(
        const torch::Tensor& filters,
        const torch::Tensor& out_importance,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_neighbors_importance_sum,
        const torch::Tensor& inp_neighbors_row_splits,
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

    auto stream = at::cuda::getCurrentCUDAStream();
    auto cuda_device_props = at::cuda::getCurrentDeviceProperties();
    const int texture_alignment = cuda_device_props->textureAlignment;

    auto device = filters.device();

    void* temp_ptr = nullptr;
    size_t temp_size = 0;
    size_t max_temp_size = 0;

    // determine temp_size
    SparseConvTransposeBackpropFilterCUDA<TFeat, TOut, TIndex, TKernelIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            filter_backprop.data_ptr<TOut>(), filter_dims,
            neighbors_row_splits.size(0) - 1,
            out_importance.size(0) ? out_importance.data_ptr<TFeat>() : nullptr,
            inp_features.size(0), inp_features.data_ptr<TFeat>(),
            inp_neighbors_importance_sum.size(0)
                    ? inp_neighbors_importance_sum.data_ptr<TFeat>()
                    : nullptr,
            inp_neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_kernel_index.data_ptr<TKernelIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(),
            out_features_gradient.data_ptr<TFeat>(), normalize);

    temp_size = std::max(
            std::min(size_t(max_temp_mem_MB) * 1024 * 1024, max_temp_size),
            temp_size);

    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually run the operation
    SparseConvTransposeBackpropFilterCUDA<TFeat, TOut, TIndex, TKernelIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            filter_backprop.data_ptr<TOut>(), filter_dims,
            neighbors_row_splits.size(0) - 1,
            out_importance.size(0) ? out_importance.data_ptr<TFeat>() : nullptr,
            inp_features.size(0), inp_features.data_ptr<TFeat>(),
            inp_neighbors_importance_sum.size(0)
                    ? inp_neighbors_importance_sum.data_ptr<TFeat>()
                    : nullptr,
            inp_neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_kernel_index.data_ptr<TKernelIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(),
            out_features_gradient.data_ptr<TFeat>(), normalize);
}
#define INSTANTIATE(TFeat, TOut, TIndex, TKernelIndex)                         \
    template void                                                              \
    SparseConvTransposeBackpropFilterCUDA<TFeat, TOut, TIndex, TKernelIndex>(  \
            const torch::Tensor& filters, const torch::Tensor& out_importance, \
            const torch::Tensor& inp_features,                                 \
            const torch::Tensor& inp_neighbors_importance_sum,                 \
            const torch::Tensor& inp_neighbors_row_splits,                     \
            const torch::Tensor& neighbors_index,                              \
            const torch::Tensor& neighbors_kernel_index,                       \
            const torch::Tensor& neighbors_importance,                         \
            const torch::Tensor& neighbors_row_splits,                         \
            const torch::Tensor& out_features_gradient, const bool normalize,  \
            const int64_t max_temp_mem_MB, torch::Tensor& filter_backprop);

INSTANTIATE(float, float, int32_t, uint8_t)
