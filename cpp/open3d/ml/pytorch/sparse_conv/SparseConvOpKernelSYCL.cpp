// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <c10/xpu/XPUStream.h>
#include <torch/script.h>

#include <vector>

#include "open3d/ml/impl/sparse_conv/SparseConvSYCL.h"
#include "open3d/ml/pytorch/TorchHelper.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvSYCL(const torch::Tensor& filters,
                    const torch::Tensor& inp_features,
                    const torch::Tensor& inp_importance,
                    const torch::Tensor& neighbors_index,
                    const torch::Tensor& neighbors_kernel_index,
                    const torch::Tensor& neighbors_importance,
                    const torch::Tensor& neighbors_row_splits,
                    const bool normalize,
                    const int64_t max_temp_mem_MB,
                        const bool allow_tf32,
                    torch::Tensor& out_features) {
    std::vector<int> filter_dims;
    for (auto d : filters.sizes()) {
        filter_dims.push_back(d);
    }

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    // texture_alignment is a CUDA-specific concept; a small, safe default
    // alignment is used for the SYCL MemoryAllocation bookkeeping instead.
    const int texture_alignment = 64;

    auto device = filters.device();

    void* temp_ptr = nullptr;
    size_t temp_size = 0;
    size_t max_temp_size = 0;

    // determine temp_size
    SparseConvComputeFeaturesSYCL<TFeat, TOut, TIndex, TKernelIndex>(
            queue, temp_ptr, temp_size, max_temp_size, texture_alignment,
            out_features.data_ptr<TOut>(), filter_dims,
            filters.data_ptr<TFeat>(), neighbors_row_splits.size(0) - 1,
            inp_features.size(0), inp_features.data_ptr<TFeat>(),
            inp_importance.size(0) ? inp_importance.data_ptr<TFeat>() : nullptr,
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_kernel_index.data_ptr<TKernelIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(), normalize, allow_tf32);

    temp_size = std::max(
            std::min(size_t(max_temp_mem_MB) * 1024 * 1024, max_temp_size),
            temp_size);

    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually run the operation
    SparseConvComputeFeaturesSYCL<TFeat, TOut, TIndex, TKernelIndex>(
            queue, temp_ptr, temp_size, max_temp_size, texture_alignment,
            out_features.data_ptr<TOut>(), filter_dims,
            filters.data_ptr<TFeat>(), neighbors_row_splits.size(0) - 1,
            inp_features.size(0), inp_features.data_ptr<TFeat>(),
            inp_importance.size(0) ? inp_importance.data_ptr<TFeat>() : nullptr,
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_kernel_index.data_ptr<TKernelIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(), normalize, allow_tf32);
}
#define INSTANTIATE(TFeat, TOut, TReal, TIndex)                              \
    template void SparseConvSYCL<TFeat, TOut, TReal, TIndex>(                \
            const torch::Tensor& filters, const torch::Tensor& inp_features, \
            const torch::Tensor& inp_importance,                             \
            const torch::Tensor& neighbors_index,                            \
            const torch::Tensor& neighbors_kernel_index,                     \
            const torch::Tensor& neighbors_importance,                       \
            const torch::Tensor& neighbors_row_splits, const bool normalize, \
            const int64_t max_temp_mem_MB, const bool allow_tf32,             \
            torch::Tensor& out_features);

INSTANTIATE(float, float, int32_t, uint8_t)
