// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------
//

#include <vector>

#include "ATen/cuda/CUDAContext.h"
#include "open3d/ml/impl/continuous_conv/ContinuousConvTranspose.cuh"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

using namespace open3d::ml::impl;

template <class TReal, class TIndex>
void ContinuousConvTransposeCUDA(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& out_importance,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_neighbors_index,
        const torch::Tensor& inp_neighbors_importance_sum,
        const torch::Tensor& inp_neighbors_row_splits,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const bool align_corners,
        const CoordinateMapping coordinate_mapping,
        const bool normalize,
        const InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        torch::Tensor& out_features) {
    const bool individual_extents = extents.size(0) > 1;
    const bool isotropic_extents = extents.size(1) == 1;
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
    CConvTransposeComputeFeaturesCUDA<TReal, TIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            out_features.data_ptr<TReal>(), filter_dims,
            filters.data_ptr<TReal>(), out_positions.size(0),
            out_positions.data_ptr<TReal>(),
            out_importance.size(0) ? out_importance.data_ptr<TReal>() : nullptr,
            inp_positions.size(0), inp_positions.data_ptr<TReal>(),
            inp_features.data_ptr<TReal>(),
            inp_neighbors_importance_sum.size(0)
                    ? inp_neighbors_importance_sum.data_ptr<TReal>()
                    : nullptr,
            inp_neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TReal>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(), extents.data_ptr<TReal>(),
            offset.data_ptr<TReal>(), interpolation, coordinate_mapping,
            align_corners, individual_extents, isotropic_extents, normalize);

    temp_size = std::max(
            std::min(size_t(max_temp_mem_MB) * 1024 * 1024, max_temp_size),
            temp_size);

    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually run the operation
    CConvTransposeComputeFeaturesCUDA<TReal, TIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            out_features.data_ptr<TReal>(), filter_dims,
            filters.data_ptr<TReal>(), out_positions.size(0),
            out_positions.data_ptr<TReal>(),
            out_importance.size(0) ? out_importance.data_ptr<TReal>() : nullptr,
            inp_positions.size(0), inp_positions.data_ptr<TReal>(),
            inp_features.data_ptr<TReal>(),
            inp_neighbors_importance_sum.size(0)
                    ? inp_neighbors_importance_sum.data_ptr<TReal>()
                    : nullptr,
            inp_neighbors_row_splits.data_ptr<int64_t>(),
            neighbors_index.size(0), neighbors_index.data_ptr<TIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TReal>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(), extents.data_ptr<TReal>(),
            offset.data_ptr<TReal>(), interpolation, coordinate_mapping,
            align_corners, individual_extents, isotropic_extents, normalize);
}
#define INSTANTIATE(TReal, TIndex)                                             \
    template void ContinuousConvTransposeCUDA<TReal, TIndex>(                  \
            const torch::Tensor& filters, const torch::Tensor& out_positions,  \
            const torch::Tensor& out_importance, const torch::Tensor& extents, \
            const torch::Tensor& offset, const torch::Tensor& inp_positions,   \
            const torch::Tensor& inp_features,                                 \
            const torch::Tensor& inp_neighbors_index,                          \
            const torch::Tensor& inp_neighbors_importance_sum,                 \
            const torch::Tensor& inp_neighbors_row_splits,                     \
            const torch::Tensor& neighbors_index,                              \
            const torch::Tensor& neighbors_importance,                         \
            const torch::Tensor& neighbors_row_splits,                         \
            const bool align_corners,                                          \
            const CoordinateMapping coordinate_mapping, const bool normalize,  \
            const InterpolationMode interpolation,                             \
            const int64_t max_temp_mem_MB, torch::Tensor& out_features);

INSTANTIATE(float, int32_t)
