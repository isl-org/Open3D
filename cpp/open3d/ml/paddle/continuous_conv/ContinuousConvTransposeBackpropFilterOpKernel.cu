// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#include <sstream>
#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTransposeBackpropFilter.cuh"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvTransposeBackpropFilterCUDA(
        const paddle::Tensor& filters,
        const paddle::Tensor& out_positions,
        const paddle::Tensor& out_importance,
        const paddle::Tensor& extents,
        const paddle::Tensor& offset,
        const paddle::Tensor& inp_positions,
        const paddle::Tensor& inp_features,
        const paddle::Tensor& inp_neighbors_importance_sum,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& neighbors_index,
        const paddle::Tensor& neighbors_importance,
        const paddle::Tensor& neighbors_row_splits,
        const paddle::Tensor& out_features_gradient,
        const bool align_corners,
        const CoordinateMapping coordinate_mapping,
        const bool normalize,
        const InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        paddle::Tensor& filter_backprop) {
    const bool individual_extents = extents.shape()[0] > 1;
    const bool isotropic_extents = extents.shape()[1] == 1;
    std::vector<int> filter_dims;
    for (auto d : filters.shape()) {
        filter_dims.push_back(static_cast<int>(d));
    }

    auto stream = filters.stream();
    // -1 means current global place
    auto cuda_device_props = phi::backends::gpu::GetDeviceProperties(-1);
    const int texture_alignment = cuda_device_props.textureAlignment;

    auto place = filters.place();

    void* temp_ptr = nullptr;
    size_t temp_size = 0;
    size_t max_temp_size = 0;

    // determine temp_size
    CConvTransposeBackpropFilterCUDA<TFeat, TOut, TReal, TIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            filter_backprop.data<TOut>(), filter_dims, out_positions.shape()[0],
            out_positions.data<TReal>(),
            out_importance.shape()[0] ? out_importance.data<TFeat>() : nullptr,
            inp_positions.shape()[0], inp_positions.data<TReal>(),
            inp_features.data<TFeat>(),
            inp_neighbors_importance_sum.shape()[0]
                    ? inp_neighbors_importance_sum.data<TFeat>()
                    : nullptr,
            inp_neighbors_row_splits.data<int64_t>(),
            neighbors_index.shape()[0], neighbors_index.data<TIndex>(),
            neighbors_importance.shape()[0] ? neighbors_importance.data<TFeat>()
                                            : nullptr,
            neighbors_row_splits.data<int64_t>(), extents.data<TReal>(),
            offset.data<TReal>(), out_features_gradient.data<TFeat>(),
            interpolation, coordinate_mapping, align_corners,
            individual_extents, isotropic_extents, normalize);

    temp_size = std::max(
            std::min(static_cast<size_t>(max_temp_mem_MB) * 1024 * 1024,
                     max_temp_size),
            temp_size);

    auto temp_tensor = CreateTempTensor(temp_size, place, &temp_ptr);

    // actually run the operation
    CConvTransposeBackpropFilterCUDA<TFeat, TOut, TReal, TIndex>(
            stream, temp_ptr, temp_size, max_temp_size, texture_alignment,
            filter_backprop.data<TOut>(), filter_dims, out_positions.shape()[0],
            out_positions.data<TReal>(),
            out_importance.shape()[0] ? out_importance.data<TFeat>() : nullptr,
            inp_positions.shape()[0], inp_positions.data<TReal>(),
            inp_features.data<TFeat>(),
            inp_neighbors_importance_sum.shape()[0]
                    ? inp_neighbors_importance_sum.data<TFeat>()
                    : nullptr,
            inp_neighbors_row_splits.data<int64_t>(),
            neighbors_index.shape()[0], neighbors_index.data<TIndex>(),
            neighbors_importance.shape()[0] ? neighbors_importance.data<TFeat>()
                                            : nullptr,
            neighbors_row_splits.data<int64_t>(), extents.data<TReal>(),
            offset.data<TReal>(), out_features_gradient.data<TFeat>(),
            interpolation, coordinate_mapping, align_corners,
            individual_extents, isotropic_extents, normalize);
}
#define INSTANTIATE(TFeat, TOut, TReal, TIndex)                               \
    template void                                                             \
    ContinuousConvTransposeBackpropFilterCUDA<TFeat, TOut, TReal, TIndex>(    \
            const paddle::Tensor& filters,                                    \
            const paddle::Tensor& out_positions,                              \
            const paddle::Tensor& out_importance,                             \
            const paddle::Tensor& extents, const paddle::Tensor& offset,      \
            const paddle::Tensor& inp_positions,                              \
            const paddle::Tensor& inp_features,                               \
            const paddle::Tensor& inp_neighbors_importance_sum,               \
            const paddle::Tensor& inp_neighbors_row_splits,                   \
            const paddle::Tensor& neighbors_index,                            \
            const paddle::Tensor& neighbors_importance,                       \
            const paddle::Tensor& neighbors_row_splits,                       \
            const paddle::Tensor& out_features_gradient,                      \
            const bool align_corners,                                         \
            const CoordinateMapping coordinate_mapping, const bool normalize, \
            const InterpolationMode interpolation,                            \
            const int64_t max_temp_mem_MB, paddle::Tensor& filter_backprop);

INSTANTIATE(float, float, float, int32_t)
