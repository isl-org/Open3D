// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConv.h"
#include "torch/script.h"

using namespace open3d::ml::impl;

template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvCPU(const torch::Tensor& filters,
                       const torch::Tensor& out_positions,
                       const torch::Tensor& extents,
                       const torch::Tensor& offset,
                       const torch::Tensor& inp_positions,
                       const torch::Tensor& inp_features,
                       const torch::Tensor& inp_importance,
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
    CConvComputeFeaturesCPU<TFeat, TOut, TReal, TIndex>(
            out_features.data_ptr<TOut>(), filter_dims,
            filters.data_ptr<TFeat>(), out_positions.size(0),
            out_positions.data_ptr<TReal>(), inp_positions.size(0),
            inp_positions.data_ptr<TReal>(), inp_features.data_ptr<TFeat>(),
            inp_importance.size(0) ? inp_importance.data_ptr<TFeat>() : nullptr,
            neighbors_index.size(0),
            (TIndex*)neighbors_index.data_ptr<TIndex>(),
            neighbors_importance.size(0)
                    ? neighbors_importance.data_ptr<TFeat>()
                    : nullptr,
            neighbors_row_splits.data_ptr<int64_t>(), extents.data_ptr<TReal>(),
            offset.data_ptr<TReal>(), interpolation, coordinate_mapping,
            align_corners, individual_extents, isotropic_extents, normalize);
}
#define INSTANTIATE(TFeat, TOut, TReal, TIndex)                               \
    template void ContinuousConvCPU<TFeat, TOut, TReal, TIndex>(              \
            const torch::Tensor& filters, const torch::Tensor& out_positions, \
            const torch::Tensor& extents, const torch::Tensor& offset,        \
            const torch::Tensor& inp_positions,                               \
            const torch::Tensor& inp_features,                                \
            const torch::Tensor& inp_importance,                              \
            const torch::Tensor& neighbors_index,                             \
            const torch::Tensor& neighbors_importance,                        \
            const torch::Tensor& neighbors_row_splits,                        \
            const bool align_corners,                                         \
            const CoordinateMapping coordinate_mapping, const bool normalize, \
            const InterpolationMode interpolation,                            \
            const int64_t max_temp_mem_MB, torch::Tensor& out_features);

INSTANTIATE(float, float, float, int32_t)
