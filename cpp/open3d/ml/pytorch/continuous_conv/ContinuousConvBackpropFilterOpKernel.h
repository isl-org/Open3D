// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"
#include "torch/script.h"

template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvBackpropFilterCPU(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_importance,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const torch::Tensor& out_features_gradient,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        torch::Tensor& filter_backprop);

#ifdef BUILD_CUDA_MODULE
template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvBackpropFilterCUDA(
        const torch::Tensor& filters,
        const torch::Tensor& out_positions,
        const torch::Tensor& extents,
        const torch::Tensor& offset,
        const torch::Tensor& inp_positions,
        const torch::Tensor& inp_features,
        const torch::Tensor& inp_importance,
        const torch::Tensor& neighbors_index,
        const torch::Tensor& neighbors_importance,
        const torch::Tensor& neighbors_row_splits,
        const torch::Tensor& out_features_gradient,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        torch::Tensor& filter_backprop);
#endif
