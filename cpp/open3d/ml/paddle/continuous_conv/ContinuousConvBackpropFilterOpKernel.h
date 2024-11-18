// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <vector>

#include "open3d/ml/impl/continuous_conv/ContinuousConvTypes.h"

template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvBackpropFilterCPU(
        const paddle::Tensor& filters,
        const paddle::Tensor& out_positions,
        const paddle::Tensor& extents,
        const paddle::Tensor& offset,
        const paddle::Tensor& inp_positions,
        const paddle::Tensor& inp_features,
        const paddle::Tensor& inp_importance,
        const paddle::Tensor& neighbors_index,
        const paddle::Tensor& neighbors_importance,
        const paddle::Tensor& neighbors_row_splits,
        const paddle::Tensor& out_features_gradient,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        paddle::Tensor& filter_backprop);

#ifdef BUILD_CUDA_MODULE
template <class TFeat, class TOut, class TReal, class TIndex>
void ContinuousConvBackpropFilterCUDA(
        const paddle::Tensor& filters,
        const paddle::Tensor& out_positions,
        const paddle::Tensor& extents,
        const paddle::Tensor& offset,
        const paddle::Tensor& inp_positions,
        const paddle::Tensor& inp_features,
        const paddle::Tensor& inp_importance,
        const paddle::Tensor& neighbors_index,
        const paddle::Tensor& neighbors_importance,
        const paddle::Tensor& neighbors_row_splits,
        const paddle::Tensor& out_features_gradient,
        const bool align_corners,
        const open3d::ml::impl::CoordinateMapping coordinate_mapping,
        const bool normalize,
        const open3d::ml::impl::InterpolationMode interpolation,
        const int64_t max_temp_mem_MB,
        paddle::Tensor& filter_backprop);
#endif
