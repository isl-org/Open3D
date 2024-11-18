// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <vector>

#include "open3d/ml/paddle/PaddleHelper.h"

template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvBackpropFilterCPU(const paddle::Tensor& filters,
                                 const paddle::Tensor& inp_features,
                                 const paddle::Tensor& inp_importance,
                                 const paddle::Tensor& neighbors_index,
                                 const paddle::Tensor& neighbors_kernel_index,
                                 const paddle::Tensor& neighbors_importance,
                                 const paddle::Tensor& neighbors_row_splits,
                                 const paddle::Tensor& out_features_gradient,
                                 const bool normalize,
                                 const int64_t max_temp_mem_MB,
                                 paddle::Tensor& filter_backprop);

#ifdef BUILD_CUDA_MODULE
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvBackpropFilterCUDA(const paddle::Tensor& filters,
                                  const paddle::Tensor& inp_features,
                                  const paddle::Tensor& inp_importance,
                                  const paddle::Tensor& neighbors_index,
                                  const paddle::Tensor& neighbors_kernel_index,
                                  const paddle::Tensor& neighbors_importance,
                                  const paddle::Tensor& neighbors_row_splits,
                                  const paddle::Tensor& out_features_gradient,
                                  const bool normalize,
                                  const int64_t max_temp_mem_MB,
                                  paddle::Tensor& filter_backprop);
#endif
