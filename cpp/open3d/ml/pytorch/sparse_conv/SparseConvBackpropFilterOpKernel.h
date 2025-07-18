// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include <torch/script.h>

#include <vector>

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
                                 torch::Tensor& filter_backprop);

#ifdef BUILD_CUDA_MODULE
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvBackpropFilterCUDA(const torch::Tensor& filters,
                                  const torch::Tensor& inp_features,
                                  const torch::Tensor& inp_importance,
                                  const torch::Tensor& neighbors_index,
                                  const torch::Tensor& neighbors_kernel_index,
                                  const torch::Tensor& neighbors_importance,
                                  const torch::Tensor& neighbors_row_splits,
                                  const torch::Tensor& out_features_gradient,
                                  const bool normalize,
                                  const int64_t max_temp_mem_MB,
                                  torch::Tensor& filter_backprop);
#endif
