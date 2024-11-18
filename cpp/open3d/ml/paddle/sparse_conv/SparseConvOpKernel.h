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
void SparseConvCPU(const paddle::Tensor& filters,
                   const paddle::Tensor& inp_features,
                   const paddle::Tensor& inp_importance,
                   const paddle::Tensor& neighbors_index,
                   const paddle::Tensor& neighbors_kernel_index,
                   const paddle::Tensor& neighbors_importance,
                   const paddle::Tensor& neighbors_row_splits,
                   const bool normalize,
                   const int64_t max_temp_mem_MB,
                   paddle::Tensor& out_features);

#ifdef BUILD_CUDA_MODULE
template <class TFeat, class TOut, class TIndex, class TKernelIndex>
void SparseConvCUDA(const paddle::Tensor& filters,
                    const paddle::Tensor& inp_features,
                    const paddle::Tensor& inp_importance,
                    const paddle::Tensor& neighbors_index,
                    const paddle::Tensor& neighbors_kernel_index,
                    const paddle::Tensor& neighbors_importance,
                    const paddle::Tensor& neighbors_row_splits,
                    const bool normalize,
                    const int64_t max_temp_mem_MB,
                    paddle::Tensor& out_features);
#endif
