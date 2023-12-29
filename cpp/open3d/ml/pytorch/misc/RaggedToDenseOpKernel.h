// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "torch/script.h"

template <class T>
torch::Tensor RaggedToDenseCPU(const torch::Tensor& values,
                               const torch::Tensor& row_splits,
                               const int64_t out_col_size,
                               const torch::Tensor& default_value);

#ifdef BUILD_CUDA_MODULE
template <class T>
torch::Tensor RaggedToDenseCUDA(const torch::Tensor& values,
                                const torch::Tensor& row_splits,
                                const int64_t out_col_size,
                                const torch::Tensor& default_value);
#endif
