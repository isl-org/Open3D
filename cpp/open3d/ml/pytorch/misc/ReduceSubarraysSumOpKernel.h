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
torch::Tensor ReduceSubarraysSumCPU(const torch::Tensor& values,
                                    const torch::Tensor& row_splits);

#ifdef BUILD_CUDA_MODULE
template <class T>
torch::Tensor ReduceSubarraysSumCUDA(const torch::Tensor& values,
                                     const torch::Tensor& row_splits);
#endif
