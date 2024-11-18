// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "paddle/extension.h"

template <class T>
paddle::Tensor ReduceSubarraysSumCPU(const paddle::Tensor& values,
                                     const paddle::Tensor& row_splits);

#ifdef BUILD_CUDA_MODULE
template <class T>
paddle::Tensor ReduceSubarraysSumCUDA(const paddle::Tensor& values,
                                      const paddle::Tensor& row_splits);
#endif
