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
paddle::Tensor RaggedToDenseCPU(const paddle::Tensor& values,
                                const paddle::Tensor& row_splits,
                                const int64_t out_col_size,
                                const paddle::Tensor& default_value);

#ifdef BUILD_CUDA_MODULE
template <class T>
paddle::Tensor RaggedToDenseCUDA(const paddle::Tensor& values,
                                 const paddle::Tensor& row_splits,
                                 const int64_t out_col_size,
                                 const paddle::Tensor& default_value);
#endif
