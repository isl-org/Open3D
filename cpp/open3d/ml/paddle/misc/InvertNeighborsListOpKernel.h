// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "open3d/ml/paddle/PaddleHelper.h"

template <class TIndex, class TAttr>
std::vector<paddle::Tensor> InvertNeighborsListCPU(
        int64_t num_points,
        const paddle::Tensor& inp_neighbors_index,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& inp_neighbors_attributes);

#ifdef BUILD_CUDA_MODULE
template <class TIndex, class TAttr>
std::vector<paddle::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        const paddle::Tensor& inp_neighbors_index,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& inp_neighbors_attributes);
#endif
