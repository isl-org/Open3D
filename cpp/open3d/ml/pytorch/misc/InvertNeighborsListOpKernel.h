// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "torch/script.h"

template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCPU(
        int64_t num_points,
        const torch::Tensor& inp_neighbors_index,
        const torch::Tensor& inp_neighbors_row_splits,
        const torch::Tensor& inp_neighbors_attributes);

#ifdef BUILD_CUDA_MODULE
template <class TIndex, class TAttr>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsListCUDA(
        int64_t num_points,
        const torch::Tensor& inp_neighbors_index,
        const torch::Tensor& inp_neighbors_row_splits,
        const torch::Tensor& inp_neighbors_attributes);
#endif
