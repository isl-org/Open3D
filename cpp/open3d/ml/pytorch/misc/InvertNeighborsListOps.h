// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "torch/script.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InvertNeighborsList(
        int64_t num_points,
        torch::Tensor inp_neighbors_index,
        torch::Tensor inp_neighbors_row_splits,
        torch::Tensor inp_neighbors_attributes);
