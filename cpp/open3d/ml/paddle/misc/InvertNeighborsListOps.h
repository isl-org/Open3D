// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

// this file seem not use
#pragma once

#include "open3d/ml/paddle/PaddleHelper.h"

std::vector<paddle::Tensor> InvertNeighborsList(
        paddle::Tensor& inp_neighbors_index,
        paddle::Tensor& inp_neighbors_row_splits,
        paddle::Tensor& inp_neighbors_attributes,
        int64_t num_points);
