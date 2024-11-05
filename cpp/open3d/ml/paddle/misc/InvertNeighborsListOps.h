// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

// this file seem not use
#pragma once

#include "open3d/ml/paddle/PaddleHelper.h"

std::vector<paddle::Tensor> InvertNeighborsList(
        int64_t num_points,
        const paddle::Tensor& inp_neighbors_index,
        const paddle::Tensor& inp_neighbors_row_splits,
        const paddle::Tensor& inp_neighbors_attributes);
