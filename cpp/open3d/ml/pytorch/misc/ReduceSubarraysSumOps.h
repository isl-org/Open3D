// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "torch/script.h"

torch::Tensor ReduceSubarraysSum(torch::Tensor values,
                                 torch::Tensor row_splits);
