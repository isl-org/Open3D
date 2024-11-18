// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
#pragma once

#include "paddle/extension.h"

std::vector<paddle::Tensor> ReduceSubarraysSum(paddle::Tensor& values,
                                               paddle::Tensor& row_splits);
