// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/ParallelFor.isph"

static inline void Square(int64_t idx,
                          float* uniform input,
                          float* uniform output) {
    float x = input[idx];
    float x2 = x * x;
    output[idx] = x2;
}

OPEN3D_EXPORT_VECTORIZED(SquareKernel, Square, float* uniform, float* uniform)
