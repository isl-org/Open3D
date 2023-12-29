// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.

#pragma once

#include "open3d/core/linalg/LU.h"

namespace open3d {
namespace core {

void LUCPU(void* A_data,
           void* ipiv_data,
           int64_t rows,
           int64_t cols,
           Dtype dtype,
           const Device& device);

#ifdef BUILD_CUDA_MODULE
void LUCUDA(void* A_data,
            void* ipiv_data,
            int64_t rows,
            int64_t cols,
            Dtype dtype,
            const Device& device);
#endif
}  // namespace core
}  // namespace open3d
