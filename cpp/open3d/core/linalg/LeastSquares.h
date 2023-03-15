// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

/// Solve AX = B with QR decomposition. A is a full-rank m x n matrix (m >= n).
void LeastSquares(const Tensor& A, const Tensor& B, Tensor& X);

#ifdef BUILD_CUDA_MODULE
void LeastSquaresCUDA(void* A_data,
                      void* B_data,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      Dtype dtype,
                      const Device& device);
#endif

void LeastSquaresCPU(void* A_data,
                     void* B_data,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     Dtype dtype,
                     const Device& device);

}  // namespace core
}  // namespace open3d
