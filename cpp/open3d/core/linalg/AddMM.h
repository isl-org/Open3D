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

/// Computes matrix multiplication C = alpha * A @ B + beta * C.
/// If matrix A is a (n x m) tensor, and B is a (m x p) tensor, C should have a
/// shape (n x p).
/// alpha and beta are scaling factors on matrix-matrix multiplication and the
/// added matrix input respectively.
void AddMM(
        const Tensor& A, const Tensor& B, Tensor& C, double alpha, double beta);

#ifdef BUILD_CUDA_MODULE
void AddMMCUDA(void* A_data,
               void* B_data,
               void* C_data,
               int64_t m,
               int64_t k,
               int64_t n,
               double alpha,
               double beta,
               bool gemmTrA,
               bool gemmTrB,
               int lda,
               int ldb,
               int ldc,
               Dtype dtype,
               const Device& device);
#endif

void AddMMCPU(void* A_data,
              void* B_data,
              void* C_data,
              int64_t m,
              int64_t k,
              int64_t n,
              double alpha,
              double beta,
              bool gemmTrA,
              bool gemmTrB,
              int lda,
              int ldb,
              int ldc,
              Dtype dtype);

}  // namespace core
}  // namespace open3d
