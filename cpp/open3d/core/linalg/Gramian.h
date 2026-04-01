// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

/// Computes the gram matrix of a Tensor (B = A.T @ A)
void Gram(const Tensor& A, Tensor& B);

/// Computes the row gram matrix of a Tensor (B = A @ A.T)
void RowGram(const Tensor& A, Tensor& B);

#ifdef BUILD_SYCL_MODULE
void GramSYCL(void* A_data,
              void* B_data,
              int64_t m,
              int64_t n,
              Dtype dtype,
              const Device& device);
void RowGramSYCL(void* A_data,
                 void* B_data,
                 int64_t m,
                 int64_t n,
                 Dtype dtype,
                 const Device& device);
#endif
#ifdef BUILD_CUDA_MODULE
void GramCUDA(void* A_data,
              void* B_data,
              int64_t m,
              int64_t n,
              Dtype dtype,
              const Device& device);
void RowGramCUDA(void* A_data,
                 void* B_data,
                 int64_t m,
                 int64_t n,
                 Dtype dtype,
                 const Device& device);
#endif
void GramCPU(void* A_data, void* B_data, int64_t m, int64_t n, Dtype dtype);
void RowGramCPU(void* A_data, void* B_data, int64_t m, int64_t n, Dtype dtype);
}  // namespace core
}  // namespace open3d
