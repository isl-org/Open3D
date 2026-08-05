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

/// Computes matrix multiplication C = AB.
void Matmul(const Tensor& A, const Tensor& B, Tensor& C);

#ifdef BUILD_SYCL_MODULE
void MatmulSYCL(void* A_data,
                void* B_data,
                void* C_data,
                int64_t m,
                int64_t k,
                int64_t n,
                Dtype dtype,
                const Device& device);
#endif
#ifdef BUILD_CUDA_MODULE
void MatmulCUDA(void* A_data,
                void* B_data,
                void* C_data,
                int64_t m,
                int64_t k,
                int64_t n,
                Dtype dtype,
                const Device& device);
#endif
void MatmulCPU(void* A_data,
               void* B_data,
               void* C_data,
               int64_t m,
               int64_t k,
               int64_t n,
               Dtype dtype);
}  // namespace core
}  // namespace open3d
