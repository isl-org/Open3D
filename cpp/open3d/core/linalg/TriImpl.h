// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/linalg/Tri.h"

namespace open3d {
namespace core {

void TriuCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCPU(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCPU(const Tensor& A,
              Tensor& upper,
              Tensor& lower,
              const int diagonal = 0);

#ifdef BUILD_CUDA_MODULE
void TriuCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TrilCUDA(const Tensor& A, Tensor& output, const int diagonal = 0);

void TriulCUDA(const Tensor& A,
               Tensor& upper,
               Tensor& lower,
               const int diagonal = 0);
#endif
}  // namespace core
}  // namespace open3d
