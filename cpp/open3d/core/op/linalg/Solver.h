// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

// MKL
// https://github.com/pytorch/pytorch/blob/afb2d27b24b515f380e889028fe53998d29d4e38/aten/src/ATen/native/BatchLinearAlgebra.cpp
// MAGMA
// https://github.com/pytorch/pytorch/blob/e98ad6c05b21f9ee04447de69e809af42f4e114a/aten/src/ATen/native/cuda/BatchLinearAlgebra.cu

namespace open3d {
namespace core {

// Solve AX = B with *gesv in MKL (CPU) and MAGMA (CUDA)
Tensor Solve(const Tensor& A, const Tensor& B);

namespace detail {
#ifdef BUILD_CUDA_MODULE
void SolverCUDABackend(
        Dtype dtype, void* A_data, void* B_data, void* ipiv_data, int n, int m);
#endif

void SolverCPUBackend(
        Dtype dtype, void* A_data, void* B_data, void* ipiv_data, int n, int m);
}  // namespace detail

}  // namespace core
}  // namespace open3d
