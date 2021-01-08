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

namespace open3d {
namespace core {

/// \brief This function performs the LU factorization of each Aarray[i] for i =
/// 0, ..., batchSize-1 by the following equation P * Aarray [ i ] = L * U where
/// P is a permutation matrix which represents partial pivoting with row
/// interchanges. L is a lower triangular matrix with unit diagonal and U is an
/// upper triangular matrix.
void LUfactorisation_batch(
        const Tensor& A, Tensor& L, Tensor& U, Tensor& P, Tensor& info);

void SolveCPU(void* A_data,
              void* B_data,
              void* ipiv_data,
              int64_t n,
              int64_t k,
              Dtype dtype,
              const Device& device);

#ifdef BUILD_CUDA_MODULE
void SolveCUDA(void* A_data,
               void* B_data,
               void* ipiv_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device);
#endif

}  // namespace core
}  // namespace open3d
