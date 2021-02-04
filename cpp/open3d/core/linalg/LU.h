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

/// \brief This function performs the LU factorization of 2D square matrix,
/// following equation P * A = L * U where P is the permutation matrix,
/// L is a lower triangular matrix with unit diagonal and U is an upper
/// triangular matrix. [computes ipiv and output tensors].
///
/// \param A [input] 2D square matrix to be factorised. [Tensor of dtype
/// Float32/64]
/// \param ipiv [output] is a 1D int tensor that contains the pivot indices,
/// indicating row i of the matrix was interchanged with row IPIV(i)).
/// \param output [output] is a 2D tensor of same dimentions as input, and has
/// L as lower triangular values and U as upper triangle values including
/// the main diagonal (diagonal elemetes of L to be taken as unity).
void LU_with_ipiv(const Tensor& A, Tensor& ipiv, Tensor& output);

/// \brief This function performs the LU factorization of 2D square matrix,
/// following equation A = P * L * U where P is the permutation matrix,
/// L is a lower triangular matrix with unit diagonal and U is an upper
/// triangular matrix, [computes P, L, U tensors].
///
/// \param A [input] 2D square matrix to be factorised. [Tensor of dtype
/// Float32/64] \param permutation [output] 2D permutation matrix.
/// \param lower [output] Lower triangular matrix.
/// \param upper [output] Upper triangular matrix.
/// \param permute_l [optional bool input (default: false)]
///  If true then returns L as P * L.
void LU(const Tensor& A,
        Tensor& permutation,
        Tensor& lower,
        Tensor& upper,
        bool permute_l = false);

}  // namespace core
}  // namespace open3d
