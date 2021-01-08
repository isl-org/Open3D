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
/// \param A [input] Batch of 2D square matrices to be factorised. [Tensor of
/// dtype Float32 or Float64, dim {m,n,n} where {n,n} is the dim. of square
/// matrices and m is the batch size]. \param L [output] Lower triangular matrix
/// \param U [output] Upper triangular matrix
/// \param P [output] Permutation matrix
/// \param info [output] array of size batchSize that info(=infoArray[i])
///           contains the information of factorization of Aarray[i].
void LUfactorisation(
        const Tensor& A, Tensor& L, Tensor& U, Tensor& P, Tensor& info);

void LUfactorisationCPU(
        void* A_data, int64_t n, int64_t k, Dtype dtype, const Device& device);

#ifdef BUILD_CUDA_MODULE
void LUfactorisationCUDA(
        void* A_data, int64_t n, int64_t k, Dtype dtype, const Device& device);
#endif

/*  TO BE REMOVED [JUST FOR DEVELOPMENT REFERENCE]
    cuBLAS: [SUPPORTS BATCH and RECT. MAT.]
        inline cublasStatus_t getrfBatched_cuda<float>(cublasHandle_t handle,
                                        int n,
                                        float* A_data,
                                        int lda,
                                        int *PivotArray,
                                        int *infoArray,
                                        int batchSize); {
            return cublasSgetrfBatched(handle, n, A_data, lda, PivotArray,
   infoArray, batchSize);
        }

    LAPACK: [TO SUPPRT BATCH: REFER COMPACT ROUTINES]
        inline OPEN3D_CPU_LINALG_INT getrfnp_cpu<float>(int layout,
                                                    OPEN3D_CPU_LINALG_INT m,
                                                    OPEN3D_CPU_LINALG_INT n,
                                                    float* A_data,
                                                    OPEN3D_CPU_LINALG_INT lda) {
            return LAPACKE_mkl_sgetrfnp(layout, m, n, A_data, lda);
        }
*/

}  // namespace core
}  // namespace open3d
