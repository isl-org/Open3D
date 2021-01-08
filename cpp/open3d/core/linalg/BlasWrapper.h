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

#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/LinalgHeadersCUDA.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {

template <typename scalar_t>
inline void gemm_cpu(CBLAS_LAYOUT layout,
                     CBLAS_TRANSPOSE trans_A,
                     CBLAS_TRANSPOSE trans_B,
                     OPEN3D_CPU_LINALG_INT m,
                     OPEN3D_CPU_LINALG_INT n,
                     OPEN3D_CPU_LINALG_INT k,
                     scalar_t alpha,
                     const scalar_t *A_data,
                     OPEN3D_CPU_LINALG_INT lda,
                     const scalar_t *B_data,
                     OPEN3D_CPU_LINALG_INT ldb,
                     scalar_t beta,
                     scalar_t *C_data,
                     OPEN3D_CPU_LINALG_INT ldc) {
    utility::LogError("Unsupported data type.");
}

template <>
inline void gemm_cpu<float>(CBLAS_LAYOUT layout,
                            CBLAS_TRANSPOSE trans_A,
                            CBLAS_TRANSPOSE trans_B,
                            OPEN3D_CPU_LINALG_INT m,
                            OPEN3D_CPU_LINALG_INT n,
                            OPEN3D_CPU_LINALG_INT k,
                            float alpha,
                            const float *A_data,
                            OPEN3D_CPU_LINALG_INT lda,
                            const float *B_data,
                            OPEN3D_CPU_LINALG_INT ldb,
                            float beta,
                            float *C_data,
                            OPEN3D_CPU_LINALG_INT ldc) {
    cblas_sgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

template <>
inline void gemm_cpu<double>(CBLAS_LAYOUT layout,
                             CBLAS_TRANSPOSE trans_A,
                             CBLAS_TRANSPOSE trans_B,
                             OPEN3D_CPU_LINALG_INT m,
                             OPEN3D_CPU_LINALG_INT n,
                             OPEN3D_CPU_LINALG_INT k,
                             double alpha,
                             const double *A_data,
                             OPEN3D_CPU_LINALG_INT lda,
                             const double *B_data,
                             OPEN3D_CPU_LINALG_INT ldb,
                             double beta,
                             double *C_data,
                             OPEN3D_CPU_LINALG_INT ldc) {
    cblas_dgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

#ifdef BUILD_CUDA_MODULE
template <typename scalar_t>
inline cublasStatus_t gemm_cuda(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const scalar_t *alpha,
                                const scalar_t *A_data,
                                int lda,
                                const scalar_t *B_data,
                                int ldb,
                                const scalar_t *beta,
                                scalar_t *C_data,
                                int ldc) {
    utility::LogError("Unsupported data type.");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <typename scalar_t>
inline cublasStatus_t trsm_cuda(cublasHandle_t handle,
                                cublasSideMode_t side,
                                cublasFillMode_t uplo,
                                cublasOperation_t trans,
                                cublasDiagType_t diag,
                                int m,
                                int n,
                                const scalar_t *alpha,
                                const scalar_t *A,
                                int lda,
                                scalar_t *B,
                                int ldb) {
    utility::LogError("Unsupported data type.");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <typename scalar_t>
inline cublasStatus_t getrfBatched_cuda(cublasHandle_t handle,
                                        int n,
                                        scalar_t *A_data[],
                                        int lda,
                                        int *PivotArray,
                                        int *infoArray,
                                        int batchSize) {
    utility::LogError("Unsupported data type.");
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
inline cublasStatus_t gemm_cuda<float>(cublasHandle_t handle,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int m,
                                       int n,
                                       int k,
                                       const float *alpha,
                                       const float *A_data,
                                       int lda,
                                       const float *B_data,
                                       int ldb,
                                       const float *beta,
                                       float *C_data,
                                       int ldc) {
    return cublasSgemm(handle, transa,
                       transb,   // A, B transpose flag
                       m, n, k,  // dimensions
                       alpha, static_cast<const float *>(A_data), lda,
                       static_cast<const float *>(B_data),
                       ldb,  // input and their leading dims
                       beta, static_cast<float *>(C_data), ldc);
}

template <>
inline cublasStatus_t gemm_cuda<double>(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const double *alpha,
                                        const double *A_data,
                                        int lda,
                                        const double *B_data,
                                        int ldb,
                                        const double *beta,
                                        double *C_data,
                                        int ldc) {
    return cublasDgemm(handle, transa,
                       transb,   // A, B transpose flag
                       m, n, k,  // dimensions
                       alpha, static_cast<const double *>(A_data), lda,
                       static_cast<const double *>(B_data),
                       ldb,  // input and their leading dims
                       beta, static_cast<double *>(C_data), ldc);
}

template <>
inline cublasStatus_t trsm_cuda<float>(cublasHandle_t handle,
                                       cublasSideMode_t side,
                                       cublasFillMode_t uplo,
                                       cublasOperation_t trans,
                                       cublasDiagType_t diag,
                                       int m,
                                       int n,
                                       const float *alpha,
                                       const float *A,
                                       int lda,
                                       float *B,
                                       int ldb) {
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

template <>
inline cublasStatus_t trsm_cuda<double>(cublasHandle_t handle,
                                        cublasSideMode_t side,
                                        cublasFillMode_t uplo,
                                        cublasOperation_t trans,
                                        cublasDiagType_t diag,
                                        int m,
                                        int n,
                                        const double *alpha,
                                        const double *A,
                                        int lda,
                                        double *B,
                                        int ldb) {
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

/// \brief cuBlas: This function performs the LU factorization of each Aarray[i]
/// for i = 0, ..., batchSize-1 by the following equation
/// P * Aarray [ i ] = L * U
/// where P is a permutation matrix which represents partial pivoting with
/// row interchanges. L is a lower triangular matrix with unit diagonal
/// and U is an upper triangular matrix.
///
/// \param handle handle to the cuBLAS library context [input].
/// \param n number of rows and columns of Aarray[i]. [input].
/// \param A_data array of pointers to <type> array, with each array of
///     dim. n x n with lda>=max(1,n). Matrices Aarray[i] should not overlap;
///     otherwise, undefined behavior is expected. [input / output].
/// \param lda leading dimension of two-dimensional array used to store each
///     matrix Aarray[i]. [input].
/// \param PivotArray array of size n x batchSize that contains the pivoting
///     sequence of each factorization of Aarray[i] stored in a linear fashion.
///     If PivotArray is nil, pivoting is disabled. [output].
/// \param infoArray array of size batchSize that info(=infoArray[i])
///           contains the information of factorization of Aarray[i]. [output].
/// \param batchSize number of pointers contained in A [input].
template <>
inline cublasStatus_t getrfBatched_cuda<float>(cublasHandle_t handle,
                                               int n,
                                               float *A_data[],
                                               int lda,
                                               int *PivotArray,
                                               int *infoArray,
                                               int batchSize) {
    return cublasSgetrfBatched(handle, n, A_data, lda, PivotArray, infoArray,
                               batchSize);
}

template <>
inline cublasStatus_t getrfBatched_cuda<double>(cublasHandle_t handle,
                                                int n,
                                                double *A_data[],
                                                int lda,
                                                int *PivotArray,
                                                int *infoArray,
                                                int batchSize) {
    return cublasDgetrfBatched(handle, n, A_data, lda, PivotArray, infoArray,
                               batchSize);
}
#endif

}  // namespace core
}  // namespace open3d
