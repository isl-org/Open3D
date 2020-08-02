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

#include "open3d/core/linalg/LAPACK.h"

namespace open3d {
namespace core {

template <>
void getrf_cpu<float>(
        int layout, int m, int n, float* A_data, int lda, int* ipiv_data) {
    LAPACKE_sgetrf(layout, m, n, A_data, lda, ipiv_data);
}
template <>
void getrf_cpu<double>(
        int layout, int m, int n, double* A_data, int lda, int* ipiv_data) {
    LAPACKE_dgetrf(layout, m, n, A_data, lda, ipiv_data);
}

template <>
void getri_cpu<float>(
        int layout, int n, float* A_data, int lda, int* ipiv_data) {
    LAPACKE_sgetri(layout, n, A_data, lda, ipiv_data);
}
template <>
void getri_cpu<double>(
        int layout, int n, double* A_data, int lda, int* ipiv_data) {
    LAPACKE_dgetri(layout, n, A_data, lda, ipiv_data);
}

template <>
void gesvd_cpu<float>(int matrix_layout,
                      char jobu,
                      char jobvt,
                      int m,
                      int n,
                      float* A_data,
                      int lda,
                      float* S_data,
                      float* U_data,
                      int ldu,
                      float* VT_data,
                      int ldvt,
                      float* superb) {
    LAPACKE_sgesvd(matrix_layout, jobu, jobvt, m, n, A_data, lda, S_data,
                   U_data, ldu, VT_data, ldvt, superb);
}

template <>
void gesvd_cpu<double>(int matrix_layout,
                       char jobu,
                       char jobvt,
                       int m,
                       int n,
                       double* A_data,
                       int lda,
                       double* S_data,
                       double* U_data,
                       int ldu,
                       double* VT_data,
                       int ldvt,
                       double* superb) {
    LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, A_data, lda, S_data,
                   U_data, ldu, VT_data, ldvt, superb);
}

template <>
void gels_cpu<float>(int matrix_layout,
                     char trans,
                     int m,
                     int n,
                     int nrhs,
                     float* A_data,
                     int lda,
                     float* B_data,
                     int ldb) {
    LAPACKE_sgels(matrix_layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

template <>
void gels_cpu<double>(int matrix_layout,
                      char trans,
                      int m,
                      int n,
                      int nrhs,
                      double* A_data,
                      int lda,
                      double* B_data,
                      int ldb) {
    LAPACKE_dgels(matrix_layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

#ifdef BUILD_CUDA_MODULE
template <>
cusolverStatus_t getrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
cusolverStatus_t getrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
cusolverStatus_t getrf_cuda<float>(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   float* A_data,
                                   int lda,
                                   float* workspace,
                                   int* ipiv_data,
                                   int* dinfo) {
    return cusolverDnSgetrf(handle, m, n, A_data, lda, workspace, ipiv_data,
                            dinfo);
}

template <>
cusolverStatus_t getrf_cuda<double>(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double* A_data,
                                    int lda,
                                    double* workspace,
                                    int* ipiv_data,
                                    int* dinfo) {
    return cusolverDnDgetrf(handle, m, n, A_data, lda, workspace, ipiv_data,
                            dinfo);
}

template <>
cusolverStatus_t getrs_cuda<float>(cusolverDnHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const float* A_data,
                                   int lda,
                                   const int* ipiv_data,
                                   float* B_data,
                                   int ldb,
                                   int* dinfo) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, A_data, lda, ipiv_data,
                            B_data, ldb, dinfo);
}

template <>
cusolverStatus_t getrs_cuda<double>(cusolverDnHandle_t handle,
                                    cublasOperation_t trans,
                                    int n,
                                    int nrhs,
                                    const double* A_data,
                                    int lda,
                                    const int* ipiv_data,
                                    double* B_data,
                                    int ldb,
                                    int* dinfo) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, A_data, lda, ipiv_data,
                            B_data, ldb, dinfo);
}

template <>
cusolverStatus_t gesvd_cuda_buffersize<float>(cusolverDnHandle_t handle,
                                              int m,
                                              int n,
                                              int* len) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, len);
}

template <>
cusolverStatus_t gesvd_cuda_buffersize<double>(cusolverDnHandle_t handle,
                                               int m,
                                               int n,
                                               int* len) {
    return cusolverDnDgesvd_bufferSize(handle, m, n, len);
}

template <>
cusolverStatus_t gesvd_cuda<float>(cusolverDnHandle_t handle,
                                   char jobu,
                                   char jobvt,
                                   int m,
                                   int n,
                                   float* A,
                                   int lda,
                                   float* S,
                                   float* U,
                                   int ldu,
                                   float* VT,
                                   int ldvt,
                                   float* workspace,
                                   int len,
                                   float* rwork,
                                   int* dinfo) {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                            ldvt, workspace, len, rwork, dinfo);
}

template <>
cusolverStatus_t gesvd_cuda<double>(cusolverDnHandle_t handle,
                                    char jobu,
                                    char jobvt,
                                    int m,
                                    int n,
                                    double* A,
                                    int lda,
                                    double* S,
                                    double* U,
                                    int ldu,
                                    double* VT,
                                    int ldvt,
                                    double* workspace,
                                    int len,
                                    double* rwork,
                                    int* dinfo) {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT,
                            ldvt, workspace, len, rwork, dinfo);
}

template <>
cusolverStatus_t geqrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
cusolverStatus_t geqrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
cusolverStatus_t geqrf_cuda<float>(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   float* A,
                                   int lda,
                                   float* tau,
                                   float* workspace,
                                   int len,
                                   int* dinfo) {
    return cusolverDnSgeqrf(handle, m, n, A, lda, tau, workspace, len, dinfo);
}

template <>
cusolverStatus_t geqrf_cuda<double>(cusolverDnHandle_t handle,
                                    int m,
                                    int n,
                                    double* A,
                                    int lda,
                                    double* tau,
                                    double* workspace,
                                    int len,
                                    int* dinfo) {
    return cusolverDnDgeqrf(handle, m, n, A, lda, tau, workspace, len, dinfo);
}

template <>
cusolverStatus_t ormqr_cuda_buffersize<float>(cusolverDnHandle_t handle,
                                              cublasSideMode_t side,
                                              cublasOperation_t trans,
                                              int m,
                                              int n,
                                              int k,
                                              int lda,
                                              int ldc,
                                              int* len) {
    return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, NULL, lda,
                                       NULL, NULL, ldc, len);
}

template <>
cusolverStatus_t ormqr_cuda_buffersize<double>(cusolverDnHandle_t handle,
                                               cublasSideMode_t side,
                                               cublasOperation_t trans,
                                               int m,
                                               int n,
                                               int k,
                                               int lda,
                                               int ldc,
                                               int* len) {
    return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, NULL, lda,
                                       NULL, NULL, ldc, len);
}

template <>
cusolverStatus_t ormqr_cuda<float>(cusolverDnHandle_t handle,
                                   cublasSideMode_t side,
                                   cublasOperation_t trans,
                                   int m,
                                   int n,
                                   int k,
                                   const float* A,
                                   int lda,
                                   const float* tau,
                                   float* C,
                                   int ldc,
                                   float* workspace,
                                   int len,
                                   int* dinfo) {
    return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                            workspace, len, dinfo);
}

template <>
cusolverStatus_t ormqr_cuda<double>(cusolverDnHandle_t handle,
                                    cublasSideMode_t side,
                                    cublasOperation_t trans,
                                    int m,
                                    int n,
                                    int k,
                                    const double* A,
                                    int lda,
                                    const double* tau,
                                    double* C,
                                    int ldc,
                                    double* workspace,
                                    int len,
                                    int* dinfo) {
    return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                            workspace, len, dinfo);
}
template <>
cublasStatus_t trsm_cuda<float>(cublasHandle_t handle,
                                cublasSideMode_t side,
                                cublasFillMode_t uplo,
                                cublasOperation_t trans,
                                cublasDiagType_t diag,
                                int m,
                                int n,
                                const float* alpha,
                                const float* A,
                                int lda,
                                float* B,
                                int ldb) {
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

template <>
cublasStatus_t trsm_cuda<double>(cublasHandle_t handle,
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 cublasDiagType_t diag,
                                 int m,
                                 int n,
                                 const double* alpha,
                                 const double* A,
                                 int lda,
                                 double* B,
                                 int ldb) {
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
#endif
}  // namespace core
}  // namespace open3d
