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
inline OPEN3D_CPU_LINALG_INT getrf_cpu(int layout,
                                       OPEN3D_CPU_LINALG_INT m,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       OPEN3D_CPU_LINALG_INT* ipiv_data) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT getri_cpu(int layout,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       OPEN3D_CPU_LINALG_INT* ipiv_data) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gesv_cpu(int layout,
                                      OPEN3D_CPU_LINALG_INT n,
                                      OPEN3D_CPU_LINALG_INT m,
                                      scalar_t* A_data,
                                      OPEN3D_CPU_LINALG_INT lda,
                                      OPEN3D_CPU_LINALG_INT* ipiv_data,
                                      scalar_t* B_data,
                                      OPEN3D_CPU_LINALG_INT ldb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gels_cpu(int matrix_layout,
                                      char trans,
                                      OPEN3D_CPU_LINALG_INT m,
                                      OPEN3D_CPU_LINALG_INT n,
                                      OPEN3D_CPU_LINALG_INT nrhs,
                                      scalar_t* A_data,
                                      OPEN3D_CPU_LINALG_INT lda,
                                      scalar_t* B_data,
                                      OPEN3D_CPU_LINALG_INT ldb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <typename scalar_t>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu(int matrix_layout,
                                       char jobu,
                                       char jobvt,
                                       OPEN3D_CPU_LINALG_INT m,
                                       OPEN3D_CPU_LINALG_INT n,
                                       scalar_t* A_data,
                                       OPEN3D_CPU_LINALG_INT lda,
                                       scalar_t* S_data,
                                       scalar_t* U_data,
                                       OPEN3D_CPU_LINALG_INT ldu,
                                       scalar_t* VT_data,
                                       OPEN3D_CPU_LINALG_INT ldvt,
                                       scalar_t* superb) {
    utility::LogError("Unsupported data type.");
    return -1;
}

template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_sgetrf(layout, m, n, A_data, lda, ipiv_data);
}
template <>
inline OPEN3D_CPU_LINALG_INT getrf_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT m,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_dgetrf(layout, m, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<float>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        float* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_sgetri(layout, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT getri_cpu<double>(
        int layout,
        OPEN3D_CPU_LINALG_INT n,
        double* A_data,
        OPEN3D_CPU_LINALG_INT lda,
        OPEN3D_CPU_LINALG_INT* ipiv_data) {
    return LAPACKE_dgetri(layout, n, A_data, lda, ipiv_data);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<float>(int layout,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT m,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             OPEN3D_CPU_LINALG_INT* ipiv_data,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_sgesv(layout, n, m, A_data, lda, ipiv_data, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesv_cpu<double>(int layout,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT m,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              OPEN3D_CPU_LINALG_INT* ipiv_data,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_dgesv(layout, n, m, A_data, lda, ipiv_data, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<float>(int layout,
                                             char trans,
                                             OPEN3D_CPU_LINALG_INT m,
                                             OPEN3D_CPU_LINALG_INT n,
                                             OPEN3D_CPU_LINALG_INT nrhs,
                                             float* A_data,
                                             OPEN3D_CPU_LINALG_INT lda,
                                             float* B_data,
                                             OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_sgels(layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gels_cpu<double>(int layout,
                                              char trans,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              OPEN3D_CPU_LINALG_INT nrhs,
                                              double* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              double* B_data,
                                              OPEN3D_CPU_LINALG_INT ldb) {
    return LAPACKE_dgels(layout, trans, m, n, nrhs, A_data, lda, B_data, ldb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<float>(int layout,
                                              char jobu,
                                              char jobvt,
                                              OPEN3D_CPU_LINALG_INT m,
                                              OPEN3D_CPU_LINALG_INT n,
                                              float* A_data,
                                              OPEN3D_CPU_LINALG_INT lda,
                                              float* S_data,
                                              float* U_data,
                                              OPEN3D_CPU_LINALG_INT ldu,
                                              float* VT_data,
                                              OPEN3D_CPU_LINALG_INT ldvt,
                                              float* superb) {
    return LAPACKE_sgesvd(layout, jobu, jobvt, m, n, A_data, lda, S_data,
                          U_data, ldu, VT_data, ldvt, superb);
}

template <>
inline OPEN3D_CPU_LINALG_INT gesvd_cpu<double>(int layout,
                                               char jobu,
                                               char jobvt,
                                               OPEN3D_CPU_LINALG_INT m,
                                               OPEN3D_CPU_LINALG_INT n,
                                               double* A_data,
                                               OPEN3D_CPU_LINALG_INT lda,
                                               double* S_data,
                                               double* U_data,
                                               OPEN3D_CPU_LINALG_INT ldu,
                                               double* VT_data,
                                               OPEN3D_CPU_LINALG_INT ldvt,
                                               double* superb) {
    return LAPACKE_dgesvd(layout, jobu, jobvt, m, n, A_data, lda, S_data,
                          U_data, ldu, VT_data, ldvt, superb);
}

#ifdef BUILD_CUDA_MODULE
template <typename scalar_t>
inline cusolverStatus_t getrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t getrf_cuda(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   scalar_t* A_data,
                                   int lda,
                                   scalar_t* workspace,
                                   int* ipiv_data,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t getrs_cuda(cusolverDnHandle_t handle,
                                   cublasOperation_t trans,
                                   int n,
                                   int nrhs,
                                   const scalar_t* A_data,
                                   int lda,
                                   const int* ipiv_data,
                                   scalar_t* B_data,
                                   int ldb,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t geqrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t geqrf_cuda(cusolverDnHandle_t handle,
                                   int m,
                                   int n,
                                   scalar_t* A,
                                   int lda,
                                   scalar_t* tau,
                                   scalar_t* workspace,
                                   int len,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t ormqr_cuda_buffersize(cusolverDnHandle_t handle,
                                              cublasSideMode_t side,
                                              cublasOperation_t trans,
                                              int m,
                                              int n,
                                              int k,
                                              int lda,
                                              int ldc,
                                              int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t ormqr_cuda(cusolverDnHandle_t handle,
                                   cublasSideMode_t side,
                                   cublasOperation_t trans,
                                   int m,
                                   int n,
                                   int k,
                                   const scalar_t* A,
                                   int lda,
                                   const scalar_t* tau,
                                   scalar_t* C,
                                   int ldc,
                                   scalar_t* workspace,
                                   int len,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t gesvd_cuda_buffersize(cusolverDnHandle_t handle,
                                              int m,
                                              int n,
                                              int* len) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <typename scalar_t>
inline cusolverStatus_t gesvd_cuda(cusolverDnHandle_t handle,
                                   char jobu,
                                   char jobvt,
                                   int m,
                                   int n,
                                   scalar_t* A,
                                   int lda,
                                   scalar_t* S,
                                   scalar_t* U,
                                   int ldu,
                                   scalar_t* VT,
                                   int ldvt,
                                   scalar_t* workspace,
                                   int len,
                                   scalar_t* rwork,
                                   int* dinfo) {
    utility::LogError("Unsupported data type.");
    return CUSOLVER_STATUS_INTERNAL_ERROR;
}

template <>
inline cusolverStatus_t getrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t getrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t getrf_cuda<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t getrf_cuda<double>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t getrs_cuda<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t getrs_cuda<double>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t geqrf_cuda_buffersize<float>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t geqrf_cuda_buffersize<double>(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len) {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, NULL, lda, len);
}

template <>
inline cusolverStatus_t geqrf_cuda<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t geqrf_cuda<double>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t ormqr_cuda_buffersize<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t ormqr_cuda_buffersize<double>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t ormqr_cuda<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t ormqr_cuda<double>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t gesvd_cuda_buffersize<float>(cusolverDnHandle_t handle,
                                                     int m,
                                                     int n,
                                                     int* len) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, len);
}

template <>
inline cusolverStatus_t gesvd_cuda_buffersize<double>(cusolverDnHandle_t handle,
                                                      int m,
                                                      int n,
                                                      int* len) {
    return cusolverDnDgesvd_bufferSize(handle, m, n, len);
}

template <>
inline cusolverStatus_t gesvd_cuda<float>(cusolverDnHandle_t handle,
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
inline cusolverStatus_t gesvd_cuda<double>(cusolverDnHandle_t handle,
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

#endif
}  // namespace core
}  // namespace open3d
