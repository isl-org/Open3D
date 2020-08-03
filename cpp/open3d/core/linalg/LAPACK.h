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

#include <mkl.h>

#ifdef BUILD_CUDA_MODULE
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace open3d {
namespace core {
template <typename scalar_t>
void getrf_cpu(int layout,
               MKL_INT m,
               MKL_INT n,
               scalar_t* A_data,
               MKL_INT lda,
               MKL_INT* ipiv_data);

template <typename scalar_t>
void getri_cpu(int layout,
               MKL_INT n,
               scalar_t* A_data,
               MKL_INT lda,
               MKL_INT* ipiv_data);

template <typename scalar_t>
void gels_cpu(int matrix_layout,
              char trans,
              MKL_INT m,
              MKL_INT n,
              MKL_INT nrhs,
              scalar_t* A_data,
              MKL_INT lda,
              scalar_t* B_data,
              MKL_INT ldb);

template <typename scalar_t>
void gesvd_cpu(int matrix_layout,
               char jobu,
               char jobvt,
               MKL_INT m,
               MKL_INT n,
               scalar_t* A_data,
               MKL_INT lda,
               scalar_t* S_data,
               scalar_t* U_data,
               MKL_INT ldu,
               scalar_t* VT_data,
               MKL_INT ldvt,
               scalar_t* superb);

#ifdef BUILD_CUDA_MODULE
template <typename scalar_t>
cusolverStatus_t getrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len);

template <typename scalar_t>
cusolverStatus_t getrf_cuda(cusolverDnHandle_t handle,
                            int m,
                            int n,
                            scalar_t* A_data,
                            int lda,
                            scalar_t* workspace,
                            int* ipiv_data,
                            int* dinfo);

template <typename scalar_t>
cusolverStatus_t getrs_cuda(cusolverDnHandle_t handle,
                            cublasOperation_t trans,
                            int n,
                            int nrhs,
                            const scalar_t* A_data,
                            int lda,
                            const int* ipiv_data,
                            scalar_t* B_data,
                            int ldb,
                            int* dinfo);

template <typename scalar_t>
cusolverStatus_t gesvd_cuda_buffersize(cusolverDnHandle_t handle,
                                       int m,
                                       int n,
                                       int* len);

template <typename scalar_t>
cusolverStatus_t gesvd_cuda(cusolverDnHandle_t handle,
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
                            int* dinfo);

template <typename scalar_t>
cusolverStatus_t geqrf_cuda_buffersize(
        cusolverDnHandle_t handle, int m, int n, int lda, int* len);

template <typename scalar_t>
cusolverStatus_t geqrf_cuda(cusolverDnHandle_t handle,
                            int m,
                            int n,
                            scalar_t* A,
                            int lda,
                            scalar_t* tau,
                            scalar_t* workspace,
                            int len,
                            int* dinfo);

template <typename scalar_t>
cusolverStatus_t ormqr_cuda_buffersize(cusolverDnHandle_t handle,
                                       cublasSideMode_t side,
                                       cublasOperation_t trans,
                                       int m,
                                       int n,
                                       int k,
                                       int lda,
                                       int ldc,
                                       int* len);

template <typename scalar_t>
cusolverStatus_t ormqr_cuda(cusolverDnHandle_t handle,
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
                            int* dinfo);
#endif
}  // namespace core
}  // namespace open3d
