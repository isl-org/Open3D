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

#include "open3d/core/op/linalg/BLAS.h"

namespace open3d {
namespace core {

template <>
void gemm_cpu<float>(CBLAS_LAYOUT layout,
                     CBLAS_TRANSPOSE trans_A,
                     CBLAS_TRANSPOSE trans_B,
                     int m,
                     int n,
                     int k,
                     float alpha,
                     const float* A_data,
                     int lda,
                     const float* B_data,
                     int ldb,
                     float beta,
                     float* C_data,
                     int ldc) {
    cblas_sgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

template <>
void gemm_cpu<double>(CBLAS_LAYOUT layout,
                      CBLAS_TRANSPOSE trans_A,
                      CBLAS_TRANSPOSE trans_B,
                      int m,
                      int n,
                      int k,
                      double alpha,
                      const double* A_data,
                      int lda,
                      const double* B_data,
                      int ldb,
                      double beta,
                      double* C_data,
                      int ldc) {
    cblas_dgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

template <>
cublasStatus_t gemm_cuda<float>(cublasHandle_t handle,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* A_data,
                                int lda,
                                const float* B_data,
                                int ldb,
                                const float* beta,
                                float* C_data,
                                int ldc) {
    return cublasSgemm(handle, transa,
                       transb,   // A, B transpose flag
                       m, n, k,  // dimensions
                       alpha, static_cast<const float*>(A_data), lda,
                       static_cast<const float*>(B_data),
                       ldb,  // input and their leading dims
                       beta, static_cast<float*>(C_data), ldc);
}

template <>
cublasStatus_t gemm_cuda<double>(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const double* alpha,
                                 const double* A_data,
                                 int lda,
                                 const double* B_data,
                                 int ldb,
                                 const double* beta,
                                 double* C_data,
                                 int ldc) {
    return cublasDgemm(handle, transa,
                       transb,   // A, B transpose flag
                       m, n, k,  // dimensions
                       alpha, static_cast<const double*>(A_data), lda,
                       static_cast<const double*>(B_data),
                       ldb,  // input and their leading dims
                       beta, static_cast<double*>(C_data), ldc);
}

}  // namespace core
}  // namespace open3d
