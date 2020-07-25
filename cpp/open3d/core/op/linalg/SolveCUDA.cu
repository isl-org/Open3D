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

// https://
// software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_sgesv_row.c.htm
#include <stdio.h>
#include <stdlib.h>

#include "open3d/core/op/linalg/LinalgUtils.h"
#include "open3d/core/op/linalg/Solve.h"

namespace open3d {
namespace core {

// cusolverDn<t1><t2>gels() is not supported until CUDA 11.0.
// We have to implement for earlier versions via
// Step 1: A = Q*R by geqrf.
// Step 2: B : = Q ^ T* B by ormqr.
// Step 3: solve R* X = B by trsm.
// Ref: https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1
void SolveCUDA(void* A_data,
               void* B_data,
               int m,
               int n,
               int k,
               Dtype dtype,
               const Device& device) {
    cusolverDnHandle_t cusolver_handle =
            CuSolverContext::GetInstance()->GetHandle();
    cublasHandle_t cublas_handle = CuBLASContext::GetInstance()->GetHandle();
    int* dinfo = static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));

    int len_geqrf, len_ormqr, len;
    switch (dtype) {
        case Dtype::Float32: {
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnSgeqrf_bufferSize(cusolver_handle, m, n, NULL, m,
                                                &len_geqrf),
                    "cusolverDnSgeqrf_bufferSize failed");
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnSormqr_bufferSize(
                            cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m,
                            k, m, NULL, m, NULL, NULL, m, &len_ormqr),
                    "cusolverDnSgeqrf_bufferSize failed");
            len = std::max(len_geqrf, len_ormqr);

            void* workspace =
                    MemoryManager::Malloc(len * sizeof(float), device);
            void* tau = MemoryManager::Malloc(n * sizeof(float), device);

            // Step 1: A = QR
            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnSgeqrf(
                            cusolver_handle, m, n, static_cast<float*>(A_data),
                            m, static_cast<float*>(tau),
                            static_cast<float*>(workspace), len, dinfo),
                    "cusolverDnSgeqrf failed with dinfo = ", dinfo, device);

            // Step 2: B' = Q^T*B
            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnSormqr(
                            cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m,
                            k, m, static_cast<float*>(A_data), m,
                            static_cast<float*>(tau),
                            static_cast<float*>(B_data), m,
                            static_cast<float*>(workspace), len, dinfo),
                    "cusolverDnSgeqrf_bufferSize failed with dinfo = ", dinfo,
                    device);

            // Step 3: Solve Rx = B'
            float alpha = 1.0f;
            OPEN3D_CUBLAS_CHECK(cublasStrsm(cublas_handle, CUBLAS_SIDE_LEFT,
                                            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                            CUBLAS_DIAG_NON_UNIT, m, k, &alpha,
                                            static_cast<float*>(A_data), m,
                                            static_cast<float*>(B_data), m),
                                "cublasStrsm failed");

            MemoryManager::Free(workspace, device);
            MemoryManager::Free(tau, device);
            break;
        }

        case Dtype::Float64: {
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnDgeqrf_bufferSize(cusolver_handle, m, n, NULL, m,
                                                &len_geqrf),
                    "cusolverDnDgeqrf_bufferSize failed");
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnDormqr_bufferSize(
                            cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m,
                            k, m, NULL, m, NULL, NULL, m, &len_ormqr),
                    "cusolverDnDgeqrf_bufferSize failed");
            len = std::max(len_geqrf, len_ormqr);

            void* workspace =
                    MemoryManager::Malloc(len * sizeof(double), device);
            void* tau = MemoryManager::Malloc(n * sizeof(double), device);

            // Step 1: A = QR
            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnDgeqrf(
                            cusolver_handle, m, n, static_cast<double*>(A_data),
                            m, static_cast<double*>(tau),
                            static_cast<double*>(workspace), len, dinfo),
                    "cusolverDnDgeqrf failed with dinfo = ", dinfo, device);

            // Step 2: B' = Q^T*B
            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnDormqr(
                            cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m,
                            k, m, static_cast<double*>(A_data), m,
                            static_cast<double*>(tau),
                            static_cast<double*>(B_data), m,
                            static_cast<double*>(workspace), len, dinfo),
                    "cusolverDnDgeqrf_bufferSize failed with dinfo = ", dinfo,
                    device);

            // Step 3 : Solve Rx = B'
            double alpha = 1.0;
            OPEN3D_CUBLAS_CHECK(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT,
                                            CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                            CUBLAS_DIAG_NON_UNIT, m, k, &alpha,
                                            static_cast<double*>(A_data), m,
                                            static_cast<double*>(B_data), m),
                                "cublasDtrsm failed");

            MemoryManager::Free(workspace, device);
            MemoryManager::Free(tau, device);
            break;
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in SolveCUDA.",
                              DtypeUtil::ToString(dtype));
        }
    }

    MemoryManager::Free(dinfo, device);
}

}  // namespace core
}  // namespace open3d
