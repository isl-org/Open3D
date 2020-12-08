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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LeastSquares.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

// cusolverDn<t1><t2>gels() is not supported until CUDA 11.0.
// We have to implement for earlier versions via
// Step 1: A = Q*R by geqrf.
// Step 2: B : = Q ^ T* B by ormqr.
// Step 3: solve R* X = B by trsm.
// Ref: https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1
void LeastSquaresCUDA(void* A_data,
                      void* B_data,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      Dtype dtype,
                      const Device& device) {
    cusolverDnHandle_t cusolver_handle =
            CuSolverContext::GetInstance()->GetHandle();
    cublasHandle_t cublas_handle = CuBLASContext::GetInstance()->GetHandle();

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len_geqrf, len_ormqr, len;
        int* dinfo =
                static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));

        OPEN3D_CUSOLVER_CHECK(geqrf_cuda_buffersize<scalar_t>(
                                      cusolver_handle, m, n, m, &len_geqrf),
                              "geqrf_buffersize failed in LeastSquaresCUDA");
        OPEN3D_CUSOLVER_CHECK(ormqr_cuda_buffersize<scalar_t>(
                                      cusolver_handle, CUBLAS_SIDE_LEFT,
                                      CUBLAS_OP_T, m, k, n, m, m, &len_ormqr),
                              "ormqr_buffersize failed in LeastSquaresCUDA");
        len = std::max(len_geqrf, len_ormqr);

        void* workspace = MemoryManager::Malloc(len * sizeof(scalar_t), device);
        void* tau = MemoryManager::Malloc(n * sizeof(scalar_t), device);

        // Step 1: A = QR
        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                geqrf_cuda<scalar_t>(
                        cusolver_handle, m, n, static_cast<scalar_t*>(A_data),
                        m, static_cast<scalar_t*>(tau),
                        static_cast<scalar_t*>(workspace), len, dinfo),
                "geqrf failed in LeastSquaresCUDA", dinfo, device);

        // Step 2: B' = Q^T*B
        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                ormqr_cuda<scalar_t>(
                        cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, k, n,
                        static_cast<scalar_t*>(A_data), m,
                        static_cast<scalar_t*>(tau),
                        static_cast<scalar_t*>(B_data), m,
                        static_cast<scalar_t*>(workspace), len, dinfo),
                "ormqr failed in LeastSquaresCUDA", dinfo, device);

        // Step 3: Solve Rx = B'
        scalar_t alpha = 1.0f;
        OPEN3D_CUBLAS_CHECK(
                trsm_cuda<scalar_t>(cublas_handle, CUBLAS_SIDE_LEFT,
                                    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, n, k, &alpha,
                                    static_cast<scalar_t*>(A_data), m,
                                    static_cast<scalar_t*>(B_data), m),
                "trsm failed in LeastSquaresCUDA");

        MemoryManager::Free(workspace, device);
        MemoryManager::Free(tau, device);
        MemoryManager::Free(dinfo, device);
    });
}

}  // namespace core
}  // namespace open3d
