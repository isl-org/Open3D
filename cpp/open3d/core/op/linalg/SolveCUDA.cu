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

void SolveCUDA(void* A_data,
               void* B_data,
               void* ipiv_data,
               void* X_data,
               int n,
               int m,
               Dtype dtype,
               const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    int* dinfo = static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));

    size_t byte_size;
    int niters;

    switch (dtype) {
        case Dtype::Float32: {
            OPEN3D_CUSOLVER_CHECK(cusolverDnSSgesv_bufferSize(
                                          handle, n, m, NULL, n, NULL, NULL, n,
                                          NULL, n, NULL, &byte_size),
                                  "cusolverDnSSgesv_bufferSize failed");
            void* workspace = MemoryManager::Malloc(byte_size, device);

            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnSSgesv(handle, n, m, static_cast<float*>(A_data),
                                     n, static_cast<int*>(ipiv_data),
                                     static_cast<float*>(B_data), n,
                                     static_cast<float*>(X_data), n, workspace,
                                     byte_size, &niters, dinfo),
                    "cusolverDnSSgesv failed with dinfo = ", dinfo, device);
            break;

            MemoryManager::Free(workspace, device);
        }

        case Dtype::Float64: {
            OPEN3D_CUSOLVER_CHECK(cusolverDnDDgesv_bufferSize(
                                          handle, n, m, NULL, n, NULL, NULL, n,
                                          NULL, n, NULL, &byte_size),
                                  "cusolverDnDDgesv_bufferSize failed");
            void* workspace = MemoryManager::Malloc(byte_size, device);

            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnDDgesv(handle, n, m, static_cast<double*>(A_data),
                                     n, static_cast<int*>(ipiv_data),
                                     static_cast<double*>(B_data), n,
                                     static_cast<double*>(X_data), n, workspace,
                                     byte_size, &niters, dinfo),
                    "cusolverDnDDgesv failed with dinfo = ", dinfo, device);
            break;

            MemoryManager::Free(workspace, device);
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
