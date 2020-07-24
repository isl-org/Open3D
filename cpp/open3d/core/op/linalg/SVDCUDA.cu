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

void SVDCUDA(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             void* superb_data,
             int m,
             int n,
             Dtype dtype,
             const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    int* dinfo = static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));
    int len;

    switch (dtype) {
        case Dtype::Float32: {
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnSgesvd_bufferSize(handle, m, n, &len),
                    "cusolverDnSgesvd_bufferSize failed");
            void* workspace =
                    MemoryManager::Malloc(len * sizeof(float), device);

            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnSgesvd(handle, 'A', 'A', m, n,
                                     const_cast<float*>(
                                             static_cast<const float*>(A_data)),
                                     m, static_cast<float*>(S_data),
                                     static_cast<float*>(U_data), m,
                                     static_cast<float*>(VT_data), n,
                                     static_cast<float*>(workspace), len,
                                     static_cast<float*>(superb_data), dinfo),
                    "cusolverDnSgesvd failed with dinfo = ", dinfo, device);

            MemoryManager::Free(workspace, Device("CUDA:0"));
            break;
        }

        case Dtype::Float64: {
            OPEN3D_CUSOLVER_CHECK(
                    cusolverDnDgesvd_bufferSize(handle, m, n, &len),
                    "cusolverDnDgesvd_bufferSize failed");
            void* workspace =
                    MemoryManager::Malloc(len * sizeof(double), device);

            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    cusolverDnDgesvd(
                            handle, 'A', 'A', m, n,
                            const_cast<double*>(
                                    static_cast<const double*>(A_data)),
                            m, static_cast<double*>(S_data),
                            static_cast<double*>(U_data), m,
                            static_cast<double*>(VT_data), n,
                            static_cast<double*>(workspace), len,
                            static_cast<double*>(superb_data), dinfo),
                    "cusolverDnDgesvd failed with dinfo = ", dinfo, device);

            MemoryManager::Free(workspace, device);
            break;
        }

        default: {  // should never reach here
            utility::LogError("Unsupported dtype {} in SVDCUDA.",
                              DtypeUtil::ToString(dtype));
        }
    }

    MemoryManager::Free(dinfo, device);
}

}  // namespace core
}  // namespace open3d
