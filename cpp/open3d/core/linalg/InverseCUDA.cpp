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

#include "open3d/core/linalg/InverseImp.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void InverseCUDA(void* A_data,
                 void* ipiv_data,
                 void* output_data,
                 int64_t n,
                 Dtype dtype,
                 const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        int* dinfo =
                static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));

        OPEN3D_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, n, n, n, &len),
                "getrf_buffersize failed in InverseCUDA");
        void* workspace = MemoryManager::Malloc(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrf_cuda<scalar_t>(handle, n, n,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<scalar_t*>(workspace),
                                     static_cast<int*>(ipiv_data), dinfo),
                "getrf failed in InverseCUDA", dinfo, device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrs_cuda<scalar_t>(handle, CUBLAS_OP_N, n, n,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<int*>(ipiv_data),
                                     static_cast<scalar_t*>(output_data), n,
                                     dinfo),
                "getrs failed in InverseCUDA", dinfo, device);

        MemoryManager::Free(workspace, device);
        MemoryManager::Free(dinfo, device);
    });
}

}  // namespace core
}  // namespace open3d
