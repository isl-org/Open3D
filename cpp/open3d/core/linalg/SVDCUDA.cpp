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

#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SVD.h"

namespace open3d {
namespace core {

void SVDCUDA(const void* A_data,
             void* U_data,
             void* S_data,
             void* VT_data,
             void* superb_data,
             int64_t m,
             int64_t n,
             Dtype dtype,
             const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        int* dinfo =
                static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));
        OPEN3D_CUSOLVER_CHECK(
                gesvd_cuda_buffersize<scalar_t>(handle, m, n, &len),
                "gesvd_buffersize failed in SVDCUDA");

        void* workspace = MemoryManager::Malloc(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                gesvd_cuda<scalar_t>(
                        handle, 'A', 'A', m, n,
                        const_cast<scalar_t*>(
                                static_cast<const scalar_t*>(A_data)),
                        m, static_cast<scalar_t*>(S_data),
                        static_cast<scalar_t*>(U_data), m,
                        static_cast<scalar_t*>(VT_data), n,
                        static_cast<scalar_t*>(workspace), len,
                        static_cast<scalar_t*>(superb_data), dinfo),
                "gesvd failed in SVDCUDA", dinfo, device);

        MemoryManager::Free(workspace, device);
    });
}
}  // namespace core
}  // namespace open3d
