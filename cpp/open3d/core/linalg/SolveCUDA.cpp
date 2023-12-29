// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/Solve.h"

namespace open3d {
namespace core {

// cuSolver's gesv will crash when A is a singular matrix.
// We implement LU decomposition-based solver (similar to Inverse) instead.
void SolveCUDA(void* A_data,
               void* B_data,
               void* ipiv_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device) {
    cusolverDnHandle_t handle =
            CuSolverContext::GetInstance().GetHandle(device);

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        Blob dinfo(sizeof(int), device);

        OPEN3D_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, n, n, n, &len),
                "getrf_buffersize failed in SolveCUDA");
        Blob workspace(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrf_cuda<scalar_t>(
                        handle, n, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()),
                        static_cast<int*>(ipiv_data),
                        static_cast<int*>(dinfo.GetDataPtr())),
                "getrf failed in SolveCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrs_cuda<scalar_t>(handle, CUBLAS_OP_N, n, k,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<int*>(ipiv_data),
                                     static_cast<scalar_t*>(B_data), n,
                                     static_cast<int*>(dinfo.GetDataPtr())),
                "getrs failed in SolveCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);
    });
}

}  // namespace core
}  // namespace open3d
