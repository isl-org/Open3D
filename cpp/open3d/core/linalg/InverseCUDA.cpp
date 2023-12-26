// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Blob.h"
#include "open3d/core/linalg/Inverse.h"
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
    cusolverDnHandle_t handle =
            CuSolverContext::GetInstance().GetHandle(device);

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        Blob dinfo(sizeof(int), device);

        OPEN3D_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, n, n, n, &len),
                "getrf_buffersize failed in InverseCUDA");
        Blob workspace(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrf_cuda<scalar_t>(
                        handle, n, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()),
                        static_cast<int*>(ipiv_data),
                        static_cast<int*>(dinfo.GetDataPtr())),
                "getrf failed in InverseCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                getrs_cuda<scalar_t>(handle, CUBLAS_OP_N, n, n,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<int*>(ipiv_data),
                                     static_cast<scalar_t*>(output_data), n,
                                     static_cast<int*>(dinfo.GetDataPtr())),
                "getrs failed in InverseCUDA",
                static_cast<int*>(dinfo.GetDataPtr()), device);
    });
}

}  // namespace core
}  // namespace open3d
