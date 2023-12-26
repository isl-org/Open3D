// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/LUImpl.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LUCUDA(void* A_data,
            void* ipiv_data,
            int64_t rows,
            int64_t cols,
            Dtype dtype,
            const Device& device) {
    cusolverDnHandle_t handle =
            CuSolverContext::GetInstance().GetHandle(device);
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        OPEN3D_CUSOLVER_CHECK(
                getrf_cuda_buffersize<scalar_t>(handle, rows, cols, rows, &len),
                "getrf_buffersize failed in LUCUDA");

        int* dinfo =
                static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));
        void* workspace = MemoryManager::Malloc(len * sizeof(scalar_t), device);

        cusolverStatus_t getrf_status = getrf_cuda<scalar_t>(
                handle, rows, cols, static_cast<scalar_t*>(A_data), rows,
                static_cast<scalar_t*>(workspace), static_cast<int*>(ipiv_data),
                dinfo);

        MemoryManager::Free(workspace, device);
        MemoryManager::Free(dinfo, device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(getrf_status, "getrf failed in LUCUDA",
                                         dinfo, device);
    });
}

}  // namespace core
}  // namespace open3d
