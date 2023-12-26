// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/Inverse.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void InverseCPU(void* A_data,
                void* ipiv_data,
                void* output_data,
                int64_t n,
                Dtype dtype,
                const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
                getrf_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, n, static_cast<scalar_t*>(A_data),
                        n, static_cast<OPEN3D_CPU_LINALG_INT*>(ipiv_data)),
                "getrf failed in InverseCPU");
        OPEN3D_LAPACK_CHECK(
                getri_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<OPEN3D_CPU_LINALG_INT*>(ipiv_data)),
                "getri failed in InverseCPU");
    });
}

}  // namespace core
}  // namespace open3d
