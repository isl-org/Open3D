// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SVD.h"

namespace open3d {
namespace core {

void SVDCPU(const void* A_data,
            void* U_data,
            void* S_data,
            void* VT_data,
            void* superb_data,
            int64_t m,
            int64_t n,
            Dtype dtype,
            const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
                gesvd_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, 'A', 'A', m, n,
                        const_cast<scalar_t*>(
                                static_cast<const scalar_t*>(A_data)),
                        m, static_cast<scalar_t*>(S_data),
                        static_cast<scalar_t*>(U_data), m,
                        static_cast<scalar_t*>(VT_data), n,
                        static_cast<scalar_t*>(superb_data)),
                "gesvd failed in SVDCPU");
    });
}

}  // namespace core
}  // namespace open3d
