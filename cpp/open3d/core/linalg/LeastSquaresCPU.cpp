// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LeastSquares.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LeastSquaresCPU(void* A_data,
                     void* B_data,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     Dtype dtype,
                     const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
                gels_cpu<scalar_t>(LAPACK_COL_MAJOR, 'N', m, n, k,
                                   static_cast<scalar_t*>(A_data), m,
                                   static_cast<scalar_t*>(B_data),
                                   std::max(m, n)),
                "gels failed in LeastSquaresCPU");
    });
}

}  // namespace core
}  // namespace open3d
