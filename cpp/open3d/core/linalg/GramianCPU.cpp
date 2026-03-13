// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/Gramian.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {
void GramCPU(void* A_data, void* B_data, int64_t m, int64_t n, Dtype dtype) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        gemm_cpu<scalar_t>(CblasColMajor, CblasNoTrans, CblasTrans, n, n, m,
                           alpha, static_cast<const scalar_t*>(A_data), n,
                           static_cast<const scalar_t*>(A_data), n, beta,
                           static_cast<scalar_t*>(B_data), n);
    });
}

void RowGramCPU(void* A_data, void* B_data, int64_t m, int64_t n, Dtype dtype) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        gemm_cpu<scalar_t>(CblasColMajor, CblasTrans, CblasNoTrans, m, m, n,
                           alpha, static_cast<const scalar_t*>(A_data), n,
                           static_cast<const scalar_t*>(A_data), n, beta,
                           static_cast<scalar_t*>(B_data), m);
    });
}

}  // namespace core
}  // namespace open3d
